"""Exercise first-use image selection without a daemon, registry, or GPU."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


DOCKER_STUB = r'''#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["VLM_TEST_CALLS"], "a") as log:
    log.write(json.dumps(args) + "\n")
if args[0] == "info":
    print(os.environ.get("VLM_TEST_DAEMON_PLATFORM", "linux/x86_64"))
    sys.exit(int(os.environ.get("VLM_TEST_DAEMON_EXIT", "0")))
if args[:2] == ["image", "inspect"]:
    platform = os.environ.get("VLM_TEST_CACHED_PLATFORM", "")
    if platform:
        print(platform)
        sys.exit(0)
    sys.exit(1)
if args[0] == "compose":
    if "--environment" in args:
        print(os.environ.get("VLM_TEST_COMPOSE_ENV", ""))
    elif "--images" in args:
        gpu = any(arg.endswith("docker-compose.gpu.yml") for arg in args)
        print(os.environ.get("VLM_TEST_IMAGE", "zgzdocker/vlm-ws:" + ("cuda12.1" if gpu else "cpu")))
    else:
        sys.exit("Unexpected Compose operation: " + repr(args))
    sys.exit(0)
if args[0] in ("pull", "build"):
    sys.exit(int(os.environ.get("VLM_TEST_" + args[0].upper() + "_EXIT", "0")))
sys.exit("Unexpected Docker operation: " + repr(args))
'''


@pytest.fixture
def run_dev(tmp_path):
    root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace with spaces"
    (workspace / "scripts").mkdir(parents=True)
    shutil.copyfile(root / "scripts/dev.sh", workspace / "scripts/dev.sh")
    shutil.copyfile(root / ".env.example", workspace / ".env.example")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(DOCKER_STUB)
    docker.chmod(0o755)
    uname = binaries / "uname"
    uname.write_text('#!/bin/sh\nprintf "%s\\n" "${VLM_TEST_HOST_ARCH:-x86_64}"\n')
    uname.chmod(0o755)
    calls_path = tmp_path / "docker-calls.jsonl"

    def run(mode="cpu", action="ensure-image", args=(), **overrides):
        calls_path.write_text("")
        env = os.environ.copy()
        for key in ("DOCKER_DEFAULT_PLATFORM", "LOCAL_UID", "LOCAL_GID", "VLM_HOST_TOOLS"):
            env.pop(key, None)
        env.update(PATH=str(binaries) + os.pathsep + env["PATH"], VLM_TEST_CALLS=str(calls_path))
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(workspace / "scripts/dev.sh"), mode, action, *args],
            env=env, capture_output=True, text=True, timeout=30,
        )
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
        return result, calls

    return run


@pytest.mark.parametrize("mode,daemon,cached", [
    ("cpu", "linux/x86_64", "linux/amd64"),
    ("cpu", "linux/aarch64", "linux/arm64"),
    ("gpu", "linux/x86_64", "linux/amd64"),
])
def test_cached_image_opens_without_network_or_build(run_dev, mode, daemon, cached):
    result, calls = run_dev(
        mode, VLM_TEST_DAEMON_PLATFORM=daemon, VLM_TEST_CACHED_PLATFORM=cached,
        VLM_TEST_PULL_EXIT="1", VLM_TEST_BUILD_EXIT="1",
    )
    assert result.returncode == 0, result.stderr
    assert "Using local image" in result.stdout
    assert all(call[0] not in ("pull", "build") for call in calls)


@pytest.mark.parametrize("mode,daemon,cached,platform,tag", [
    ("cpu", "linux/x86_64", "", "linux/amd64", "cpu"),
    ("cpu", "linux/aarch64", "", "linux/arm64", "cpu"),
    ("gpu", "linux/x86_64", "", "linux/amd64", "cuda12.1"),
    ("cpu", "linux/x86_64", "linux/arm64", "linux/amd64", "cpu"),
    ("cpu", "linux/aarch64", "linux/amd64", "linux/arm64", "cpu"),
])
def test_missing_or_wrong_architecture_image_is_pulled(run_dev, mode, daemon, cached, platform, tag):
    result, calls = run_dev(mode, VLM_TEST_DAEMON_PLATFORM=daemon, VLM_TEST_CACHED_PLATFORM=cached)
    assert result.returncode == 0, result.stderr
    assert [call for call in calls if call[0] == "pull"] == [
        ["pull", "--platform", platform, "zgzdocker/vlm-ws:" + tag],
    ]
    assert all(call[0] != "build" for call in calls)


@pytest.mark.parametrize("mode,tag,base,index", [
    ("cpu", "cpu", "python:3.10-slim-bookworm", "cpu"),
    ("gpu", "cuda12.1", "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime", "cu121"),
])
def test_pull_failure_builds_the_selected_backend(run_dev, mode, tag, base, index):
    result, calls = run_dev(mode, VLM_TEST_PULL_EXIT="1")
    assert result.returncode == 0, result.stderr
    assert "Pull failed" in result.stderr
    pull_index = next(i for i, call in enumerate(calls) if call[0] == "pull")
    build_index = next(i for i, call in enumerate(calls) if call[0] == "build")
    assert pull_index < build_index
    build = calls[build_index]
    assert build[1:3] == ["--tag", "zgzdocker/vlm-ws:" + tag]
    assert "BASE_IMAGE=" + base in build
    assert "TORCH_INDEX_URL=https://download.pytorch.org/whl/" + index in build
    assert "INSTALL_OPTIONAL_SHELL_TOOLS=0" in build
    assert build[-3:-1] == ["--platform", "linux/amd64"]
    assert Path(build[-1]).name == "workspace with spaces"


def test_fallback_build_failure_stops_initialization(run_dev):
    result, calls = run_dev(VLM_TEST_PULL_EXIT="1", VLM_TEST_BUILD_EXIT="19")
    assert result.returncode == 19
    assert calls[-1][0] == "build"


def test_daemon_failure_is_reported_without_pull_or_build(run_dev):
    result, calls = run_dev(VLM_TEST_DAEMON_EXIT="37")
    assert result.returncode == 37
    assert [call[0] for call in calls] == ["info"]


def test_digest_pinned_image_is_not_replaced_by_a_local_build(run_dev):
    result, calls = run_dev(VLM_TEST_PULL_EXIT="1", VLM_TEST_IMAGE="example/vlm@sha256:" + "a" * 64)
    assert result.returncode == 2
    assert "digest-pinned image cannot be replaced" in result.stderr
    assert any(call[0] == "pull" for call in calls)
    assert all(call[0] != "build" for call in calls)


@pytest.mark.parametrize("mode,platform", [("cpu", "linux/arm64"), ("gpu", "linux/amd64")])
def test_platform_override_is_respected_by_cpu_and_cuda_stays_amd64(run_dev, mode, platform):
    result, calls = run_dev(mode, DOCKER_DEFAULT_PLATFORM="linux/arm64/v8")
    assert result.returncode == 0, result.stderr
    pull = next(call for call in calls if call[0] == "pull")
    assert pull[1:3] == ["--platform", platform]


def test_arm64_fallback_build_uses_arm64(run_dev):
    result, calls = run_dev(VLM_TEST_DAEMON_PLATFORM="linux/aarch64", VLM_TEST_PULL_EXIT="1")
    assert result.returncode == 0, result.stderr
    assert calls[-1][0] == "build"
    assert calls[-1][-3:-1] == ["--platform", "linux/arm64"]


def test_gpu_on_arm_host_is_rejected_before_docker_operations(run_dev):
    result, calls = run_dev("gpu", VLM_TEST_HOST_ARCH="aarch64")
    assert result.returncode == 2
    assert "use cpu on ARM64" in result.stderr
    assert calls == []


def test_explicit_build_keeps_custom_base_tools_and_build_arguments(run_dev):
    result, calls = run_dev(
        action="build", args=("--no-cache",), VLM_TEST_IMAGE="local/vlm:custom",
        VLM_TEST_COMPOSE_ENV="CPU_BASE_IMAGE=python:3.10-bookworm\nINSTALL_OPTIONAL_SHELL_TOOLS=1",
    )
    assert result.returncode == 0, result.stderr
    build = calls[-1]
    assert build[:3] == ["build", "--tag", "local/vlm:custom"]
    assert "BASE_IMAGE=python:3.10-bookworm" in build
    assert "INSTALL_OPTIONAL_SHELL_TOOLS=1" in build
    assert "--no-cache" in build
    assert all(call[0] not in ("info", "image", "pull") for call in calls)
