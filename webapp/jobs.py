"""Background job manager for training, evaluation, zero-shot and preparation."""
from __future__ import annotations

import copy
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, WEB_JOB_LOG_ROOT, ensure_runtime_dirs, sanitize_run_name

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TQDM_RE = re.compile(r"Epoch\s+(\d+)\s+\[(train|val)\]:\s+(\d+)%")
TRAIN_RE = re.compile(
    r"Epoch\s+(\d+)\s+train\s*\|\s*loss=([0-9.eE+-]+)\s+bce=([0-9.eE+-]+)\s+"
    r"contrastive=([0-9.eE+-]+)\s+lr=([0-9.eE+-]+)"
)
VAL_RE = re.compile(
    r"Epoch\s+(\d+)\s+val\s*\|\s*frame_auc=([0-9.eE+-]+)\s+frame_ap=([0-9.eE+-]+)\s+"
    r"clip_auc=([0-9.eE+-]+)\s+clip_ap=([0-9.eE+-]+)"
)
EVAL_RE = re.compile(
    r"Eval\s*\|\s*frame_auc=(\S+)\s+frame_ap=(\S+)\s+clip_auc=(\S+)\s+clip_ap=(\S+)"
    r"(?:\s+video_auc=(\S+))?"
)
ZERO_RE = re.compile(
    r"Zero-shot\s*\|\s*prompts\(K=(\d+)[^|]*\|\s*frame_auc=(\S+)\s+frame_ap=(\S+)"
)
ERROR_RE = re.compile(
    r"(Traceback \(most recent call last\)|RuntimeError|FileNotFoundError|"
    r"ValueError|KeyError|AssertionError|CUDA out of memory|OutOfMemoryError|"
    r"ModuleNotFoundError|ImportError)"
)
TERMINAL = {"completed", "failed", "stopped"}


def _number(value: str) -> float | None:
    text = (value or "").strip()
    if text.lower() in {"none", "null", "nan", "undefined", ""}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


@dataclass
class Job:
    id: str
    kind: str
    run_name: str
    command: list[str]
    log_path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    pid: int | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    exit_code: int | None = None
    error: str | None = None
    stop_requested: bool = False
    progress: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    log_lines: list[str] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def snapshot(self, include_log: bool = True) -> dict[str, Any]:
        with self._lock:
            elapsed = (self.ended_at or time.time()) - (self.started_at or self.created_at)
            payload = {
                "id": self.id,
                "kind": self.kind,
                "run_name": self.run_name,
                "status": self.status,
                "pid": self.pid,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "elapsed_sec": round(max(0.0, elapsed), 1),
                "exit_code": self.exit_code,
                "error": self.error,
                "progress": copy.deepcopy(self.progress),
                "metrics": copy.deepcopy(self.metrics),
                "meta": copy.deepcopy(self.meta),
                "command": self.command,
                "log_path": str(self.log_path),
            }
            if include_log:
                payload["log_tail"] = list(self.log_lines[-120:])
            return payload


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        ensure_runtime_dirs()

    # -- lifecycle ---------------------------------------------------------
    def start(
        self,
        kind: str,
        run_name: str,
        command: list[str],
        meta: dict[str, Any] | None = None,
    ) -> Job:
        clean_run = sanitize_run_name(run_name)
        with self._lock:
            active = self.active()
            if active is not None:
                raise RuntimeError(
                    "Another job is already running: " + active.id + " (" + active.kind + ")."
                )
            job_id = uuid.uuid4().hex[:12]
            safe_kind = re.sub(r"[^A-Za-z0-9_-]", "_", kind)[:24] or "job"
            log_path = WEB_JOB_LOG_ROOT / (job_id + "_" + safe_kind + "_" + clean_run + ".log")
            job = Job(
                id=job_id,
                kind=kind,
                run_name=clean_run,
                command=[str(part) for part in command],
                log_path=log_path,
                meta=dict(meta or {}),
            )
            self._jobs[job_id] = job

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("MPLCONFIGDIR", "/tmp/devhome/.config/matplotlib")

        try:
            with open(log_path, "wb") as log_file:
                process = subprocess.Popen(
                    job.command,
                    cwd=str(PROJECT_ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
        except Exception as exc:
            with job._lock:
                job.status = "failed"
                job.error = str(exc)
                job.ended_at = time.time()
            raise

        with job._lock:
            job.pid = process.pid
            job.status = "running"
            job.started_at = time.time()
            job.meta.setdefault("total_epochs", None)

        thread = threading.Thread(
            target=self._monitor, args=(job, process), name="job-" + job.id, daemon=True
        )
        thread.start()
        return job

    def _read_new(self, path: Path, offset: int) -> tuple[bytes, int]:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return b"", offset
        if size <= offset:
            return b"", offset
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(size - offset)
        return data, offset + len(data)

    def _monitor(self, job: Job, process: subprocess.Popen) -> None:
        offset = 0
        pending = ""
        while True:
            data, offset = self._read_new(job.log_path, offset)
            if data:
                pending += data.decode("utf-8", errors="replace")
                parts = re.split(r"[\r\n]", pending)
                pending = parts.pop()
                for line in parts:
                    self._parse(job, line)
                if len(pending) > 4000:
                    self._parse(job, pending)
                    pending = ""

            code = process.poll()
            if code is not None:
                data, offset = self._read_new(job.log_path, offset)
                if data:
                    pending += data.decode("utf-8", errors="replace")
                for line in re.split(r"[\r\n]", pending):
                    self._parse(job, line)
                with job._lock:
                    job.exit_code = int(code)
                    job.ended_at = time.time()
                    if job.stop_requested:
                        job.status = "stopped"
                    elif code == 0:
                        job.status = "completed"
                    else:
                        job.status = "failed"
                        if not job.error:
                            job.error = "Process exited with code " + str(code)
                return
            time.sleep(0.35)

    # -- parsing -----------------------------------------------------------
    def _parse(self, job: Job, raw_line: str) -> None:
        line = ANSI_RE.sub("", raw_line).strip()
        if not line:
            return
        with job._lock:
            job.log_lines.append(line)
            if len(job.log_lines) > 600:
                job.log_lines = job.log_lines[-600:]

            match = TQDM_RE.search(line)
            if match:
                epoch, phase, percent = match.groups()
                job.progress.update({
                    "epoch": int(epoch),
                    "phase": phase,
                    "percent": int(percent),
                })
                return

            match = TRAIN_RE.search(line)
            if match:
                epoch, loss, bce, contrastive, lr = match.groups()
                entry = self._history_entry(job, int(epoch))
                entry["train"] = {
                    "loss": float(loss),
                    "bce": float(bce),
                    "contrastive": float(contrastive),
                    "lr": float(lr),
                }
                job.metrics["train"] = dict(entry["train"])
                job.progress.update({"epoch": int(epoch), "phase": "train", "percent": 100})
                return

            match = VAL_RE.search(line)
            if match:
                epoch, frame_auc, frame_ap, clip_auc, clip_ap = match.groups()
                entry = self._history_entry(job, int(epoch))
                entry["val"] = {
                    "frame_auc": _number(frame_auc),
                    "frame_ap": _number(frame_ap),
                    "clip_auc": _number(clip_auc),
                    "clip_ap": _number(clip_ap),
                }
                job.metrics["val"] = dict(entry["val"])
                job.progress.update({"epoch": int(epoch), "phase": "val", "percent": 100})
                return

            match = EVAL_RE.search(line)
            if match:
                frame_auc, frame_ap, clip_auc, clip_ap, video_auc = match.groups()
                job.metrics["test"] = {
                    "frame_auc": _number(frame_auc),
                    "frame_ap": _number(frame_ap),
                    "clip_auc": _number(clip_auc),
                    "clip_ap": _number(clip_ap),
                    "video_auc": _number(video_auc),
                }
                job.progress.update({"phase": "test", "percent": 100})
                return

            match = ZERO_RE.search(line)
            if match:
                count, frame_auc, frame_ap = match.groups()
                job.metrics["zero_shot"] = {
                    "num_prompts": int(count),
                    "frame_auc": _number(frame_auc),
                    "frame_ap": _number(frame_ap),
                }
                job.progress.update({"phase": "test", "percent": 100})
                return

            if "Checkpoint saved to" in line:
                job.progress["checkpoint"] = line.split("Checkpoint saved to", 1)[1].strip()
                return
            if "Results saved to" in line:
                job.progress["result_dir"] = line.split("Results saved to", 1)[1].strip()
                return
            if line.startswith("[prep]"):
                job.progress.update({"phase": "prepare", "message": line[6:].strip()})
                return
            if "Training finished" in line:
                job.progress.update({"phase": "finished", "message": line})
                return
            if ERROR_RE.search(line):
                job.error = line[:500]

    def _history_entry(self, job: Job, epoch: int) -> dict[str, Any]:
        history = job.metrics.setdefault("history", [])
        for item in history:
            if int(item.get("epoch", -1)) == epoch:
                return item
        item: dict[str, Any] = {"epoch": epoch}
        history.append(item)
        history.sort(key=lambda value: int(value.get("epoch", 0)))
        return item

    # -- public ------------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
        return [job.snapshot(include_log=False) for job in jobs]

    def active(self) -> Job | None:
        with self._lock:
            for job in self._jobs.values():
                if job.status in {"queued", "running", "stopping"}:
                    return job
        return None

    def active_summary(self) -> dict[str, Any] | None:
        job = self.active()
        return job.snapshot(include_log=False) if job else None

    def stop(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Job not found: " + job_id)
        with job._lock:
            if job.status in TERMINAL:
                return job.snapshot()
            job.stop_requested = True
            job.status = "stopping"
            pid = job.pid
        if pid is None:
            return job.snapshot()
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        threading.Thread(target=self._force_kill, args=(pid,), daemon=True).start()
        return job.snapshot()

    def _force_kill(self, pid: int) -> None:
        time.sleep(5.0)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    def tail(self, job_id: str, lines: int = 200) -> str:
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError("Job not found: " + job_id)
        lines = max(1, min(int(lines), 2000))
        try:
            text = job.log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""
        chunks = re.split(r"[\r\n]", text)
        return "\n".join(chunk for chunk in chunks[-lines:] if chunk.strip())


manager = JobManager()
