"""VLM-VAD Studio: a local Flask GUI for datasets, training and validation."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from werkzeug.exceptions import HTTPException

from . import data_catalog, runs, training
from .config import (
    CHECKPOINT_ROOT,
    OFFICIAL_LABEL_ROOT,
    PROJECT_ROOT,
    RESULTS_ROOT,
    ensure_runtime_dirs,
    sanitize_run_name,
)
from .jobs import TERMINAL, manager

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024
app.config["JSON_AS_ASCII"] = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_error(message: str, status: int = 400) -> tuple[Response, int]:
    return jsonify({"error": message}), status


def _device_info() -> dict[str, Any]:
    info = {"device": "cpu", "cuda": False, "gpu_name": None, "torch": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["device"] = "cuda" if info["cuda"] else "cpu"
        if info["cuda"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Pages and health
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status=204)


@app.get("/api/health")
def health() -> Response:
    device = _device_info()
    return jsonify({
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "time": time.time(),
        "device": device,
        "active_job": manager.active_summary(),
        "dataset_count": len(data_catalog.list_datasets()),
        "run_count": len(runs.list_runs()),
        "official_labels": (OFFICIAL_LABEL_ROOT / "testing" / "manifest.json").is_file(),
    })


@app.get("/api/options")
def options() -> Response:
    return jsonify({
        "prompt_types": ["label", "scene", "contrast"],
        "fusions": ["concat", "gated", "crossattn"],
        "temporals": ["identity", "transformer"],
        "devices": ["auto", "cpu", "cuda"],
        "label_modes": ["motion_diff", "pixel"],
        "use_feature": ["auto", "true", "false"],
    })


@app.get("/api/overview")
def overview() -> Response:
    dataset_rows = data_catalog.list_datasets()
    run_rows = runs.list_runs()
    valid_rows = [row for row in run_rows if row.get("benchmark_valid")]
    pool = valid_rows or run_rows
    best = None
    for row in pool:
        auc = (row.get("metrics") or {}).get("frame_auc")
        if auc is None:
            continue
        if best is None or auc > (best.get("metrics") or {}).get("frame_auc", -1):
            best = row
    best_scope = "official" if any(best is row for row in valid_rows) else "all"
    return jsonify({
        "datasets": dataset_rows,
        "runs": run_rows,
        "best_run": best,
        "best_scope": best_scope,
        "recent_runs": run_rows[:6],
        "active_job": manager.active_summary(),
        "official_labels": (OFFICIAL_LABEL_ROOT / "testing" / "manifest.json").is_file(),
    })


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@app.get("/api/datasets")
def api_datasets() -> Response:
    return jsonify({"datasets": data_catalog.list_datasets(force=True)})


@app.get("/api/datasets/detail")
def api_dataset_detail() -> Response:
    dataset_id = request.args.get("id", "")
    return jsonify(data_catalog.dataset_detail(dataset_id))


@app.get("/api/datasets/frame")
def api_dataset_frame() -> Response:
    data = data_catalog.read_frame(
        dataset_id=request.args.get("id", ""),
        split=request.args.get("split", "training"),
        video_id=request.args.get("video", ""),
        index=int(request.args.get("index", 0)),
        width=int(request.args.get("w", 640)),
    )
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})


@app.post("/api/datasets/upload")
def api_upload() -> Response:
    files = request.files.getlist("files") or request.files.getlist("file")
    raw_paths = request.form.get("paths", "[]")
    try:
        rel_paths = json.loads(raw_paths) if raw_paths else []
    except json.JSONDecodeError:
        rel_paths = []
    if not isinstance(rel_paths, list):
        rel_paths = []
    name = (request.form.get("name") or "").strip()
    if not name and files:
        first = str(files[0].filename or "")
        if first.lower().endswith(".zip"):
            name = Path(first).stem
    summary = data_catalog.upload_dataset(files, [str(item) for item in rel_paths], name)
    return jsonify({"dataset": summary})


@app.post("/api/datasets/register")
def api_register() -> Response:
    payload = request.get_json(silent=True) or {}
    summary = data_catalog.register_dataset(str(payload.get("path", "")))
    return jsonify({"dataset": summary})


@app.post("/api/datasets/prepare")
def api_prepare() -> Response:
    payload = request.get_json(silent=True) or {}
    dataset_id = str(payload.get("id", ""))
    labels = bool(payload.get("labels", True))
    features = bool(payload.get("features", False))
    device = str(payload.get("device", "auto"))
    command = data_catalog.prepare_command(dataset_id, labels=labels, features=features, device=device)
    run_name = sanitize_run_name(("prep_" + dataset_id.replace("/", "_"))[:64])
    job = manager.start(
        kind="prepare",
        run_name=run_name,
        command=command,
        meta={"dataset_id": dataset_id, "labels": labels, "features": features, "device": device},
    )
    return jsonify({"job": job.snapshot(include_log=False)})


# ---------------------------------------------------------------------------
# Runs and files
# ---------------------------------------------------------------------------

@app.get("/api/runs")
def api_runs() -> Response:
    return jsonify({"runs": runs.list_runs()})


@app.get("/api/runs/detail")
def api_run_detail() -> Response:
    run_id = request.args.get("id", "")
    return jsonify(runs.run_detail(run_id))


@app.get("/api/results/file")
def api_result_file() -> Response:
    path = runs.result_file(request.args.get("path", ""))
    return send_file(path, conditional=True, max_age=0)


@app.get("/api/compare")
def api_compare() -> Response:
    ids = [item.strip() for item in request.args.get("ids", "").split(",") if item.strip()]
    return jsonify({"runs": runs.compare_runs(ids)})


# ---------------------------------------------------------------------------
# Training / evaluation / zero-shot
# ---------------------------------------------------------------------------

@app.post("/api/train")
def api_train() -> Response:
    payload = request.get_json(silent=True) or {}
    built = training.build_training_config(payload)
    job = manager.start(
        kind="train",
        run_name=built["run_name"],
        command=built["command"],
        meta={
            "dataset_id": payload.get("dataset_id"),
            "config": built["config_rel"],
            "epochs": built["epochs"],
            "prompt_types": built["prompt_types"],
            "fusion": built["fusion"],
            "temporal": built["temporal"],
        },
    )
    return jsonify({"job": job.snapshot(include_log=False), "run": built})


@app.post("/api/runs/evaluate")
def api_evaluate() -> Response:
    payload = request.get_json(silent=True) or {}
    built = training.build_eval_config(str(payload.get("id", "")), str(payload.get("mode", "motion")))
    job = manager.start(
        kind="eval",
        run_name=built["result_id"],
        command=built["command"],
        meta={
            "source_run": built["run_id"],
            "result_run": built["result_id"],
            "mode": built["mode"],
            "config": built["config_rel"],
            "checkpoint": built["checkpoint"],
        },
    )
    return jsonify({"job": job.snapshot(include_log=False), "eval": built})


@app.post("/api/runs/demo")
def api_demo() -> Response:
    payload = request.get_json(silent=True) or {}
    built = training.build_demo_job(str(payload.get("id", "")), list(payload.get("videos", [])))
    job = manager.start(
        kind="demo",
        run_name=built["result_id"],
        command=built["command"],
        meta={
            "source_run": built["run_id"],
            "result_run": built["result_id"],
            "videos": built["videos"],
            "config": built["config_rel"],
            "checkpoint": built["checkpoint"],
        },
    )
    return jsonify({"job": job.snapshot(include_log=False), "demo": built})


@app.post("/api/zero-shot")
def api_zero_shot() -> Response:
    payload = request.get_json(silent=True) or {}
    built = training.build_zero_shot_job(
        str(payload.get("prompt", "scene")),
        official=bool(payload.get("official", True)),
    )
    job = manager.start(
        kind="zero_shot",
        run_name=built["result_id"],
        command=built["command"],
        meta={"prompt": built["prompt"], "config": built["config"]},
    )
    return jsonify({"job": job.snapshot(include_log=False), "zero_shot": built})


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@app.get("/api/jobs")
def api_jobs() -> Response:
    return jsonify({"jobs": manager.list(), "active": manager.active_summary()})


@app.get("/api/jobs/<job_id>")
def api_job(job_id: str) -> Response:
    job = manager.get(job_id)
    if job is None:
        return _json_error("Job not found", 404)
    return jsonify(job.snapshot())


@app.get("/api/jobs/<job_id>/events")
def api_job_events(job_id: str) -> Response:
    def generate():
        while True:
            job = manager.get(job_id)
            if job is None:
                yield "data: " + json.dumps({"error": "Job not found"}) + "\n\n"
                break
            snapshot = job.snapshot()
            yield "data: " + json.dumps(snapshot, ensure_ascii=False) + "\n\n"
            if snapshot["status"] in TERMINAL:
                break
            time.sleep(0.8)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/jobs/<job_id>/log")
def api_job_log(job_id: str) -> Response:
    try:
        text = manager.tail(job_id, int(request.args.get("tail", 300)))
    except FileNotFoundError:
        return _json_error("Job not found", 404)
    return Response(text, mimetype="text/plain")


@app.post("/api/jobs/<job_id>/stop")
def api_job_stop(job_id: str) -> Response:
    try:
        return jsonify({"job": manager.stop(job_id)})
    except FileNotFoundError:
        return _json_error("Job not found", 404)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(ValueError)
def handle_value_error(error: ValueError):
    return _json_error(str(error), 400)


@app.errorhandler(FileNotFoundError)
def handle_missing(error: FileNotFoundError):
    return _json_error(str(error), 404)


@app.errorhandler(RuntimeError)
def handle_runtime(error: RuntimeError):
    return _json_error(str(error), 409)


@app.errorhandler(413)
def handle_too_large(error):
    return _json_error("Upload is larger than the 8 GB limit", 413)


@app.errorhandler(Exception)
def handle_unexpected(error: Exception):
    if isinstance(error, HTTPException):
        return error
    traceback.print_exc()
    return _json_error("Internal error: " + str(error), 500)


def main() -> None:
    ensure_runtime_dirs()
    parser = argparse.ArgumentParser(description="VLM-VAD Studio web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
