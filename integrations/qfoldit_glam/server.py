#!/usr/bin/env python3
"""Loopback-only qFoldIT adapter for GLAM's existing adaptive property search.

This adapter intentionally invokes only the fixed GLAM property-search entry
point. It does not expose arbitrary command execution or scientific authority.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_VERSION = "0.1.0"
REQUEST_CONTRACT = "qfoldit.glam-adapter.job-request/0.1"
TASK_ID = "adaptive-property-search"

PROPERTY_DATASETS = {
    "demo",
    "esol",
    "freesolv",
    "lipophilicity",
    "physprop_perturb",
    "bbbp",
    "bace",
    "sider",
    "toxcast",
    "tox21",
}
PARAMETER_DEFAULTS = {
    "n_init_configs": 5,
    "n_low_fidelity_seed": 1,
    "n_top_blend": 2,
    "n_high_fidelity_seed": 1,
    "seed": 1234,
    "split_seed": 1234,
}
PARAMETER_BOUNDS = {
    "n_init_configs": (1, 20),
    "n_low_fidelity_seed": (1, 3),
    "n_top_blend": (1, 5),
    "n_high_fidelity_seed": (1, 3),
    "seed": (0, 2_147_483_647),
    "split_seed": (0, 2_147_483_647),
}
MAX_REQUEST_BYTES = 16 * 1024
MAX_CONCURRENT_JOBS = 1
MAX_LOG_TAIL_CHARS = 8000
DEFAULT_JOB_TIMEOUT_SECONDS = 6 * 60 * 60

_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_string(value: Any, field: str, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise RequestError(400, f"{field} must be a string")
    value = value.strip()
    if not value or len(value) > maximum:
        raise RequestError(400, f"{field} must contain 1..{maximum} characters")
    return value


def normalize_request(payload: Any) -> dict[str, Any]:
    """Validate a request without touching files or launching a process."""
    if not isinstance(payload, dict):
        raise RequestError(400, "request body must be a JSON object")

    allowed = {
        "contract",
        "task",
        "mission_id",
        "mission_version",
        "dataset",
        "dataset_revision",
        "parameters",
    }
    unexpected = set(payload) - allowed
    if unexpected:
        raise RequestError(400, "unsupported request fields: " + ", ".join(sorted(unexpected)))

    required = {
        "contract",
        "task",
        "mission_id",
        "mission_version",
        "dataset",
        "dataset_revision",
    }
    missing = required - set(payload)
    if missing:
        raise RequestError(400, "missing required fields: " + ", ".join(sorted(missing)))

    if payload["contract"] != REQUEST_CONTRACT:
        raise RequestError(400, f"contract must equal {REQUEST_CONTRACT}")
    if payload["task"] != TASK_ID:
        raise RequestError(400, f"unsupported task; expected {TASK_ID}")

    mission_id = _require_string(payload["mission_id"], "mission_id", 128)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", mission_id):
        raise RequestError(400, "mission_id contains unsupported characters")

    mission_version = _require_string(payload["mission_version"], "mission_version", 64)
    dataset = _require_string(payload["dataset"], "dataset", 64).lower()
    if dataset not in PROPERTY_DATASETS:
        raise RequestError(400, "dataset is not in the adapter's allowlist")

    dataset_revision = _require_string(payload["dataset_revision"], "dataset_revision", 256)

    raw_parameters = payload.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        raise RequestError(400, "parameters must be a JSON object")
    unknown_parameters = set(raw_parameters) - set(PARAMETER_DEFAULTS)
    if unknown_parameters:
        raise RequestError(
            400,
            "unsupported parameters: " + ", ".join(sorted(unknown_parameters)),
        )

    parameters: dict[str, int] = dict(PARAMETER_DEFAULTS)
    for name, value in raw_parameters.items():
        minimum, maximum = PARAMETER_BOUNDS[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise RequestError(400, f"parameters.{name} must be an integer")
        if not minimum <= value <= maximum:
            raise RequestError(
                400,
                f"parameters.{name} must be between {minimum} and {maximum}",
            )
        parameters[name] = value

    return {
        "contract": REQUEST_CONTRACT,
        "task": TASK_ID,
        "mission_id": mission_id,
        "mission_version": mission_version,
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "parameters": parameters,
    }


def resolve_dataset_root(dataset: str) -> Path:
    """Resolve paths from server configuration, never from request input."""
    if dataset == "demo":
        return (ROOT / "src_1gp" / "demo").resolve()
    configured = os.environ.get("GLAM_GP_DATASET_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (ROOT / "Dataset" / "GLAM-GP").resolve()


def build_command(
    request: dict[str, Any],
    dataset_root: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build argv as a list; no shell interpolation is used."""
    parameters = request["parameters"]
    return [
        python_executable,
        "glam.py",
        "--dataset",
        request["dataset"],
        "--dataset_root",
        str(dataset_root),
        "--n_init_configs",
        str(parameters["n_init_configs"]),
        "--n_low_fidelity_seed",
        str(parameters["n_low_fidelity_seed"]),
        "--n_top_blend",
        str(parameters["n_top_blend"]),
        "--n_high_fidelity_seed",
        str(parameters["n_high_fidelity_seed"]),
        "--seed",
        str(parameters["seed"]),
        "--split_seed",
        str(parameters["split_seed"]),
    ]


def _timeout_seconds() -> int:
    try:
        raw = int(os.environ.get("QFOLDIT_GLAM_JOB_TIMEOUT_SECONDS", DEFAULT_JOB_TIMEOUT_SECONDS))
    except ValueError:
        return DEFAULT_JOB_TIMEOUT_SECONDS
    return min(max(raw, 60), 7 * 24 * 60 * 60)


def _job_root() -> Path:
    configured = os.environ.get("QFOLDIT_GLAM_JOB_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "qfoldit-glam-jobs").resolve()


def _persist_state(job_dir: Path, state: dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = job_dir / "state.json"
    temporary = job_dir / "state.json.tmp"
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    if os.name != "nt":
        try:
            job_dir.chmod(0o700)
            destination.chmod(0o600)
        except OSError:
            pass


def _read_log_tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - MAX_LOG_TAIL_CHARS * 4), os.SEEK_SET)
            content = stream.read().decode("utf-8", errors="replace")
        return content[-MAX_LOG_TAIL_CHARS:]
    except FileNotFoundError:
        return ""


def _public_state(state: dict[str, Any], log_path: Path | None = None) -> dict[str, Any]:
    result = dict(state)
    if log_path is not None:
        result["log_tail"] = _read_log_tail(log_path)
    result["scientific_validation_status"] = "NOT_PERFORMED"
    result["evidence_status"] = "NOT_CREATED"
    return result


def _watch_job(job_id: str, process: subprocess.Popen[Any], job_dir: Path) -> None:
    try:
        exit_code = process.wait(timeout=_timeout_seconds())
        status = "SUCCEEDED" if exit_code == 0 else "FAILED"
        error = None if exit_code == 0 else f"GLAM process exited with code {exit_code}"
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        exit_code = process.returncode
        status = "TIMED_OUT"
        error = "GLAM process exceeded the configured job timeout"

    with _JOB_LOCK:
        record = _JOBS.get(job_id)
        if record is None:
            return
        state = record["state"]
        state.update(
            {
                "status": status,
                "exit_code": exit_code,
                "error": error,
                "completed_at": utc_now(),
            }
        )
        _persist_state(job_dir, state)


def _start_job(request: dict[str, Any]) -> dict[str, Any]:
    script_dir = ROOT / "src_1gp"
    script = script_dir / "glam.py"
    if not script.is_file():
        raise RequestError(503, "GLAM property-search entry point is missing")

    dataset_root = resolve_dataset_root(request["dataset"])
    if not dataset_root.is_dir():
        raise RequestError(
            503,
            "configured dataset root is unavailable; prepare the dataset and set GLAM_GP_DATASET_ROOT",
        )

    job_id = str(uuid.uuid4())
    job_dir = _job_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    log_path = job_dir / "run.log"
    command = build_command(request, dataset_root)

    with _JOB_LOCK:
        active = sum(1 for item in _JOBS.values() if item["state"]["status"] == "RUNNING")
        if active >= MAX_CONCURRENT_JOBS:
            raise RequestError(409, "adapter concurrency limit reached")

        environment = os.environ.copy()
        environment.pop("QFOLDIT_GLAM_TOKEN", None)
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            with log_path.open("ab") as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=script_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    close_fds=True,
                    env=environment,
                )
        except OSError as exc:
            raise RequestError(500, f"could not start GLAM: {exc.__class__.__name__}") from exc

        state = {
            "job_id": job_id,
            "contract": "qfoldit.glam-adapter.job-response/0.1",
            "task": request["task"],
            "mission_id": request["mission_id"],
            "mission_version": request["mission_version"],
            "dataset": request["dataset"],
            "dataset_revision": request["dataset_revision"],
            "parameters": request["parameters"],
            "status": "RUNNING",
            "created_at": utc_now(),
            "started_at": utc_now(),
            "completed_at": None,
            "exit_code": None,
            "error": None,
            "scientific_validation_status": "NOT_PERFORMED",
            "evidence_status": "NOT_CREATED",
        }
        _JOBS[job_id] = {"state": state, "process": process, "log_path": log_path}
        _persist_state(job_dir, state)
        threading.Thread(
            target=_watch_job,
            args=(job_id, process, job_dir),
            name=f"glam-job-{job_id[:8]}",
            daemon=True,
        ).start()

    return _public_state(state, log_path)


def _load_state(job_id: str) -> tuple[dict[str, Any], Path] | None:
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", job_id):
        return None
    with _JOB_LOCK:
        record = _JOBS.get(job_id)
        if record is not None:
            return record["state"], record["log_path"]

    job_dir = _job_root() / job_id
    state_path = job_dir / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return state, job_dir / "run.log"


class Handler(BaseHTTPRequestHandler):
    server_version = "qFoldIT-GLAM-Adapter/" + ADAPTER_VERSION

    def log_message(self, format_string: str, *args: Any) -> None:
        # Avoid logging request headers, tokens, or full payloads.
        sys.stderr.write("%s - %s\n" % (self.address_string(), format_string % args))

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = os.environ.get("QFOLDIT_GLAM_TOKEN")
        if not expected:
            self._send_json(
                503,
                {
                    "error": "job endpoints are disabled until QFOLDIT_GLAM_TOKEN is configured",
                    "execution_enabled": False,
                },
            )
            return False
        supplied = self.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, "Bearer " + expected):
            self._send_json(401, {"error": "bearer token required"})
            return False
        return True

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "adapter_id": "qfoldit.glam",
                    "adapter_version": ADAPTER_VERSION,
                    "execution_enabled": bool(os.environ.get("QFOLDIT_GLAM_TOKEN")),
                    "bind": "127.0.0.1",
                },
            )
            return
        if path == "/v1/capabilities":
            self._send_json(
                200,
                {
                    "adapter_id": "qfoldit.glam",
                    "adapter_version": ADAPTER_VERSION,
                    "contract": REQUEST_CONTRACT,
                    "capabilities": [
                        {
                            "id": TASK_ID,
                            "status": "EXPERIMENTAL",
                            "description": "Run GLAM's existing adaptive molecular-property training/model-selection pipeline.",
                            "datasets": sorted(PROPERTY_DATASETS),
                            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
                            "scientific_validation": "external-required",
                        }
                    ],
                    "not_supported": [
                        "direct molecular inference API",
                        "drug-target interaction run orchestration",
                        "drug-drug interaction run orchestration",
                        "Evidence 1.1 publication",
                    ],
                },
            )
            return
        prefix = "/v1/jobs/"
        if path.startswith(prefix):
            if not self._authorized():
                return
            job_id = path[len(prefix):]
            loaded = _load_state(job_id)
            if loaded is None:
                self._send_json(404, {"error": "job not found"})
                return
            state, log_path = loaded
            self._send_json(200, _public_state(state, log_path))
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path != "/v1/jobs":
            self._send_json(404, {"error": "not found"})
            return
        if not self._authorized():
            return

        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._send_json(415, {"error": "Content-Type must be application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json(413 if length > MAX_REQUEST_BYTES else 400, {"error": "request body size is invalid"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            request = normalize_request(payload)
            state = _start_job(request)
        except RequestError as exc:
            self._send_json(exc.status, {"error": exc.message})
            return
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "request body must be valid UTF-8 JSON"})
            return

        self._send_json(202, state)


def main() -> int:
    try:
        port = int(os.environ.get("QFOLDIT_GLAM_PORT", "8787"))
    except ValueError:
        print("QFOLDIT_GLAM_PORT must be an integer", file=sys.stderr)
        return 2
    if not 1 <= port <= 65535:
        print("QFOLDIT_GLAM_PORT must be between 1 and 65535", file=sys.stderr)
        return 2

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    print(f"qFoldIT GLAM adapter listening on http://127.0.0.1:{port}")
    print("Execution is disabled unless QFOLDIT_GLAM_TOKEN is configured.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping qFoldIT GLAM adapter.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
