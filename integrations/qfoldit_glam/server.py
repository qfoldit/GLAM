#!/usr/bin/env python3
"""qFoldIT GLAM worker adapter.

The adapter exposes only a bounded training/model-selection task. It records
process completion and reproducible run artifacts, but never self-issues
scientific acceptance or Evidence 1.1.
"""
from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ID = "qfoldit.glam"
ADAPTER_VERSION = "0.3.0"
REQUEST_CONTRACT = "qfoldit.glam-adapter.job-request/0.3"
RESPONSE_CONTRACT = "qfoldit.glam-adapter.job-response/0.3"
RUN_RESULT_CONTRACT = "qfoldit.glam.run-result/1.1"
TASK_ID = "adaptive-property-search"

PROPERTY_DATASETS = {
    "demo", "esol", "freesolv", "lipophilicity", "physprop_perturb",
    "bbbp", "bace", "sider", "toxcast", "tox21",
}
PARAMETER_DEFAULTS: dict[str, Any] = {
    "n_init_configs": 5,
    "n_low_fidelity_seed": 1,
    "n_top_blend": 2,
    "n_high_fidelity_seed": 1,
    "seed": 1234,
    "split_seed": 1234,
    "split": "scaffold",
}
PARAMETER_BOUNDS = {
    "n_init_configs": (1, 20),
    "n_low_fidelity_seed": (1, 3),
    "n_top_blend": (1, 5),
    "n_high_fidelity_seed": (1, 3),
    "seed": (0, 2_147_483_647),
    "split_seed": (0, 2_147_483_647),
}
ALLOWED_SPLITS = {"scaffold", "random"}
MAX_REQUEST_BYTES = 16 * 1024
MAX_CONCURRENT_JOBS = 1
MAX_LOG_TAIL_CHARS = 8000
DEFAULT_JOB_TIMEOUT_SECONDS = 6 * 60 * 60
MAX_ARTIFACT_COUNT = 5000

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
    if not value or len(value) > maximum or any(ord(ch) < 32 for ch in value):
        raise RequestError(400, f"{field} must contain 1..{maximum} printable characters")
    return value


def normalize_request(payload: Any) -> dict[str, Any]:
    """Validate a request without touching files or launching a process."""
    if not isinstance(payload, dict):
        raise RequestError(400, "request body must be a JSON object")

    allowed = {
        "contract", "task", "mission_id", "mission_version", "mission_inception_sha256", "submission_id",
        "action_envelope_hash", "tenant_id", "dataset", "dataset_revision",
        "parameters",
    }
    unexpected = set(payload) - allowed
    if unexpected:
        raise RequestError(400, "unsupported request fields: " + ", ".join(sorted(unexpected)))
    required = {
        "contract", "task", "mission_id", "mission_version", "mission_inception_sha256", "submission_id",
        "action_envelope_hash", "dataset", "dataset_revision",
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
    mission_inception_sha256 = _require_string(payload["mission_inception_sha256"], "mission_inception_sha256", 71)
    if not re.fullmatch(r"sha256:[A-Fa-f0-9]{64}", mission_inception_sha256):
        raise RequestError(400, "mission_inception_sha256 must use sha256:<64-hex>")

    submission_raw = _require_string(payload["submission_id"], "submission_id", 36)
    try:
        submission_id = str(uuid.UUID(submission_raw))
    except ValueError as exc:
        raise RequestError(400, "submission_id must be a UUID") from exc

    action_hash = _require_string(payload["action_envelope_hash"], "action_envelope_hash", 71)
    if not re.fullmatch(r"sha256:[A-Fa-f0-9]{64}", action_hash):
        raise RequestError(400, "action_envelope_hash must use sha256:<64-hex>")

    tenant_id = payload.get("tenant_id")
    if tenant_id is not None:
        tenant_id = _require_string(tenant_id, "tenant_id", 128)

    dataset = _require_string(payload["dataset"], "dataset", 64).lower()
    if dataset not in PROPERTY_DATASETS:
        raise RequestError(400, "dataset is not in the adapter allowlist")
    dataset_revision = _require_string(payload["dataset_revision"], "dataset_revision", 256)

    raw_parameters = payload.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        raise RequestError(400, "parameters must be a JSON object")
    unknown_parameters = set(raw_parameters) - set(PARAMETER_DEFAULTS)
    if unknown_parameters:
        raise RequestError(400, "unsupported parameters: " + ", ".join(sorted(unknown_parameters)))

    parameters: dict[str, Any] = dict(PARAMETER_DEFAULTS)
    for name, value in raw_parameters.items():
        if name == "split":
            if not isinstance(value, str) or value not in ALLOWED_SPLITS:
                raise RequestError(400, "parameters.split must be scaffold or random")
            parameters[name] = value
            continue
        minimum, maximum = PARAMETER_BOUNDS[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise RequestError(400, f"parameters.{name} must be an integer")
        if not minimum <= value <= maximum:
            raise RequestError(400, f"parameters.{name} must be between {minimum} and {maximum}")
        parameters[name] = value

    return {
        "contract": REQUEST_CONTRACT,
        "task": TASK_ID,
        "mission_id": mission_id,
        "mission_version": mission_version,
        "mission_inception_sha256": mission_inception_sha256,
        "submission_id": submission_id,
        "action_envelope_hash": action_hash,
        "tenant_id": tenant_id,
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "parameters": parameters,
    }


def resolve_dataset_root(dataset: str) -> Path:
    """Resolve datasets from server configuration; callers cannot supply paths."""
    if dataset == "demo":
        return (ROOT / "src_1gp" / "demo").resolve()
    configured = os.environ.get("GLAM_GP_DATASET_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (ROOT / "Dataset" / "GLAM-GP").resolve()


def resolve_dataset_file(dataset_root: Path, dataset: str) -> Path:
    candidates = (
        dataset_root / "raw" / f"{dataset}.csv",
        dataset_root / f"{dataset}.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RequestError(503, f"raw dataset file for {dataset!r} is unavailable in the configured dataset root")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_command(
    request: dict[str, Any],
    dataset_root: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    """Build a fixed argv vector. Never pass caller input to a shell."""
    parameters = request["parameters"]
    return [
        python_executable, "glam.py",
        "--dataset", request["dataset"],
        "--dataset_root", str(dataset_root),
        "--n_init_configs", str(parameters["n_init_configs"]),
        "--n_low_fidelity_seed", str(parameters["n_low_fidelity_seed"]),
        "--n_top_blend", str(parameters["n_top_blend"]),
        "--n_high_fidelity_seed", str(parameters["n_high_fidelity_seed"]),
        "--seed", str(parameters["seed"]),
        "--split_seed", str(parameters["split_seed"]),
        "--split", str(parameters["split"]),
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
            return stream.read().decode("utf-8", errors="replace")[-MAX_LOG_TAIL_CHARS:]
    except FileNotFoundError:
        return ""


def _public_state(state: dict[str, Any], log_path: Path | None = None) -> dict[str, Any]:
    result = dict(state)
    if log_path is not None:
        result["log_tail"] = _read_log_tail(log_path)
    return result


def _runtime_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    configured = os.environ.get("QFOLDIT_GIT_COMMIT")
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=3, check=False,
        )
        value = result.stdout.strip()
        return value if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _artifact_records(output_dir: Path) -> list[dict[str, Any]]:
    if not output_dir.is_dir():
        raise RuntimeError("GLAM output directory was not created")
    records: list[dict[str, Any]] = []
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and not path.name.endswith(".tmp"))
    if len(files) > MAX_ARTIFACT_COUNT:
        raise RuntimeError(f"GLAM produced too many artifacts ({len(files)})")
    for path in files:
        records.append({
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    required = {"inf_ckpt_selected.csv", "inference_log.txt", "search_result.csv"}
    names = {Path(item["path"]).name for item in records}
    if not any(Path(item["path"]).name == "best_save.ckpt" for item in records):
        raise RuntimeError("GLAM did not produce a high-fidelity best_save.ckpt")
    missing = required - names
    if missing:
        raise RuntimeError("GLAM did not complete model selection/inference artifacts: " + ", ".join(sorted(missing)))
    return records


def _write_run_result(job_dir: Path, state: dict[str, Any], output_dir: Path) -> tuple[str, int]:
    artifacts = _artifact_records(output_dir)
    run_result = {
        "contract": RUN_RESULT_CONTRACT,
        "run_id": state["job_id"],
        "adapter": {"id": ADAPTER_ID, "version": ADAPTER_VERSION},
        "engine": {
            "name": "GLAM",
            "source_commit": _git_commit(),
            "python_version": sys.version.split()[0],
            "torch_version": _runtime_version("torch"),
            "torch_geometric_version": _runtime_version("torch-geometric"),
            "rdkit_version": _runtime_version("rdkit"),
        },
        "mission": {
            "mission_id": state["mission_id"],
            "mission_version": state["mission_version"],
            "mission_inception_sha256": state["mission_inception_sha256"],
            "submission_id": state["submission_id"],
            "action_envelope_hash": state["action_envelope_hash"],
            "tenant_id": state.get("tenant_id"),
        },
        "task": state["task"],
        "dataset": {
            "id": state["dataset"],
            "revision": state["dataset_revision"],
            "sha256": state["dataset_sha256"],
            "split": state["parameters"]["split"],
            "split_seed": state["parameters"]["split_seed"],
        },
        "parameters": state["parameters"],
        "artifacts": artifacts,
        "completed_at": state["completed_at"],
        "scientific_validation_status": "NOT_PERFORMED",
        "evidence_status": "NOT_CREATED",
    }
    encoded = json.dumps(run_result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    destination = job_dir / "run-result.json"
    destination.write_bytes(encoded)
    if os.name != "nt":
        try:
            destination.chmod(0o600)
        except OSError:
            pass
    return hashlib.sha256(encoded).hexdigest(), len(artifacts)


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _watch_job(job_id: str, process: subprocess.Popen[Any], job_dir: Path) -> None:
    try:
        exit_code = process.wait(timeout=_timeout_seconds())
        status = "SUCCEEDED" if exit_code == 0 else "FAILED"
        error = None if exit_code == 0 else f"GLAM process exited with code {exit_code}"
    except subprocess.TimeoutExpired:
        _terminate_process_tree(process)
        exit_code = process.returncode
        status = "TIMED_OUT"
        error = "GLAM process tree exceeded the configured job timeout"

    with _JOB_LOCK:
        record = _JOBS.get(job_id)
        if record is None:
            return
        state = record["state"]
        state.update({
            "status": status,
            "exit_code": exit_code,
            "error": error,
            "completed_at": utc_now(),
        })
        if status == "SUCCEEDED":
            try:
                digest, artifact_count = _write_run_result(job_dir, state, job_dir / "output")
                state["run_result_sha256"] = digest
                state["artifact_count"] = artifact_count
                state["run_result_path"] = "run-result.json"
            except Exception as exc:
                state["status"] = "FAILED"
                state["error"] = f"GLAM artifact verification failed: {exc.__class__.__name__}: {exc}"
        _persist_state(job_dir, state)


def _start_job(request: dict[str, Any]) -> dict[str, Any]:
    script_dir = ROOT / "src_1gp"
    if not (script_dir / "glam.py").is_file():
        raise RequestError(503, "GLAM property-search entry point is missing")

    dataset_root = resolve_dataset_root(request["dataset"])
    if not dataset_root.is_dir():
        raise RequestError(503, "configured dataset root is unavailable")
    dataset_file = resolve_dataset_file(dataset_root, request["dataset"])
    dataset_digest = sha256_file(dataset_file)

    job_id = str(uuid.uuid4())
    job_dir = _job_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_path = job_dir / "run.log"
    command = build_command(request, dataset_root)

    with _JOB_LOCK:
        active = sum(1 for item in _JOBS.values() if item["state"]["status"] == "RUNNING")
        if active >= MAX_CONCURRENT_JOBS:
            raise RequestError(409, "adapter concurrency limit reached")

        environment = os.environ.copy()
        environment.pop("QFOLDIT_GLAM_TOKEN", None)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["GLAM_OUTPUT_ROOT"] = str(output_dir)
        environment["GLAM_RUN_ID"] = job_id
        environment.setdefault("PYTHONHASHSEED", str(request["parameters"]["seed"]))
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
                    start_new_session=(os.name == "posix"),
                )
        except OSError as exc:
            raise RequestError(500, f"could not start GLAM: {exc.__class__.__name__}") from exc

        state = {
            "job_id": job_id,
            "contract": RESPONSE_CONTRACT,
            "task": request["task"],
            "mission_id": request["mission_id"],
            "mission_version": request["mission_version"],
            "mission_inception_sha256": request["mission_inception_sha256"],
            "submission_id": request["submission_id"],
            "action_envelope_hash": request["action_envelope_hash"],
            "tenant_id": request.get("tenant_id"),
            "dataset": request["dataset"],
            "dataset_revision": request["dataset_revision"],
            "dataset_sha256": dataset_digest,
            "parameters": request["parameters"],
            "status": "RUNNING",
            "created_at": utc_now(),
            "started_at": utc_now(),
            "completed_at": None,
            "exit_code": None,
            "error": None,
            "run_result_sha256": None,
            "artifact_count": 0,
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
    try:
        canonical_id = str(uuid.UUID(job_id))
    except (ValueError, AttributeError):
        return None
    with _JOB_LOCK:
        record = _JOBS.get(canonical_id)
        if record is not None:
            return record["state"], record["log_path"]

    job_dir = _job_root() / canonical_id
    try:
        state = json.loads((job_dir / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return state, job_dir / "run.log"


class Handler(BaseHTTPRequestHandler):
    server_version = "qFoldIT-GLAM-Adapter/" + ADAPTER_VERSION

    def log_message(self, format_string: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format_string % args))

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = os.environ.get("QFOLDIT_GLAM_TOKEN", "")
        if not expected:
            self._send_json(503, {"error": "service_not_configured", "execution_enabled": False})
            return False
        supplied = self.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, "Bearer " + expected):
            self._send_json(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path in {"/healthz", "/health"}:
            self._send_json(200, {
                "status": "ok",
                "service": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION,
                "execution_enabled": bool(os.environ.get("QFOLDIT_GLAM_TOKEN")),
                "bind": "127.0.0.1",
            })
            return
        if path == "/readyz":
            missing = [
                module for module in ("torch", "torch_geometric", "rdkit")
                if importlib.util.find_spec(module) is None
            ]
            ready = bool(os.environ.get("QFOLDIT_GLAM_TOKEN")) and not missing and (ROOT / "src_1gp" / "glam.py").is_file()
            self._send_json(200 if ready else 503, {
                "ready": ready,
                "authentication_configured": bool(os.environ.get("QFOLDIT_GLAM_TOKEN")),
                "missing_dependencies": missing,
            })
            return
        if path == "/v1/version":
            self._send_json(200, {
                "adapter_id": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION,
                "request_contract": REQUEST_CONTRACT,
                "response_contract": RESPONSE_CONTRACT,
                "run_result_contract": RUN_RESULT_CONTRACT,
                "python_version": sys.version.split()[0],
                "torch_version": _runtime_version("torch"),
                "torch_geometric_version": _runtime_version("torch-geometric"),
                "rdkit_version": _runtime_version("rdkit"),
            })
            return
        if path == "/v1/capabilities":
            self._send_json(200, {
                "adapter_id": ADAPTER_ID,
                "adapter_version": ADAPTER_VERSION,
                "request_contract": REQUEST_CONTRACT,
                "capabilities": [{
                    "id": TASK_ID,
                    "status": "EXPERIMENTAL",
                    "description": "Run GLAM adaptive molecular-property training/model selection and emit reproducible model artifacts.",
                    "datasets": sorted(PROPERTY_DATASETS),
                    "split_strategies": sorted(ALLOWED_SPLITS),
                    "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
                    "independent_validation": "external-required",
                }],
                "not_supported": [
                    "direct per-molecule inference",
                    "drug-target interaction run orchestration",
                    "drug-drug interaction run orchestration",
                    "independent scientific acceptance",
                    "Evidence 1.1 publication",
                ],
            })
            return
        job_prefix = "/v1/jobs/"
        if path.startswith(job_prefix):
            if not self._authorized():
                return
            job_id = path[len(job_prefix):]
            loaded = _load_state(job_id)
            if loaded is None:
                self._send_json(404, {"error": "job_not_found"})
                return
            state, log_path = loaded
            self._send_json(200, _public_state(state, log_path))
            return
        run_prefix = "/v1/runs/"
        if path.startswith(run_prefix) and path.endswith("/manifest"):
            if not self._authorized():
                return
            job_id = path[len(run_prefix):-len("/manifest")].strip("/")
            try:
                canonical_id = str(uuid.UUID(job_id))
            except ValueError:
                self._send_json(400, {"error": "invalid_run_id"})
                return
            manifest_path = _job_root() / canonical_id / "run-result.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._send_json(404, {"error": "run_manifest_not_found"})
                return
            self._send_json(200, manifest)
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path != "/v1/jobs":
            self._send_json(404, {"error": "not_found"})
            return
        if not self._authorized():
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._send_json(415, {"error": "content_type_must_be_application_json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "invalid_content_length"})
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json(413 if length > MAX_REQUEST_BYTES else 400, {"error": "invalid_request_size"})
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self._send_json(400, {"error": "truncated_request_body"})
            return
        try:
            request = normalize_request(json.loads(body.decode("utf-8")))
            state = _start_job(request)
        except RequestError as exc:
            self._send_json(exc.status, {"error": exc.message})
            return
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "request_must_be_utf8_json"})
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
