#!/usr/bin/env python3
"""qFoldIT GLAM adapter: scoped execution, durable status and hashed run manifests."""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
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
ADAPTER_VERSION = "0.2.0"
REQUEST_CONTRACT = "qfoldit.glam-adapter.job-request/0.2"
RESPONSE_CONTRACT = "qfoldit.glam-adapter.job-response/0.2"
RUN_RESULT_CONTRACT = "qfoldit.glam.run-result/1.0"
TASK_ID = "adaptive-property-search"

PROPERTY_DATASETS = {
    "demo", "esol", "freesolv", "lipophilicity", "physprop_perturb",
    "bbbp", "bace", "sider", "toxcast", "tox21",
}
PARAMETER_DEFAULTS = {
    "n_init_configs": 5,
    "n_low_fidelity_seed": 1,
    "n_top_blend": 2,
    "n_high_fidelity_seed": 1,
    "seed": 1234,
    "split_seed": 1234,
    "search_seed": 1234,
}
PARAMETER_BOUNDS = {
    "n_init_configs": (1, 20),
    "n_low_fidelity_seed": (1, 3),
    "n_top_blend": (1, 5),
    "n_high_fidelity_seed": (1, 3),
    "seed": (0, 2_147_483_647),
    "split_seed": (0, 2_147_483_647),
    "search_seed": (0, 2_147_483_647),
}
MAX_REQUEST_BYTES = 16 * 1024
MAX_CONCURRENT_JOBS = 1
MAX_LOG_TAIL_CHARS = 8000
DEFAULT_JOB_TIMEOUT_SECONDS = 6 * 60 * 60
REQUIRED_RUNTIME_MODULES = ("torch", "torch_geometric", "torch_scatter", "rdkit", "pandas", "numpy")

_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def require_string(value: Any, field: str, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise RequestError(400, f"{field} must be a string")
    value = value.strip()
    if not value or len(value) > maximum:
        raise RequestError(400, f"{field} must contain 1..{maximum} characters")
    return value


def normalize_request(payload: Any) -> dict[str, Any]:
    """Validate the versioned mission request without touching files or launching work."""
    if not isinstance(payload, dict):
        raise RequestError(400, "request body must be a JSON object")

    allowed = {
        "contract", "task", "mission_id", "mission_version", "submission_id",
        "action_envelope_hash", "tenant_id", "dataset", "dataset_revision",
        "split", "parameters",
    }
    unexpected = set(payload) - allowed
    if unexpected:
        raise RequestError(400, "unsupported request fields: " + ", ".join(sorted(unexpected)))

    required = {
        "contract", "task", "mission_id", "mission_version", "submission_id",
        "action_envelope_hash", "dataset", "dataset_revision",
    }
    missing = required - set(payload)
    if missing:
        raise RequestError(400, "missing required fields: " + ", ".join(sorted(missing)))

    if payload["contract"] != REQUEST_CONTRACT:
        raise RequestError(400, f"contract must equal {REQUEST_CONTRACT}")
    if payload["task"] != TASK_ID:
        raise RequestError(400, f"unsupported task; expected {TASK_ID}")

    mission_id = require_string(payload["mission_id"], "mission_id", 128)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", mission_id):
        raise RequestError(400, "mission_id contains unsupported characters")
    mission_version = require_string(payload["mission_version"], "mission_version", 64)

    submission_id = require_string(payload["submission_id"], "submission_id", 36)
    try:
        submission_id = str(uuid.UUID(submission_id))
    except ValueError as exc:
        raise RequestError(400, "submission_id must be a UUID") from exc

    action_envelope_hash = require_string(payload["action_envelope_hash"], "action_envelope_hash", 71)
    if not re.fullmatch(r"sha256:[A-Fa-f0-9]{64}", action_envelope_hash):
        raise RequestError(400, "action_envelope_hash must use sha256:<64-hex>")

    tenant_id = payload.get("tenant_id")
    if tenant_id is not None:
        tenant_id = require_string(tenant_id, "tenant_id", 128)

    dataset = require_string(payload["dataset"], "dataset", 64).lower()
    if dataset not in PROPERTY_DATASETS:
        raise RequestError(400, "dataset is not in the adapter's allowlist")
    dataset_revision = require_string(payload["dataset_revision"], "dataset_revision", 256)

    split = payload.get("split", "scaffold")
    if split not in {"scaffold", "random"}:
        raise RequestError(400, "split must be scaffold or random")

    raw_parameters = payload.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        raise RequestError(400, "parameters must be a JSON object")
    unknown_parameters = set(raw_parameters) - set(PARAMETER_DEFAULTS)
    if unknown_parameters:
        raise RequestError(400, "unsupported parameters: " + ", ".join(sorted(unknown_parameters)))

    parameters: dict[str, int] = dict(PARAMETER_DEFAULTS)
    for name, value in raw_parameters.items():
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
        "submission_id": submission_id,
        "action_envelope_hash": action_envelope_hash,
        "tenant_id": tenant_id,
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "split": split,
        "parameters": parameters,
    }


def resolve_dataset_root(dataset: str) -> Path:
    """Resolve the dataset path from trusted server configuration, never request input."""
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
    """Build an argument vector for the fixed GLAM entry point; shell execution is disabled."""
    parameters = request["parameters"]
    return [
        python_executable,
        "glam.py",
        "--dataset", request["dataset"],
        "--dataset_root", str(dataset_root),
        "--n_init_configs", str(parameters["n_init_configs"]),
        "--n_low_fidelity_seed", str(parameters["n_low_fidelity_seed"]),
        "--n_top_blend", str(parameters["n_top_blend"]),
        "--n_high_fidelity_seed", str(parameters["n_high_fidelity_seed"]),
        "--seed", str(parameters["seed"]),
        "--split_seed", str(parameters["split_seed"]),
        "--search_seed", str(parameters["search_seed"]),
        "--split", request["split"],
    ]


def build_run_manifest(state: dict[str, Any], job_dir: Path) -> dict[str, Any]:
    """Hash the actual outputs and bind them to the mission, data and execution environment."""
    output_root = job_dir / "outputs"
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root).as_posix()
        artifacts.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "kind": "model-checkpoint" if path.suffix == ".ckpt" else "run-output",
        })

    if not any(
        item["kind"] == "model-checkpoint" and item["path"].endswith("best_save.ckpt")
        for item in artifacts
    ):
        raise RuntimeError("GLAM completed without producing a best_save.ckpt model checkpoint")

    manifest = {
        "contract": RUN_RESULT_CONTRACT,
        "job_id": state["job_id"],
        "mission_id": state["mission_id"],
        "mission_version": state["mission_version"],
        "submission_id": state["submission_id"],
        "action_envelope_hash": state["action_envelope_hash"],
        "tenant_id": state.get("tenant_id"),
        "task": state["task"],
        "dataset": {
            "id": state["dataset"],
            "revision": state["dataset_revision"],
            "sha256": state["dataset_sha256"],
        },
        "execution": {
            "adapter_version": ADAPTER_VERSION,
            "git_commit": os.environ.get("GLAM_GIT_COMMIT"),
            "container_digest": os.environ.get("QFOLDIT_GLAM_CONTAINER_DIGEST"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                "torch": distribution_version("torch"),
                "torch-geometric": distribution_version("torch-geometric"),
                "torch-scatter": distribution_version("torch-scatter"),
                "rdkit": distribution_version("rdkit"),
                "numpy": distribution_version("numpy"),
                "pandas": distribution_version("pandas"),
                "scikit-learn": distribution_version("scikit-learn"),
            },
        },
        "model_search": {
            "split": state["split"],
            "parameters": state["parameters"],
        },
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "created_at": utc_now(),
        "scientific_validation_status": "NOT_PERFORMED",
        "evidence_status": "NOT_CREATED",
    }

    manifest_path = job_dir / "run_manifest.json"
    encoded = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest_bytes = encoded + b"\n"
    manifest_path.write_bytes(manifest_bytes)
    return {
        **manifest,
        "run_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def timeout_seconds() -> int:
    try:
        raw = int(os.environ.get("QFOLDIT_GLAM_JOB_TIMEOUT_SECONDS", 6 * 60 * 60))
    except ValueError:
        return 6 * 60 * 60
    return min(max(raw, 60), 7 * 24 * 60 * 60)


def job_root() -> Path:
    configured = os.environ.get("QFOLDIT_GLAM_JOB_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "qfoldit-glam-jobs").resolve()


def persist_state(job_dir: Path, state: dict[str, Any]) -> None:
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


def read_log_tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - MAX_LOG_TAIL_CHARS * 4), os.SEEK_SET)
            content = stream.read().decode("utf-8", errors="replace")
        return content[-MAX_LOG_TAIL_CHARS:]
    except FileNotFoundError:
        return ""


def public_state(state: dict[str, Any], log_path: Path | None = None) -> dict[str, Any]:
    result = dict(state)
    if log_path is not None:
        result["log_tail"] = read_log_tail(log_path)
    result["scientific_validation_status"] = "NOT_PERFORMED"
    result["evidence_status"] = "NOT_CREATED"
    return result


def watch_job(job_id: str, process: subprocess.Popen[Any], job_dir: Path) -> None:
    manifest_data = None
    try:
        exit_code = process.wait(timeout=timeout_seconds())
        status = "SUCCEEDED" if exit_code == 0 else "FAILED"
        error = None if exit_code == 0 else f"GLAM process exited with code {exit_code}"
        if status == "SUCCEEDED":
            with _JOB_LOCK:
                snapshot = dict(_JOBS[job_id]["state"])
            try:
                manifest_data = build_run_manifest(snapshot, job_dir)
            except Exception as exc:
                status = "FAILED"
                error = (
                    "GLAM exited successfully but artifact finalization failed: "
                    f"{exc.__class__.__name__}: {exc}"
                )
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
        state.update({
            "status": status,
            "exit_code": exit_code,
            "error": error,
            "completed_at": utc_now(),
            "artifact_status": "CREATED" if manifest_data else "NOT_CREATED",
            "artifact_count": manifest_data["artifact_count"] if manifest_data else 0,
            "run_manifest_sha256": manifest_data["run_manifest_sha256"] if manifest_data else None,
        })
        persist_state(job_dir, state)


def start_job(request: dict[str, Any]) -> dict[str, Any]:
    script_dir = ROOT / "src_1gp"
    if not (script_dir / "glam.py").is_file():
        raise RequestError(503, "GLAM property-search entry point is missing")

    dataset_root = resolve_dataset_root(request["dataset"])
    if not dataset_root.is_dir():
        raise RequestError(
            503,
            "configured dataset root is unavailable; prepare it and set GLAM_GP_DATASET_ROOT",
        )
    dataset_file = dataset_root / "raw" / f"{request['dataset']}.csv"
    if not dataset_file.is_file():
        raise RequestError(503, "configured dataset CSV is missing from the server-side dataset root")
    dataset_sha256 = sha256_file(dataset_file)

    job_id = str(uuid.uuid4())
    current_job_root = job_root()
    current_job_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    job_dir = current_job_root / job_id
    job_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    output_root = job_dir / "outputs"
    output_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    log_path = job_dir / "run.log"
    command = build_command(request, dataset_root)

    with _JOB_LOCK:
        active = sum(
            1 for item in _JOBS.values() if item["state"]["status"] == "RUNNING"
        )
        if active >= MAX_CONCURRENT_JOBS:
            raise RequestError(409, "adapter concurrency limit reached")

        environment = os.environ.copy()
        environment.pop("QFOLDIT_GLAM_TOKEN", None)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["QFOLDIT_GLAM_RUN_ROOT"] = str(output_root)
        environment["GLAM_GP_DATASET_ROOT"] = str(dataset_root)

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
            "contract": RESPONSE_CONTRACT,
            "task": request["task"],
            "mission_id": request["mission_id"],
            "mission_version": request["mission_version"],
            "submission_id": request["submission_id"],
            "action_envelope_hash": request["action_envelope_hash"],
            "tenant_id": request.get("tenant_id"),
            "dataset": request["dataset"],
            "dataset_revision": request["dataset_revision"],
            "dataset_sha256": dataset_sha256,
            "split": request["split"],
            "parameters": request["parameters"],
            "status": "RUNNING",
            "created_at": utc_now(),
            "started_at": utc_now(),
            "completed_at": None,
            "exit_code": None,
            "error": None,
            "artifact_status": "PENDING",
            "artifact_count": 0,
            "run_manifest_sha256": None,
            "scientific_validation_status": "NOT_PERFORMED",
            "evidence_status": "NOT_CREATED",
        }
        _JOBS[job_id] = {"state": state, "process": process, "log_path": log_path}
        persist_state(job_dir, state)
        threading.Thread(
            target=watch_job,
            args=(job_id, process, job_dir),
            name=f"glam-job-{job_id[:8]}",
            daemon=True,
        ).start()

    return public_state(state, log_path)


def load_state(job_id: str) -> tuple[dict[str, Any], Path] | None:
    try:
        canonical_id = str(uuid.UUID(job_id))
    except (ValueError, AttributeError):
        return None

    with _JOB_LOCK:
        record = _JOBS.get(canonical_id)
        if record is not None:
            return record["state"], record["log_path"]

    directory = job_root() / canonical_id
    state_path = directory / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # A RUNNING record without an in-memory process belongs to a previous adapter
    # process. Its terminal outcome is unknown and must never be inferred as success.
    if state.get("status") == "RUNNING":
        state.update({
            "status": "INTERRUPTED",
            "error": "adapter restarted while the job was running; outcome requires operator review",
            "completed_at": utc_now(),
            "artifact_status": "UNKNOWN",
        })
        persist_state(directory, state)
    return state, directory / "run.log"


def readiness_payload() -> dict[str, Any]:
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in REQUIRED_RUNTIME_MODULES
    }
    current_job_root = job_root()
    writable = True
    try:
        current_job_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        probe = current_job_root / ".qfoldit-ready-probe"
        probe.write_text("ready", encoding="utf-8")
        probe.unlink()
    except OSError:
        writable = False
    token_configured = bool(os.environ.get("QFOLDIT_GLAM_TOKEN"))
    ready = token_configured and all(dependencies.values()) and writable
    return {
        "ready": ready,
        "adapter_id": "qfoldit.glam",
        "adapter_version": ADAPTER_VERSION,
        "execution_enabled": token_configured,
        "dependencies": dependencies,
        "job_root_writable": writable,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "qFoldIT-GLAM-Adapter/" + ADAPTER_VERSION

    def log_message(self, format_string: str, *args: Any) -> None:
        # Do not log request bodies, bearer tokens, SMILES or dataset contents.
        sys.stderr.write("%s - %s\n" % (self.address_string(), format_string % args))

    def send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        expected = os.environ.get("QFOLDIT_GLAM_TOKEN")
        if not expected:
            self.send_json(503, {
                "error": "job endpoints are disabled until QFOLDIT_GLAM_TOKEN is configured",
                "execution_enabled": False,
            })
            return False
        supplied = self.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, "Bearer " + expected):
            self.send_json(401, {"error": "bearer token required"})
            return False
        return True

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path in {"/health", "/healthz"}:
            self.send_json(200, {
                "status": "ok",
                "service": "qfoldit-glam-adapter",
                "adapter_id": "qfoldit.glam",
                "adapter_version": ADAPTER_VERSION,
                "execution_enabled": bool(os.environ.get("QFOLDIT_GLAM_TOKEN")),
            })
            return
        if path == "/readyz":
            payload = readiness_payload()
            self.send_json(200 if payload["ready"] else 503, payload)
            return
        if path == "/v1/version":
            self.send_json(200, {
                "service": "qfoldit-glam-adapter",
                "adapterVersion": ADAPTER_VERSION,
                "requestContract": REQUEST_CONTRACT,
                "responseContract": RESPONSE_CONTRACT,
                "runResultContract": RUN_RESULT_CONTRACT,
            })
            return
        if path == "/v1/capabilities":
            self.send_json(200, {
                "adapter_id": "qfoldit.glam",
                "adapter_version": ADAPTER_VERSION,
                "contract": REQUEST_CONTRACT,
                "capabilities": [{
                    "id": TASK_ID,
                    "status": "EXPERIMENTAL",
                    "description": "Run GLAM adaptive molecular-property training/model selection and emit hashed run artifacts.",
                    "datasets": sorted(PROPERTY_DATASETS),
                    "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
                    "scientific_validation": "external-required",
                }],
                "not_supported": [
                    "direct per-molecule inference endpoint",
                    "drug-target interaction run orchestration",
                    "drug-drug interaction run orchestration",
                    "independent scientific validation",
                    "Evidence 1.1 publication",
                ],
            })
            return

        prefix = "/v1/jobs/"
        if path.startswith(prefix) and path.endswith("/manifest"):
            if not self.authorized():
                return
            job_id = path[len(prefix):-len("/manifest")]
            loaded = load_state(job_id)
            if loaded is None:
                self.send_json(404, {"error": "job not found"})
                return
            state, _ = loaded
            manifest_path = job_root() / str(uuid.UUID(job_id)) / "run_manifest.json"
            if not manifest_path.is_file():
                self.send_json(409, {
                    "error": "run manifest is not available",
                    "status": state.get("status"),
                })
                return
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.send_json(500, {"error": "run manifest could not be read"})
                return
            self.send_json(200, manifest)
            return

        if path.startswith(prefix):
            if not self.authorized():
                return
            job_id = path[len(prefix):]
            loaded = load_state(job_id)
            if loaded is None:
                self.send_json(404, {"error": "job not found"})
                return
            state, log_path = loaded
            self.send_json(200, public_state(state, log_path))
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path != "/v1/jobs":
            self.send_json(404, {"error": "not found"})
            return
        if not self.authorized():
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self.send_json(415, {"error": "Content-Type must be application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "invalid Content-Length"})
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self.send_json(413 if length > MAX_REQUEST_BYTES else 400, {"error": "request body size is invalid"})
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise RequestError(400, "request body was truncated")
            payload = json.loads(body.decode("utf-8"))
            request = normalize_request(payload)
            state = start_job(request)
        except RequestError as exc:
            self.send_json(exc.status, {"error": exc.message})
            return
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {"error": "request body must be valid UTF-8 JSON"})
            return
        self.send_json(202, state)


def main() -> int:
    try:
        port = int(os.environ.get("QFOLDIT_GLAM_PORT", "8787"))
    except ValueError:
        print("QFOLDIT_GLAM_PORT must be an integer", file=sys.stderr)
        return 2
    if not 1 <= port <= 65535:
        print("QFOLDIT_GLAM_PORT must be between 1 and 65535", file=sys.stderr)
        return 2

    host = os.environ.get("QFOLDIT_GLAM_BIND_HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(f"qFoldIT GLAM adapter listening on http://{host}:{port}")
    print("Use only behind an authenticated private service gateway for remote deployments.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping qFoldIT GLAM adapter.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
