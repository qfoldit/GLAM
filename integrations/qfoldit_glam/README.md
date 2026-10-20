# qFoldIT GLAM Adapter (experimental)

This directory defines the first controlled integration slice between the GLAM research implementation and the qFoldIT Scientific Mission Control Plane.

## Scope

The current adapter exposes a loopback-only HTTP API for launching GLAM's existing adaptive molecular-property training/model-selection pipeline. It is intentionally limited to that verified execution path.

It does not yet expose per-molecule inference, drug-target interaction screening, drug-drug interaction prediction, a scientific validator, or Evidence 1.1 publication. A process exiting successfully means only that the GLAM command completed; it is not a scientifically accepted result.

## API

- GET /health — service status and whether job execution is enabled.
- GET /v1/capabilities — declared task, datasets and run bounds.
- POST /v1/jobs — validate and start an adaptive property-search run.
- GET /v1/jobs/{job_id} — retrieve status and a bounded log tail.

Job creation and job inspection require a bearer token provided through QFOLDIT_GLAM_TOKEN. Without this variable the service remains read-only. The server binds to 127.0.0.1 only; do not expose it directly to a network.

## Run locally

From the repository root, install and verify the GLAM research dependencies as documented in the root README. Set GLAM_GP_DATASET_ROOT to a prepared GLAM-GP dataset directory, except when using the bundled demo dataset. Configure a strong local token:

    export GLAM_GP_DATASET_ROOT="$PWD/Dataset/GLAM-GP"
    export QFOLDIT_GLAM_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    python integrations/qfoldit_glam/server.py

Windows PowerShell equivalent:

    $env:GLAM_GP_DATASET_ROOT = "$PWD\Dataset\GLAM-GP"
    $env:QFOLDIT_GLAM_TOKEN = [guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")
    python integrations/qfoldit_glam/server.py

The bundled demo uses src_1gp/demo and does not require GLAM_GP_DATASET_ROOT, but it still requires the legacy GLAM dependencies.

## Example job request

    {
      "contract": "qfoldit.glam-adapter.job-request/0.1",
      "task": "adaptive-property-search",
      "mission_id": "property-benchmark-001",
      "mission_version": "1.0.0",
      "dataset": "demo",
      "dataset_revision": "bundled-demo",
      "parameters": {
        "n_init_configs": 1,
        "n_low_fidelity_seed": 1,
        "n_top_blend": 1,
        "n_high_fidelity_seed": 1,
        "seed": 1234,
        "split_seed": 1234
      }
    }

For example, to submit the request locally, save it as request.json and use:

    curl -X POST http://127.0.0.1:8787/v1/jobs \
      -H "Authorization: Bearer $QFOLDIT_GLAM_TOKEN" \
      -H "Content-Type: application/json" \
      --data-binary @request.json

The API accepts dataset identifiers, bounded numeric parameters and mission metadata only. It never accepts shell commands, script paths, arbitrary Python expressions, uploaded code, or user-selected filesystem paths.

## Data and provenance

For non-demo datasets, the adapter uses the server-configured GLAM_GP_DATASET_ROOT. Callers cannot override the path in a job request. dataset_revision is recorded as caller-supplied provenance metadata; this adapter does not independently attest the content of the dataset. A future ingestion service should calculate and attest content digests.

The adapter persists job status and logs under the user cache directory by default, or QFOLDIT_GLAM_JOB_ROOT if set. These artifacts may contain operational details; protect them using host-level access controls.

## qFoldIT integration boundary

qFoldIT Mission Contracts route work to this adapter through a Scientific MCP or control-plane service. UEFN/Verse is a presentation/runtime surface and must not call GLAM directly. GLAM's output is a model-run artifact, not a validator decision. An approved independent validator must evaluate candidate submissions before any Evidence record, scientific attribution or commercial settlement is accepted.

## Verification

Run the adapter's standard-library tests:

    python -m py_compile integrations/qfoldit_glam/server.py
    python -m unittest discover -s integrations/qfoldit_glam -p "test_*.py"

The tests exercise request bounds and command construction; they do not install GLAM's PyTorch/RDKit dependencies or demonstrate a successful scientific run.

## Maturity

Status: experimental integration scaffold. Compatibility with current Python/PyTorch/PyG/RDKit releases, dataset-specific behavior, benchmark performance, and production deployment require separate verification.
