# qFoldIT GLAM Adapter — contract 0.2

**Adapter:** qfoldit.glam  
**Adapter version:** 0.2.0  
**Request:** qfoldit.glam-adapter.job-request/0.2  
**Response:** qfoldit.glam-adapter.job-response/0.2  
**Run result:** qfoldit.glam.run-result/1.0  
**Lifecycle:** experimental execution adapter; scientific validation remains external.

## Scope and authority

The adapter exposes only GLAM's existing adaptive molecular-property training and model-selection pipeline. It does not expose arbitrary molecule inference, DTI/DDI screening, docking, clinical interpretation, independent scientific validation or Evidence 1.1 publication.

A terminal SUCCEEDED job means:
1. the GLAM process exited with code zero;
2. a best_save.ckpt model checkpoint was produced;
3. a content-addressed run manifest was written.

It does **not** mean the model is scientifically accepted. Responses retain:
- scientific_validation_status = NOT_PERFORMED
- evidence_status = NOT_CREATED

## HTTP contract

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /healthz | process liveness |
| GET | /readyz | execution readiness: token, Python modules and writable job store |
| GET | /v1/version | adapter and contract versions |
| GET | /v1/capabilities | actual capability allowlist |
| POST | /v1/jobs | submit a bounded property-search job |
| GET | /v1/jobs/{job_id} | lifecycle, exit code and bounded log tail |
| GET | /v1/jobs/{job_id}/manifest | immutable run manifest with artifact hashes |

The service binds to 127.0.0.1 by default. Job creation and job inspection require QFOLDIT_GLAM_TOKEN. A remote deployment may set QFOLDIT_GLAM_BIND_HOST=0.0.0.0 only behind a private network boundary, TLS-terminating gateway, workload identity, tenant authorization, rate limiting and audited secret management. Do not publish this adapter directly to the internet.

## Example request

The request is tied to the canonical qFoldIT submission and action envelope:

    {
      "contract": "qfoldit.glam-adapter.job-request/0.2",
      "task": "adaptive-property-search",
      "mission_id": "property-benchmark-001",
      "mission_version": "1.0.0",
      "submission_id": "11111111-1111-4111-8111-111111111111",
      "action_envelope_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "dataset": "demo",
      "dataset_revision": "bundled-demo",
      "split": "scaffold",
      "parameters": {
        "n_init_configs": 1,
        "n_low_fidelity_seed": 1,
        "n_top_blend": 1,
        "n_high_fidelity_seed": 1,
        "seed": 1234,
        "split_seed": 1234,
        "search_seed": 4321
      }
    }

submission_id must be a UUID. action_envelope_hash must be sha256:<64-hex>. The dataset_revision label is retained for caller context; the adapter independently hashes the actual server-configured CSV.

## Local execution

Create a strong token and start the adapter:

    export QFOLDIT_GLAM_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
    export QFOLDIT_GLAM_JOB_ROOT="$PWD/.qfoldit-glam-jobs"
    python integrations/qfoldit_glam/server.py

For a non-demo dataset, configure GLAM_GP_DATASET_ROOT to a prepared GLAM-GP root containing raw/{dataset}.csv. Dataset paths are never accepted from the caller.

Submit a job:

    curl -X POST http://127.0.0.1:8787/v1/jobs \
      -H "Authorization: Bearer $QFOLDIT_GLAM_TOKEN" \
      -H "Content-Type: application/json" \
      --data-binary @request.json

Poll /v1/jobs/{job_id}. After a successful run, retrieve /v1/jobs/{job_id}/manifest.

## Reproducibility and artifact isolation

Each job receives a private output root under its job directory. The adapter sets QFOLDIT_GLAM_RUN_ROOT for the worker; the worker does not write checkpoints into a shared dataset log directory.

The GLAM search uses an isolated seeded RNG. The split protocol is explicit (scaffold or random) and the split seed is applied to the random permutation. The default protocol is scaffold. The manifest records:
- canonical mission/submission identity and action-envelope hash;
- SHA-256 of the server-side input CSV;
- dataset revision label and split protocol;
- model/search seeds and bounded parameters;
- Python/platform and available dependency versions;
- optional pinned Git commit and OCI image digest;
- every emitted artifact's relative path, size, kind and SHA-256.

A successful process without best_save.ckpt is converted to FAILED. A job left RUNNING when the adapter process restarts is returned as INTERRUPTED; the outcome is never inferred as successful.

The run manifest is not an Evidence record. It must pass the mission's independent scientific validation profile before Evidence 1.1 can be produced.

## Runtime environment

deployment/glam-legacy-cpu-environment.yml records a conservative CPU environment aligned with GLAM's original dependency generation. It is a pinned starting point, not proof of an end-to-end scientific benchmark. The full environment solve, imports, demo run and emitted manifest must be archived before a release is called reproducible.

GPU deployments require a separately tested, pinned PyTorch/CUDA/PyG extension matrix. Record GPU model, driver and CUDA runtime. Do not infer GPU readiness from CPU tests.

For production deployment, set:
- GLAM_GIT_COMMIT to the exact GLAM source commit baked into the image;
- QFOLDIT_GLAM_CONTAINER_DIGEST to the immutable OCI image digest;
- QFOLDIT_GLAM_JOB_ROOT to persistent, access-controlled storage.

If source or container identity is not available, the manifest leaves it null rather than inventing provenance.

## Tests

    python -m py_compile integrations/qfoldit_glam/server.py src_1gp/glam.py src_1gp/trainer.py src_1gp/dataset.py
    python -m unittest discover -s src_1gp -p "test_reproducibility.py"
    python -m unittest discover -s integrations/qfoldit_glam -p "test_*.py"

These tests cover request contracts, provenance validation, command construction, seeded helper behavior and manifest hashing. They do not replace the required full demo run, GPU smoke test or scientific benchmark on controlled datasets.

## Security and operating limits

- Request bodies are bounded and unknown fields are rejected.
- Dataset paths are server-owned; callers cannot send shell commands, scripts, Python expressions or paths.
- The worker receives no adapter bearer token.
- The subprocess uses an argument vector with shell=False.
- Job concurrency and execution time are bounded.
- The adapter is loopback-only by default and has no CORS allowance.
- Logs are bounded in API responses and request bodies/tokens are not logged.
- The first implementation persists state and artifacts locally; a horizontally scaled production queue, cancellation/recovery coordinator, mTLS/workload identity and centralized observability remain deployment-level acceptance gates.
