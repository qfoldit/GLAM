# qFoldIT GLAM Adapter

**Adapter:** `qfoldit.glam`  
**Adapter version:** `0.2.0`  
**Request:** `qfoldit.glam-adapter.job-request/0.2`  
**Job response:** `qfoldit.glam-adapter.job-response/0.2`  
**Run result:** `qfoldit.glam.run-result/1.0`  
**Authority:** molecular-property model training / model selection and run artifacts only

This is a bounded service adapter around GLAM's existing adaptive molecular-property training and model-selection pipeline. It is not an independent scientific validator and cannot issue Evidence 1.1.

## qFoldIT architecture placement

```text
Mission Contract + Scientific Action Envelope
                    |
                    v
          qFoldIT Control Plane
                    |
          authenticated adapter call
                    |
                    v
            GLAM adapter / worker
                    |
        model training / model selection
                    |
                    v
         immutable run-result manifest
                    |
          GLAM prediction artifacts
                    |
                    v
       independent, task-appropriate validator
                    |
       validation.completed/1.1 / Evidence 1.1
```

GLAM owns its computational run and artifacts. Mission semantics, authorization, queue admission, customer policy and Evidence authority remain outside GLAM. UEFN/Verse is not a GLAM client and must never receive GLAM credentials, dataset paths, or worker commands.

## Supported capability

The only capability exposed by this adapter is `adaptive-property-search`: launch the existing `src_1gp/glam.py` adaptive training/model-selection pipeline against a server-configured dataset root.

The current adapter does **not** claim:
- direct inference for a molecule supplied by a mission;
- validated arbitrary candidate ranking;
- a production DTI or DDI screening API;
- an activity/affinity result for a specific molecule;
- independent scientific acceptance;
- Evidence 1.1 publication.

The repository contains DTI/DDI research modules, but those are not exposed by this first qFoldIT adapter. They require separate contracts, dataset manifests, benchmark protocols and integration tests.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/healthz` | Process liveness |
| GET | `/readyz` | Dependency and execution readiness |
| GET | `/v1/version` | Adapter, contract and dependency versions |
| GET | `/v1/capabilities` | Declared and bounded capabilities |
| POST | `/v1/jobs` | Submit one adaptive property-search job |
| GET | `/v1/jobs/{job_id}` | Read job lifecycle and bounded log tail |
| GET | `/v1/runs/{job_id}/manifest` | Read the completed run-result manifest |

Job creation, job inspection and run-manifest retrieval require `Authorization: Bearer <QFOLDIT_GLAM_TOKEN>`. The adapter binds to `127.0.0.1` by default. Remote deployments require a reviewed private gateway, workload identity, tenant authorization, TLS, managed secrets, durable queue/recovery, audit and retention controls.

## Request contract

A request binds the computation to the canonical mission and submission:

```json
{
  "contract": "qfoldit.glam-adapter.job-request/0.2",
  "task": "adaptive-property-search",
  "mission_id": "property-benchmark-001",
  "mission_version": "1.0.0",
  "submission_id": "11111111-1111-4111-8111-111111111111",
  "action_envelope_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "tenant_id": "tenant-example",
  "dataset": "demo",
  "dataset_revision": "bundled-demo",
  "parameters": {
    "n_init_configs": 1,
    "n_low_fidelity_seed": 1,
    "n_top_blend": 1,
    "n_high_fidelity_seed": 1,
    "seed": 1234,
    "split_seed": 1234,
    "split": "scaffold"
  }
}
```

The accepted datasets and numeric bounds are enforced by the adapter and the JSON Schema. Supported split strategies are `scaffold` and `random`; `scaffold` is the default. Requests cannot contain commands, scripts, Python expressions, arbitrary paths, environment variables, or uploaded code.

## Run-result manifest and artifacts

A process exit code is not enough to declare the GLAM task complete. When the worker exits successfully, the adapter verifies that the job-scoped output contains the expected high-fidelity and model-selection artifacts, including:
- at least one `best_save.ckpt`;
- `inf_ckpt_selected.csv`;
- `inference_log.txt`;
- `search_result.csv`.

Each artifact receives a SHA-256 digest and byte count. The adapter writes `run-result.json` conforming to `run-result.schema.json`. The manifest records:
- mission ID/version, submission UUID and action-envelope hash;
- dataset ID/revision and calculated raw CSV SHA-256;
- split method and split seed;
- GLAM Git commit when available;
- Python, PyTorch, PyG and RDKit versions when available;
- normalized run parameters;
- output artifact paths, byte counts and hashes;
- run completion timestamp.

Job outputs are isolated under `QFOLDIT_GLAM_JOB_ROOT/<job_id>/output`. No job shares a mutable `log_<dataset>` directory with another qFoldIT job.

The adapter deliberately returns:
- `scientific_validation_status = NOT_PERFORMED`;
- `evidence_status = NOT_CREATED`.

These are invariant authority boundaries, not temporary UI states. Only a separately approved validator and the canonical Evidence boundary can advance them.

## Reproducibility and worker safety changes

The qFoldIT pipeline hardening includes:
- `subprocess.Popen(..., shell=False)` for low- and high-fidelity workers;
- non-zero child exit codes fail the parent run;
- seeded architecture sampling uses a local `random.Random(seed)`;
- random split uses a local NumPy generator;
- scaffold split uses a local seeded NumPy generator and deterministic scaffold sampler;
- the requested split is passed explicitly to the dataset loader and recorded in configuration;
- split-cache filenames include a digest of the raw dataset file;
- high-fidelity checkpoint selection excludes low-fidelity search checkpoints;
- high-fidelity model outputs are isolated by job output root;
- log configuration parsing uses `ast.literal_eval`, not `eval`.

This establishes code-level and adapter-level controls. It does not by itself demonstrate that a particular GLAM benchmark passed, that all historical datasets are compatible with the current Python/PyTorch stack, or that the model generalizes to an external target.

## Local run

From the repository root, install the research dependencies using the GLAM environment documented in the root README. Prepare the selected dataset under a server-controlled directory and configure:

```bash
export GLAM_GP_DATASET_ROOT="$PWD/Dataset/GLAM-GP"
export QFOLDIT_GLAM_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export QFOLDIT_GLAM_JOB_ROOT="$PWD/.qfoldit-glam-jobs"
python integrations/qfoldit_glam/server.py
```

For the bundled demo, `GLAM_GP_DATASET_ROOT` is not required. The adapter still requires the GLAM research dependencies and a prepared demo CSV.

## Verification

Run the light-weight adapter and source-contract suite:

```bash
python -m py_compile integrations/qfoldit_glam/server.py
python -m compileall -q src_1gp
python -m unittest discover -s integrations/qfoldit_glam -p "test_*.py" -v
```

These tests cover request validation, provenance, parameter bounds, safe argv construction, artifact-manifest requirements and source-level worker/split invariants. They do not replace an end-to-end training run, model-performance benchmark, locked holdout evaluation or independent scientific review.

## Production acceptance gates

Do not mark this adapter production-ready until all of the following have separate evidence:
1. A pinned, reproducible Python/PyTorch/PyG/RDKit environment and immutable container digest.
2. A successful end-to-end demo job showing low-fidelity training, high-fidelity training, checkpoint selection, ensemble inference and manifest generation.
3. A representative real-dataset run with verified dataset hash and split manifest.
4. Locked holdout evaluation, baselines, repeated seeds, uncertainty/calibration and applicability-domain analysis.
5. Model artifacts can be retrieved from durable controlled storage and reproduced by an independent operator.
6. Remote workload identity, tenant isolation, durable job queue/recovery, cancellation, monitoring and retention have passed operational review.
7. An independent validator appropriate to the mission has evaluated the submitted prediction artifacts.
8. The Evidence 1.1 conformance path has consumed the validator's signed result.

## Scientific and licensing boundary

GLAM's research method and original publication remain attributed to the original authors. qFoldIT's adapter is a separate integration layer. Preserve the upstream license, publication citation, dataset licenses and third-party dependency notices. The qFoldIT wrapper must not be interpreted as a transfer of ownership of upstream GLAM research.
