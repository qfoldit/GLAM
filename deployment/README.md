# GLAM execution environment

The qFoldIT adapter is a wrapper around the original GLAM research implementation.
Do not silently upgrade PyTorch, PyTorch Geometric, RDKit, NumPy or scikit-learn
inside a benchmark run: such changes can alter graph construction, numerical
results, checkpoint compatibility and metric behavior.

## Reproducible legacy CPU baseline

`glam-legacy-cpu-environment.yml` records a conservative CPU environment
aligned with the dependency generation documented by the original GLAM authors.
Create it with:

```bash
conda env create -f deployment/glam-legacy-cpu-environment.yml
conda activate qfoldit-glam-legacy-cpu
```

For PyG extensions whose wheels are not available from the default index, install
the matching CPU wheels for PyTorch 1.10.0 from the official PyG wheel index.
Do not substitute a wheel built for a different PyTorch/CUDA ABI.

This file is a pinned environment specification, not a claim that the full
benchmark has already been reproduced on every operating system. The release
gate requires a successful environment solve, import smoke test, adapter tests,
and an actual end-to-end demo run with the emitted run manifest archived.

## GPU profiles

GPU execution must use a separately tested environment with an exact NVIDIA
driver/CUDA runtime and matching PyTorch/PyG extension ABI. Do not infer GPU
readiness from the CPU environment. Record the accelerator model, driver,
CUDA runtime, PyTorch build and GPU determinism settings in the run manifest.

## Output isolation

Set `QFOLDIT_GLAM_RUN_ROOT` to a writable per-job directory. The trainer writes
all checkpoints, search summaries and inference logs beneath that directory.
Never share a run-output directory between concurrent jobs. The qFoldIT adapter
assigns one private output directory per job and hashes every produced artifact.
