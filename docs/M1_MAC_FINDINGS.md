# M1 Mac Development Slice: Findings and Handoff

Date: 2026-10-16  
Status: Mac implementation and FP16 loading validated; M1 remains incomplete
until CUDA/bitsandbytes quantization and inference are audited.

## 1. Objective

This work established the M0 environment prerequisite and implemented the
Mac-compatible portion of M1. The goal was to inspect the real
`HuggingFaceTB/SmolVLM2-2.2B-Instruct` architecture, define the requested
FP16/NF4 candidate grid, audit stored model state, and measure FP16 weight
storage without claiming unsupported NF4 results.

The scientific requirements and M1 acceptance criteria remain defined in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) and
[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md).

## 2. Environment Established

The Conda environment is named `mpvs-m1` and uses Python 3.11.17. The tested
direct dependencies are recorded in
[environment-macos.yml](../environment-macos.yml), while the complete installed
package inventory is in
[requirements-macos-lock.txt](../requirements-macos-lock.txt).

Important resolved versions:

| Package | Version |
| --- | ---: |
| PyTorch | 2.14.1 |
| Transformers | 4.52.4 |
| Accelerate | 1.15.0 |
| NumPy | 2.4.6 |
| Pillow | 13.0.0 |

Transformers 5.20.1 was initially installed but rejected after its SmolVLM
image-token merger failed during the smoke test. Version 4.52.4 loads the
checkpoint correctly and is the recorded Mac compatibility version.

Environment commands:

```bash
conda create -n mpvs-m1 python=3.11 pip -y
conda activate mpvs-m1
python -m pip install torch torchvision transformers accelerate \
  Pillow numpy pytest sentencepiece protobuf num2words
python -m pip check
python scripts/preflight.py --output artifacts/preflight-macos-m1.json
```

For reproduction from the tracked specification:

```bash
conda env create -f environment-macos.yml
conda activate mpvs-m1
python -m pip install -e .
```

## 3. Model and Processor Resolution

Both model and processor are pinned to immutable revision:

```text
Model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
Revision: 482adb537c021c86670beed01cd58990d01e72e4
```

The defaults and fresh FP16 loader are implemented in
[model.py](../src/mpvs/model.py). Each load starts from the original checkpoint;
the code does not quantize or dequantize an already modified model.

Inspection of the installed model implementation established these component
roots:

| Research component | Verified module root |
| --- | --- |
| Vision encoder | `model.vision_model` |
| Connector | `model.connector` |
| Language backbone | `model.text_model` |
| Language output head | `lm_head` |

The mapping and strict prefix-boundary classification are implemented in
[quantization.py](../src/mpvs/quantization.py).

## 4. Implemented M1 Functionality

- [model.py](../src/mpvs/model.py) defines immutable model metadata, device
  resolution, fresh FP16 loading, component discovery, and a deterministic
  synthetic-image smoke-inference function.
- [quantization.py](../src/mpvs/quantization.py) defines all eight requested
  `(vision, connector, language)` FP16/NF4 candidates, module-state inspection,
  requested-versus-realized coverage, and effective-configuration signatures.
- [memory.py](../src/mpvs/memory.py) counts unique underlying parameter and
  persistent-buffer storage without double-counting shared storage, and reports
  component totals separately.
- [cli.py](../src/mpvs/cli.py) exposes `candidates` and `audit` commands, writes
  audit JSON atomically, and distinguishes unsupported candidates, load
  failures, and inference failures.
- [test_quantization.py](../tests/test_quantization.py) and
  [test_memory.py](../tests/test_memory.py) cover candidate stability, module
  boundaries, realized allocation reporting, signatures, NF4 rejection, shared
  storage, buffers, and unclassified tensors.

Example commands:

```bash
mpvs candidates

mpvs audit \
  --candidate v-fp16_c-fp16_l-fp16 \
  --device cpu \
  --output artifacts/m1-fp16-macos.json
```

NF4 requests currently return a structured `unsupported` record and a nonzero
exit code. This gate prevents a requested NF4 candidate from being silently
loaded and reported as FP16.

## 5. Verified Findings

The full FP16 checkpoint loaded successfully on CPU and its module/tensor state
was audited. Under the implemented unique-storage convention:

| Measurement | Bytes |
| --- | ---: |
| Vision parameters | 825,974,496 |
| Connector parameters | 42,467,328 |
| Language parameters, including `lm_head` | 3,625,127,936 |
| Persistent buffers | 128 |
| Total FP16 weight storage | 4,493,569,888 |

These are model-storage measurements, not peak process memory or GPU-memory
measurements. NF4 quantization metadata is not included because no real NF4
model has been constructed yet.

Validation completed:

```text
conda run -n mpvs-m1 python -m pytest -q
7 passed

conda run -n mpvs-m1 python -m pip check
No broken requirements found.
```

## 6. Limitations and Failed Checks

- This Apple M5 Mac has no NVIDIA CUDA runtime or bitsandbytes installation.
- PyTorch reports `mps_built=True` but `mps_available=False` in the current
  execution environment.
- The FP16 checkpoint loaded on CPU, but synthetic FP16 CPU generation was not
  validated. One attempt exposed an NNPack tensor-type/backend limitation; the
  corrected long-running CPU attempt did not produce a successful replacement
  audit. It must not be reported as inference success.
- No NF4 candidate has been built, loaded, measured, deduplicated, or run.
- No memory cap is established because M1 requires at least two distinct,
  feasible measured candidates on the target execution host.
- No dataset, pilot question, development/test example, training, calibration,
  or experimental evaluation was used.

Generated audits and downloaded weights live in ignored cache/artifact paths.
The concise tracked status is maintained in [live_STATUS.md](live_STATUS.md), and material
environment decisions are recorded in [DECISIONS.md](DECISIONS.md).

## 7. Next Steps on the NVIDIA Host

1. Run [preflight.py](../scripts/preflight.py) and record the GPU model, memory,
   driver, CUDA runtime, disk availability, and installed package state.
2. Create a separate CUDA environment and test a mutually compatible PyTorch,
   Transformers 4.52.4, Accelerate, and bitsandbytes combination. Do not assume
   the Mac lock file is the CUDA lock file.
3. Run one batch-size-1 FP16 synthetic inference and record peak allocated and
   reserved GPU memory separately from weight storage.
4. Implement selective NF4 replacement for supported linear layers using the
   verified component roots. Reconstruct every candidate independently from
   revision `482adb537c021c86670beed01cd58990d01e72e4`.
5. Audit actual module classes, `Params4bit`/quantization state, compute dtype,
   double-quantization setting, exceptions, buffers, and quantization metadata.
6. Run one synthetic inference per feasible candidate in isolated processes,
   record failures honestly, and compute effective signatures to remove
   duplicate realized configurations.
7. Identify at least one measured weight-memory cap containing two distinct
   feasible candidates. Only then can M1 acceptance be considered satisfied.
8. Update [live_STATUS.md](live_STATUS.md) and [DECISIONS.md](DECISIONS.md) with exact CUDA
   commands and evidence before beginning M2.

Do not start bulk inference or claim M1 complete until these CUDA checks pass.
