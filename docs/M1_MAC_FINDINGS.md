# M1 Model Audit Findings: Mac Development and T4 Validation

Date: 2026-10-28
Status: **M1 acceptance criteria met** by the complete CUDA audit at repository
commit `fdeea24367cf8f64fa9b5b8e9aea9de3beb5eafe`.

## 1. Scope and conclusion

M1 asked whether the vision encoder, connector, and language backbone of
`HuggingFaceTB/SmolVLM2-2.2B-Instruct` can be controlled independently at FP16
or NF4 for supported linear layers, while measuring actual stored weight state
instead of trusting loader flags.

The Mac work established the environment, immutable checkpoint revision,
architecture mapping, FP16 reference, audit code, and unit tests. The final
audit then ran on a Colab Tesla T4. All eight requested FP16/NF4 tuples:

- loaded independently from the same original checkpoint;
- matched their requested component allocation;
- had distinct effective-configuration signatures;
- completed one deterministic synthetic-image inference; and
- produced measured unique weight-storage totals, including exposed
  quantization metadata and persistent buffers.

The smallest measured cap containing at least two distinct supported
configurations is **1,587,421,408 bytes**. This satisfies the M1 acceptance
criterion in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). It demonstrates
candidate feasibility only; it does not freeze the final experimental cap or
provide model-accuracy evidence.

## 2. Immutable model and component mapping

```text
Checkpoint: HuggingFaceTB/SmolVLM2-2.2B-Instruct
Model revision: 482adb537c021c86670beed01cd58990d01e72e4
Processor revision: 482adb537c021c86670beed01cd58990d01e72e4
```

The verified research-component roots are:

| Research component | Verified module root |
| --- | --- |
| Vision encoder | `model.vision_model` |
| Connector | `model.connector` |
| Language backbone | `model.text_model` |
| Preserved language output head | `lm_head` |

`lm_head` is a documented FP16 exception for every candidate. In this project,
an “NF4 component” means that all supported linear layers under that component
are NF4. Unsupported layer types, embeddings, normalization parameters,
convolutions, biases, and the output head may remain FP16 and are included in
the measured storage total.

## 3. Environments used

### 3.1 Mac development environment

The Conda environment is `mpvs-m1` with Python 3.11.17. Reproducible direct
dependencies are recorded in [environment-macos.yml](../environment-macos.yml),
and the complete installed inventory is recorded in
[requirements-macos-lock.txt](../requirements-macos-lock.txt).

| Package | Mac version |
| --- | ---: |
| PyTorch | 2.14.1 |
| Transformers | 4.52.4 |
| Accelerate | 1.15.0 |
| NumPy | 2.4.6 |
| Pillow | 13.0.0 |

Transformers 5.20.1 was rejected after its SmolVLM image-token merger failed
during the initial smoke-test work. Transformers 4.52.4 loaded the pinned
checkpoint correctly and was retained for both the Mac architecture audit and
the T4 quantization audit.

The Mac has no NVIDIA runtime, bitsandbytes was absent, and PyTorch reported
MPS built but unavailable. Consequently, the Mac was used for FP16 loading,
architecture inspection, implementation, and unit tests—not NF4 validation.

### 3.2 Colab CUDA audit environment

| Item | Measured value |
| --- | --- |
| GPU | Tesla T4, 15,360 MiB |
| Driver reported by `nvidia-smi` | 580.82.07 |
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |
| Compute capability | 7.5 |
| Transformers | 4.52.4 |
| Accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |
| NumPy | 2.1.3 |
| Pillow | 11.3.0 |

The environment inventory is stored in the ignored artifact
`artifacts/m1-audit-t4-fdeea243/preflight-colab-t4.json`. The preflight file is
an inventory, not the quantization result; the eight candidate audit files are
the validation evidence.

## 4. Implemented audit path

- [model.py](../src/mpvs/model.py) records immutable revisions, resolves the
  device, loads every candidate freshly from the original checkpoint, discovers
  components, and runs deterministic synthetic inference.
- [quantization.py](../src/mpvs/quantization.py) defines the stable eight-tuple
  grid, builds the selective bitsandbytes loader policy, classifies modules,
  inspects realized quantization state, detects mismatches, and computes stable
  effective signatures.
- [memory.py](../src/mpvs/memory.py) counts unique underlying storage across
  parameters, persistent buffers, and tensors reachable from bitsandbytes
  `quant_state`, without double-counting shared storage.
- [cli.py](../src/mpvs/cli.py) writes atomic machine-readable candidate audits,
  distinguishes unsupported/load/audit/inference failures, and summarizes the
  isolated grid.
- [m1_colab_cuda_nf4.ipynb](../notebooks/m1_colab_cuda_nf4.ipynb) installs the
  validated CUDA-side versions, runs each candidate in a fresh subprocess, and
  packages the generated audit artifacts.
- [test_quantization.py](../tests/test_quantization.py),
  [test_memory.py](../tests/test_memory.py), [test_cli.py](../tests/test_cli.py),
  and [test_cli_audit.py](../tests/test_cli_audit.py) cover stable candidates,
  component boundaries, requested-versus-realized state, effective signatures,
  memory deduplication, quantization metadata, grid acceptance, and structured
  unsupported results.

The fixed CUDA NF4 policy was:

| Setting | Value |
| --- | --- |
| Quantization type | NF4 |
| Compute dtype | FP16 |
| Quantized storage dtype | uint8 |
| Double quantization | Disabled |
| CPU offload | Disabled |
| Preserved exception | `lm_head` in FP16 |
| Candidate isolation | Fresh subprocess and checkpoint load |

## 5. Realized component coverage

The fully NF4 candidate established the number of independently controlled
supported linear layers:

| Component | Supported linear modules | Quantized in fully NF4 | Exceptions |
| --- | ---: | ---: | --- |
| Vision encoder | 162 | 162 | None |
| Connector | 1 | 1 | None |
| Language backbone | 168 | 168 | `lm_head` retained in FP16 |
| Total | 331 | 331 | One documented output-head exception |

Across all eight reports:

- every requested NF4 linear module was a
  `bitsandbytes.nn.modules.Linear4bit` converted from
  `torch.nn.modules.linear.Linear`;
- every quantized module reported NF4, FP16 compute, uint8 storage, and no
  double quantization;
- every requested FP16 component had zero quantized modules and retained FP16
  stored weights;
- `lm_head` remained unquantized FP16;
- every component mismatch list was empty; and
- no unclassified module was quantized.

The audit therefore validates independent component control. It is not global
quantization relabeled as mixed precision.

## 6. Candidate storage and smoke-inference results

Tuple order below is **vision / connector / language**. Weight storage counts
unique parameter storage, persistent buffers, and exposed quantization metadata.
Peak reserved memory is a separate CUDA runtime measurement from the synthetic
inference.

| Candidate | Weight bytes | Weight GiB | Peak reserved GiB | Generation seconds | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| NF4 / NF4 / NF4 | 1,556,898,080 | 1.450 | 2.283 | 3.98 | Success |
| NF4 / FP16 / NF4 | 1,587,421,408 | 1.478 | 2.283 | 4.05 | Success |
| FP16 / NF4 / NF4 | 2,147,801,504 | 2.000 | 2.600 | 3.90 | Success |
| FP16 / FP16 / NF4 | 2,178,324,832 | 2.029 | 2.627 | 3.89 | Success |
| NF4 / NF4 / FP16 | 3,872,143,136 | 3.606 | 4.182 | 3.98 | Success |
| NF4 / FP16 / FP16 | 3,902,666,464 | 3.635 | 4.213 | 3.85 | Success |
| FP16 / NF4 / FP16 | 4,463,046,560 | 4.157 | 4.781 | 4.01 | Success |
| FP16 / FP16 / FP16 | 4,493,569,888 | 4.185 | 4.957 | 3.78 | Success |

Every candidate returned `Yes.` for the deterministic 384×384 synthetic blue
image prompt, using greedy decoding and at most four new tokens. This only
checks execution viability. It is not a task-accuracy, robustness, real-image,
or throughput result.

For the fully NF4 candidate, the total consists of:

| Storage category | Bytes |
| --- | ---: |
| Parameter storage | 1,429,194,464 |
| Persistent buffers | 128 |
| Exposed quantization metadata | 127,703,488 |
| Total | 1,556,898,080 |

The FP16 reference contains 4,493,569,760 parameter bytes plus 128 persistent
buffer bytes, matching the earlier Mac CPU measurement exactly.

## 7. Independent component effects on storage

Holding the other two components at FP16, replacing one component's supported
linear layers with NF4 reduced measured storage by:

| Component changed to NF4 | Bytes saved versus all-FP16 |
| --- | ---: |
| Vision encoder | 590,903,424 |
| Connector | 30,523,328 |
| Language backbone | 2,315,245,056 |

These savings are exactly additive:

```text
4,493,569,888
- 590,903,424
-  30,523,328
- 2,315,245,056
= 1,556,898,080 bytes
```

The result provides a storage-level cross-check that the three component
controls were applied independently.

## 8. Measured feasible caps

All eight candidates are supported and have distinct effective signatures. The
smallest exact threshold admitting at least two candidates is:

```text
1,587,421,408 bytes
```

At that threshold, the eligible candidates are:

1. NF4 / NF4 / NF4 — 1,556,898,080 bytes.
2. NF4 / FP16 / NF4 — 1,587,421,408 bytes.

The audit summary also enumerates larger exact thresholds admitting three
through eight candidates. These measurements establish that a common cap is
possible, satisfying M1. The cap or caps used for the main selection experiment
must still be chosen and frozen during the pilot without examining comparative
test accuracy. The smallest cap may be scientifically narrow because its two
candidates differ only in connector precision.

## 9. Verification and acceptance decision

Mac-side validation after the CUDA implementation:

```text
conda run -n mpvs-m1 python -m pytest -q
14 passed

Notebook JSON and code-cell compilation: passed
Python source compilation: passed
Wheel construction: passed
Structured unsupported-candidate check: passed
Git whitespace check: passed
```

Independent inspection of the downloaded T4 artifacts confirmed:

```text
Raw candidate reports: 8
Unique candidate IDs: 8
Raw memory values match summary: yes
Identical model and processor revisions: yes
Identical recorded execution environment: yes
All loads successful: yes
All requested allocations matched: yes
All inference smoke tests successful: yes
Failure reasons/stages present: none
Unclassified quantized modules: none
Distinct effective signatures: 8
Summary acceptance: true
```

The primary machine-readable evidence is retained locally under the ignored
directory:

```text
artifacts/m1-audit-t4-fdeea243/
  m1-audit-t4-summary.json
  m1-audit-t4-v-fp16_c-fp16_l-fp16.json
  m1-audit-t4-v-fp16_c-fp16_l-nf4.json
  m1-audit-t4-v-fp16_c-nf4_l-fp16.json
  m1-audit-t4-v-fp16_c-nf4_l-nf4.json
  m1-audit-t4-v-nf4_c-fp16_l-fp16.json
  m1-audit-t4-v-nf4_c-fp16_l-nf4.json
  m1-audit-t4-v-nf4_c-nf4_l-fp16.json
  m1-audit-t4-v-nf4_c-nf4_l-nf4.json
  preflight-colab-t4.json
```

**Conclusion:** M1 is implemented, unit-tested, CUDA/NF4-audited, and accepted.
Independent control is achievable for the full requested grid, so no model or
quantizer fallback is required at this milestone.

## 10. Limitations

- The smoke input is synthetic and contains no dataset example.
- The generated answer verifies execution only; it provides no accuracy or
  robustness evidence.
- Peak CUDA memory is for batch size 1, one 384×384 image, a short prompt, and
  four generated tokens. It does not prove that the eventual pilot or full
  evaluation workload fits unchanged.
- Weight storage and peak GPU memory are different quantities. Activations, KV
  cache, allocator behavior, and temporary buffers are not part of the weight
  cap.
- Load time includes local cache and runtime effects and is not a stable
  throughput benchmark.
- The final experimental cap set, prompt, parser, corruption severities, and
  batch/execution constraints remain pilot decisions.
- No POPE-COCO or CelebA data, training, fine-tuning, calibration, development
  selection, or held-out test evaluation occurred in M1.

## 11. Reproduction commands

Mac development setup:

```bash
conda env create -f environment-macos.yml
conda activate mpvs-m1
python -m pip install -e .
python -m pytest -q
python scripts/preflight.py --output artifacts/preflight-macos-m1.json
```

List the stable grid:

```bash
mpvs candidates
```

Audit one CUDA candidate from a fresh process:

```bash
python -m mpvs.cli audit \
  --candidate v-nf4_c-nf4_l-nf4 \
  --device cuda \
  --smoke-inference \
  --output artifacts/m1-audit-t4-v-nf4_c-nf4_l-nf4.json
```

The reproducible complete-grid entry point is
[m1_colab_cuda_nf4.ipynb](../notebooks/m1_colab_cuda_nf4.ipynb). It records the
repository commit, environment, immutable model revisions, per-candidate audit,
and summary before packaging the ignored artifacts.

## 12. Handoff to M2 and M3

M2 may now proceed with normalized POPE-COCO and CelebA records, group-disjoint
pilot/development/test manifests, deterministic darkness and Gaussian-blur
transforms, and synthetic leakage/determinism tests. M2 must not use test
accuracy to choose data, corruptions, or severities.

Before bulk inference, M3 must still implement and validate fixed prompting,
strict yes/no parsing, resumable prediction caching, real-image pilot execution,
resource estimation, and protocol freezing. The pilot must select final caps
and execution constraints before comparative test accuracy is examined.
