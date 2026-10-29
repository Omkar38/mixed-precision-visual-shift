# Mixed Precision Allocation Under Visual Distribution Shift

CSE590-01, Fall 2026. Three-person team project.

**Question:** under the same model-weight memory limit, does a compression
setting chosen using clear images still work well on darker or blurrier images?

A small vision-language model has three parts — a vision encoder, a connector,
and a language backbone — and each one's weights can be stored at 4-bit NF4 or
FP16 independently. That gives eight configurations. Standard practice picks the
best one by testing on clean benchmark photos. This project measures whether
that choice survives when the deployed images are dark or blurry.

The headline number is a single subtraction, on the same degraded test set:

> **Δ = accuracy of the configuration chosen using degraded images − accuracy of
> the configuration chosen using clean images**

A positive Δ means clean-image selection cost real accuracy. A Δ near zero means
the choice is portable, which confirms an assumption the field currently makes
without evidence. **Both outcomes are reportable results.**

## Current state

| Milestone | State |
|---|---|
| **M0** — environment and access | **Done.** macOS development and Colab CUDA environments pinned and measured |
| **M1** — component-control audit | **Accepted.** All 8 configurations audited on a Tesla T4 |
| **M2** — data and corruption pipeline | **Implemented**, 42 tests passing. POPE-COCO validated on real data; CelebA annotations validated, images not downloaded |
| **M3** — pilot, cache, protocol freeze | Not started |
| **M4** — selection and statistics | Not started |
| **M5** — main runs and report | Not started |

**There is no accuracy result.** No pilot has run, no real image has been
through the model, and no severity, split size, or memory cap is frozen. The
transfer-penalty measurement this project exists to produce has not started.

### What M1 established

Independent component control is measured, not assumed. All 331 supported
linear modules converted exactly as requested — 162 vision, 1 connector, 168
backbone — with zero mismatches, zero unclassified quantized modules, eight
distinct effective signatures, and `lm_head` preserved at FP16 throughout.

Measured weight storage, with per-component savings that are **exactly
additive**, confirming the three controls do not contaminate each other:

| Vision | Connector | Backbone | Weight bytes |
|---|---|---|---:|
| NF4 | NF4 | NF4 | 1,556,898,080 |
| NF4 | FP16 | NF4 | 1,587,421,408 |
| FP16 | NF4 | NF4 | 2,147,801,504 |
| FP16 | FP16 | NF4 | 2,178,324,832 |
| NF4 | NF4 | FP16 | 3,872,143,136 |
| NF4 | FP16 | FP16 | 3,902,666,464 |
| FP16 | NF4 | FP16 | 4,463,046,560 |
| FP16 | FP16 | FP16 | 4,493,569,888 |

Full report: [docs/M1_MAC_FINDINGS.md](docs/M1_MAC_FINDINGS.md).

## Start here

1. [docs/MIDTERM_M0_M2.md](docs/MIDTERM_M0_M2.md) — progress report with the measured results.
2. [docs/PROJECT_DEEP_DIVE.md](docs/PROJECT_DEEP_DIVE.md) — plain-English explanation, technical deep dive, and hardware requirements. Also available as a PDF.
3. [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — full project definition, measurement, and controls.
4. [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — ordered milestones with acceptance criteria.
5. [AGENTS.md](AGENTS.md) — instructions shared by coding agents; `CLAUDE.md` imports it.

Record progress in [docs/live_STATUS.md](docs/live_STATUS.md) and protocol
decisions in [docs/DECISIONS.md](docs/DECISIONS.md).

## Requirements

**An NVIDIA GPU is mandatory.** bitsandbytes NF4 has no Apple Silicon or CPU
backend, so without CUDA there is no 4-bit configuration and no experiment.
Apple Silicon is usable for code, unit tests, module audits, and FP16 CPU
loading only.

| Resource | Minimum | Validated |
|---|---|---|
| GPU | NVIDIA 8 GB, compute capability ≥ 7.5 | Tesla T4, 15,360 MiB, CC 7.5 |
| System RAM | 16 GB | — |
| Disk | 25 GB free | — |
| Python | 3.11+ | 3.11.17 (dev) / 3.13.16 (Colab) |
| Transformers | **4.52.4** | 5.20.1 breaks SmolVLM image-token merging |
| bitsandbytes | CUDA build | 0.50.2 |

The largest configuration peaked at 4.957 GiB reserved on one synthetic image,
so 8 GB is viable with headroom. Real images may use more; re-check in the
pilot.

## Running it

```bash
conda env create -f environment-macos.yml
conda activate mpvs-m1
python -m pip install -e .
PYTHONPATH=src python -m pytest -q
python scripts/preflight.py --output artifacts/preflight.json
```

`scripts/preflight.py` is a dependency-free environment inventory; it downloads
nothing and runs no inference.

A single `mpvs` CLI covers both the audit and the data pipeline:

```bash
mpvs candidates                  # print the stable requested candidate grid
mpvs audit                       # audit one freshly loaded candidate
mpvs summarize                   # summarize isolated candidate audit JSON
mpvs m2-normalize-pope           # normalize one POPE sampling variant
mpvs m2-normalize-celeba         # normalize selected CelebA attribute questions
mpvs m2-split                    # deterministic group-disjoint split manifests
mpvs m2-validate-splits          # image, identity, question, and group leakage
mpvs m2-expand-variants          # expand a manifest with explicit corruption specs
mpvs m2-validate-answerability   # model-independent corrupted-pilot decisions
```

For the GPU audit, open
[notebooks/m1_colab_cuda_nf4.ipynb](notebooks/m1_colab_cuda_nf4.ipynb) in Colab,
select a T4, and run all cells. It installs the project, runs a feasibility
gate, then audits all eight candidates in fresh subprocesses, preserves failure
JSON, and downloads the artifacts.

## Pipeline

| Stage | Contents |
| --- | --- |
| Input | Images, yes/no questions, labels, split and group identifiers, a pinned checkpoint, candidate precision settings |
| Processing | Split originals into disjoint groups, then corrupt; audit and evaluate fixed configurations; select settings using development data only, under measured memory caps |
| Output | Cached predictions, configuration audit, memory and runtime measurements, selection decisions, accuracy differences with cluster-aware uncertainty |

Two invariants carry the study. **Split before corrupting**, by original image
for POPE-COCO and by identity for CelebA, so no image or person straddles a
split and every corrupted variant inherits its original's split. And **never
consult test accuracy during selection** — both Δ terms come from development
data.

### Data

POPE-COCO is validated against the official files: 3,000 questions over 500
images per sampling variant, balanced 1,500/1,500, six questions per image, no
duplicate IDs, no missing images, deterministic normalization. CelebA annotation
files match their published MD5 values — 202,599 images, 40 attributes, 10,177
identities — but the images are not yet downloaded. 615 identities carry
inconsistent `Male` labels, documented as source label noise rather than
repaired.

See [docs/M2_PIPELINE.md](docs/M2_PIPELINE.md) and
[docs/M2_REAL_DATA_VALIDATION.md](docs/M2_REAL_DATA_VALIDATION.md).

## Open decisions

None of these are frozen, and the pilot cannot be frozen without them:

- **NF4 double quantization**, currently off and costing ~128 MB of metadata. Turning it on would shift every measured cap.
- **Blur severity units.** The Gaussian radius is in absolute pixels, so one radius is far stronger on a ~178 px CelebA crop than on a ~640 px COCO image. Either define it relative to the short edge or treat severities as per-domain.
- **Which measured caps are primary.** The smallest cap admitting two candidates compares two that differ only in connector precision, which is 0.9% of the weights.
- **Steady-state throughput**, the number that decides whether the full design is affordable.

## Data and outputs

Keep dataset images under `data/`, model weights under `models/`, and generated
results under `artifacts/` or `runs/`. These paths are Git-ignored. Commit code,
configuration, small permitted split manifests, and report summaries
deliberately. Do not commit CelebA images, model weights, credentials, or full
prediction dumps.

## Original proposal

[Google Docs — Final tab](https://docs.google.com/document/d/1gtlE4NJRYpqTqe1G5gV5SIBgzHL_DG9FJyWphVvco9Q/edit?tab=t.9i03s424878a),
captured in [docs/PROPOSAL_SNAPSHOT.md](docs/PROPOSAL_SNAPSHOT.md).

This repository expands the proposal into implementation instructions. It does
not claim professor approval. Changes to the scientific scope must be recorded
in the decision log.
