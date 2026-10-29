# Mixed Precision Allocation Under Visual Distribution Shift

CSE590-01, Fall 2026. Three-person team project.

**Question:** Under the same model-weight memory limit, do compression settings selected using clear images still work well on darker or blurrier images?

**Current state:** M0 and M1 complete; M2 implemented. The eight-candidate
FP16/NF4 component grid has been audited on an NVIDIA T4 with measured weight
storage, and the POPE-COCO pipeline is validated on real annotations.
**No accuracy experiment has run:** there is no pilot, no evaluation on real
images, and no transfer-penalty result. See
[the midterm progress report](docs/MIDTERM_M0_M2.md).

## Start here

1. Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the complete project explanation.
2. Read [AGENTS.md](AGENTS.md) for instructions shared by coding agents.
3. Follow [the implementation plan](docs/IMPLEMENTATION_PLAN.md), beginning with M0.
4. Record progress in [live_STATUS.md](docs/live_STATUS.md) and decisions in [DECISIONS.md](docs/DECISIONS.md).

Open this repository folder in Claude Code or Codex. Claude's entry file imports the shared agent instructions; Codex uses `AGENTS.md`. Give either agent this prompt:

> Read AGENTS.md, PROJECT_CONTEXT.md, docs/IMPLEMENTATION_PLAN.md, and docs/live_STATUS.md. Start with M0 and then implement M1. Verify the environment, inspect the actual model architecture, and build the component-level quantization audit. Do not assume all eight precision configurations work. Use small, reviewable changes and report what you actually ran. Keep the current research question and evaluation protocol.

## What goes in and comes out?

| Stage | Contents |
| --- | --- |
| Input | Images, yes/no questions, labels, split/group identifiers, a model checkpoint, and candidate precision settings |
| Processing | Split originals into disjoint groups; create image corruptions; audit and evaluate fixed model configurations; select settings using development data under memory caps |
| Output | Cached predictions, configuration audit, memory/runtime measurements, selection decisions, accuracy differences with uncertainty, and final figures/report |

Example: We ask “Is there a dog in this image?” on clear and dark versions of a photo. We test whether a configuration chosen on clear development photos loses accuracy on dark test photos compared with one chosen on dark development photos, while both obey the same memory cap.

## What is available now?

- `PROJECT_CONTEXT.md`: full handoff, scope, hypotheses, and definitions.
- `AGENTS.md` and `CLAUDE.md`: consistent coding-agent instructions.
- `docs/EXPERIMENT_PROTOCOL.md`: experimental controls and output contracts.
- `docs/IMPLEMENTATION_PLAN.md`: ordered work with acceptance criteria.
- `docs/PROPOSAL_SNAPSHOT.md`: current proposal captured on 2026-10-05.
- `docs/MIDTERM_M0_M2.md`: midterm progress report with the measured M1 results.
- `docs/PROJECT_DEEP_DIVE.md`: plain-English explanation, technical deep dive, and hardware requirements.
- `docs/M1_MAC_FINDINGS.md`: full component-control audit report.
- `docs/M2_PIPELINE.md` and `docs/M2_REAL_DATA_VALIDATION.md`: data pipeline and its real-data validation.
- `configs/protocol.example.json`: draft protocol; null fields deliberately mark unresolved pilot decisions.
- `scripts/preflight.py`: dependency-free environment inventory; it does not download models or execute inference.

Run the inventory with Python 3.10 or later:

```bash
python3 scripts/preflight.py
python3 scripts/preflight.py --output artifacts/preflight.json
```

The `mpvs` CLI lists and audits M1 candidates. The complete eight-candidate
CUDA/NF4 grid was validated on a Tesla T4; see
[the M1 findings](docs/M1_MAC_FINDINGS.md).

## Colab M1 component audit

The [M1 Colab CUDA notebook](notebooks/m1_colab_cuda_nf4.ipynb)
contains the completed T4 audit. It ran each requested FP16/NF4 allocation from
a fresh checkpoint process, audited realized modules and weight storage,
performed one synthetic inference, and produced a grid summary. This is a
feasibility audit, not an accuracy experiment.

## M2 data pipeline

The M2 Python pipeline normalizes an explicitly selected POPE-COCO variant or
explicit CelebA attributes, verifies source images, creates deterministic
image/identity-disjoint manifests, expands clean/darkness/Gaussian-blur
variants, and validates model-independent pilot answerability decisions. See
[the M2 pipeline handoff](docs/M2_PIPELINE.md) and
[the real-data validation](docs/M2_REAL_DATA_VALIDATION.md).

POPE-COCO is validated against the official files: 3,000 questions over 500
images per variant, balanced 1,500/1,500, no duplicate IDs, no missing images.
CelebA annotation files match their published MD5 values, but the images are not
yet downloaded. Dataset choices, severities, and memory caps remain pending
pilot validation and are not frozen.

## Data and outputs

Keep dataset images under `data/`, model weights under `models/`, and generated results under `artifacts/` or `runs/`. These paths are ignored by Git. Commit code, configuration, small permitted split manifests, and report summaries deliberately. Do not commit CelebA images, model weights, credentials, or full prediction dumps.

## Original proposal

[Google Docs — Final tab](https://docs.google.com/document/d/1gtlE4NJRYpqTqe1G5gV5SIBgzHL_DG9FJyWphVvco9Q/edit?tab=t.9i03s424878a)

The repository expands the proposal into implementation instructions. It does not claim professor approval. Changes to the scientific scope must be recorded in the decision log.
