# M2 Data and Corruption Pipeline

Date: 2026-10-28

Status: implemented and synthetic-unit-tested; real POPE-COCO and CelebA
schemas, counts, access, balance, and pilot choices still require Gauri's
validation.

## Purpose

M2 converts source annotations into a common yes/no record, assigns indivisible
groups to pilot/development/test before any corruption is created, and expands
those records into deterministic clean/darkness/Gaussian-blur variants.

The implementation enforces leakage and reproducibility rules. It deliberately
does not choose the POPE variant, CelebA attributes, split fractions, random
seed, or corruption severities. Those remain explicit inputs that must be
validated and recorded before the main experiment.

## Pipeline

```text
POPE JSONL ───────┐
                  ├─> normalized records ─> group split ─> variant manifest
CelebA metadata ──┘          │                  │                │
                             │                  │                ├─ clean
                             │                  │                ├─ darkness
                             │                  │                └─ Gaussian blur
                             │                  │
                     image SHA-256       leakage report
                                                                │
                                                   pilot answerability review
```

The split happens before variant expansion. POPE is grouped by original COCO
image. CelebA is grouped by identity, which also keeps every image of the same
person together.

## Normalized record

Every question contains:

```text
dataset
original_image_id
image_relpath
question_id
question
label                         # yes or no
identity_id                   # required for CelebA; null for POPE
source_variant                # selected POPE variant
attribute                     # selected CelebA attribute
source_image_sha256           # required for finalized real-data manifests
```

Absolute image paths are rejected. Manifests use portable paths relative to an
explicit dataset image root. Raw data, images, generated variants, and
manifests remain ignored and must not be committed.

Each normalization command also writes an immutable `.metadata.json` sidecar
containing source annotation paths and SHA-256 values, field/attribute choices,
the output-manifest SHA-256, and validation results.

## POPE normalization

The POPE loader accepts exactly one explicitly named sampling variant. Default
source fields are `image`, `text`, `label`, and `question_id`; every field name
can be overridden after Gauri inspects the actual annotation schema.

```bash
mpvs m2-normalize-pope \
  --annotations /path/to/selected-pope-variant.jsonl \
  --image-root /path/to/coco/images \
  --variant VARIANT_CONFIRMED_BY_TEAM \
  --output artifacts/m2/pope-normalized.jsonl
```

The loader never merges POPE variants implicitly. Missing fields, invalid
labels, duplicate question IDs, missing images, and unsafe paths fail loudly.

## CelebA normalization

The CelebA loader reads the official-style attribute and identity metadata.
Every selected attribute requires an explicit natural-language question so the
code does not invent attribute semantics.

```bash
mpvs m2-normalize-celeba \
  --attributes /path/to/list_attr_celeba.txt \
  --identities /path/to/identity_CelebA.txt \
  --image-root /path/to/img_align_celeba \
  --attribute-question 'Eyeglasses=Is this person wearing eyeglasses?' \
  --output artifacts/m2/celeba-normalized.jsonl
```

The team must choose visible attributes and a balancing/sampling rule after
checking the source data and terms. The loader only normalizes the explicitly
selected attributes; it does not silently rebalance them.

## Deterministic group splitting

```bash
mpvs m2-split \
  --input artifacts/m2/pope-normalized.jsonl \
  --output-dir artifacts/m2/pope-splits-v1 \
  --seed SEED_TO_FREEZE \
  --pilot-fraction PILOT_FRACTION \
  --development-fraction DEVELOPMENT_FRACTION \
  --test-fraction TEST_FRACTION
```

Assignment is independent of input row order. Groups are ordered by a SHA-256
hash of dataset, group ID, and seed, then allocated as whole groups. Each
positive split receives at least one group when enough groups exist. Exact
question counts can differ from target fractions because whole-group integrity
takes priority.

The command writes immutable `pilot.jsonl`, `development.jsonl`, `test.jsonl`,
and `split-summary.json` files. Repeating the identical run is allowed;
attempting to replace an existing manifest with different content fails.
The summary records the seed, requested fractions, achieved record/group/label
counts, and SHA-256 of each manifest.

Independent validation:

```bash
mpvs m2-validate-splits \
  artifacts/m2/pope-splits-v1/pilot.jsonl \
  artifacts/m2/pope-splits-v1/development.jsonl \
  artifacts/m2/pope-splits-v1/test.jsonl
```

The validator rejects duplicate question IDs and any original-image, CelebA
identity, or group leakage.

## Corruption variants

Corruptions are deterministic Pillow operations on RGB source images before
the model processor performs resizing. Darkness uses a multiplicative
brightness factor strictly between 0 and 1. Gaussian blur uses Pillow's
positive radius convention. The library and parameter convention are recorded
so values are not confused with another implementation's kernel or sigma.

The corruption configuration is a JSON array. The following is a synthetic
format example, not a proposed final severity choice:

```json
[
  {"kind": "clean", "severity_id": "clean", "value": null},
  {"kind": "darkness", "severity_id": "synthetic-dark", "value": 0.5},
  {"kind": "gaussian_blur", "severity_id": "synthetic-blur", "value": 1.0}
]
```

Exactly one clean specification is required. The real M2 configuration should
contain three pilot-checked levels for darkness and three for blur only after
those values are selected without test accuracy.

```bash
mpvs m2-expand-variants \
  --input artifacts/m2/pope-splits-v1/pilot.jsonl \
  --corruptions /path/to/pilot-corruptions.json \
  --output artifacts/m2/pope-pilot-variants.jsonl
```

Every variant repeats the base question's split and group and records its
corruption parameters and transform stage. Images can be transformed on demand
with `mpvs.corruptions.apply_corruption`; pre-rendering duplicate images per
question is unnecessary. A metadata sidecar records the base-manifest hash,
exact corruption specifications, variant-manifest hash, and inheritance report.

## Model-independent answerability review

Severity decisions must use pilot examples and must not depend on whether the
VLM answered correctly. A review decision JSONL row has:

```json
{
  "schema_version": 1,
  "variant_id": "question-id::darkness::severity-id",
  "answerable": false,
  "review_method": "human_model_blind_review",
  "reason": "the queried object is no longer visually discernible"
}
```

Unanswerable decisions require a reason. Validate one decision per corrupted
pilot variant with:

```bash
mpvs m2-validate-answerability \
  --variants artifacts/m2/pope-pilot-variants.jsonl \
  --decisions artifacts/m2/pope-pilot-answerability.jsonl
```

The validator rejects missing, duplicate, unknown, and non-pilot decisions.
This records the review process; it does not automate semantic judgment.

## Code map

| Module | Responsibility |
| --- | --- |
| `src/mpvs/data/records.py` | Normalized and split-assigned record contracts |
| `src/mpvs/data/pope.py` | Strict one-variant POPE JSONL normalization |
| `src/mpvs/data/celeba.py` | Attribute and identity parsing with explicit questions |
| `src/mpvs/data/images.py` | Source-image existence and SHA-256 verification |
| `src/mpvs/data/manifests.py` | Canonical, atomic, immutable manifest I/O |
| `src/mpvs/splits.py` | Deterministic group assignment and leakage validation |
| `src/mpvs/corruptions.py` | Deterministic transforms and inherited variants |
| `src/mpvs/data/review.py` | Model-independent pilot answerability decisions |
| `src/mpvs/data/cli.py` | M2 command-line entry points |

Synthetic fixtures are in `tests/test_m2_loaders.py`, `tests/test_splits.py`,
`tests/test_corruptions.py`, and `tests/test_answerability.py`.

## Gauri's real-data validation handoff

Before M2 is accepted on actual data, validation should record:

1. Official/source file locations, access terms, hashes, and actual schemas.
2. The single selected POPE variant and observed image/question/label counts.
3. CelebA attribute polarity, identity coverage, selected visible attributes,
   natural-language questions, and any sampling/balance rule.
4. Missing/duplicate image and annotation checks.
5. Source-image checksum completion.
6. Requested versus achieved split records, groups, and label counts.
7. Zero original-image leakage and zero CelebA identity leakage.
8. Byte-identical manifests from repeated runs with the same seed.
9. Byte-identical transformed image pixels for repeated calls.
10. Pilot-only, model-independent answerability decisions for proposed
    corruption severities.

Synthetic test success is implementation evidence, not real-dataset evidence.
Do not mark M2 complete until these checks run on the authorized data and the
material choices are recorded in `docs/DECISIONS.md`.
