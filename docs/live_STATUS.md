# Current status

Updated: 2026-11-26.

## Simple summary

- M1 is complete. All eight FP16/NF4 model configurations were checked on a
  Tesla T4.
- The M2 data-processing and placeholder CelebA sampling code is implemented
  and has 48 passing tests.
- POPE-COCO annotations and all 500 source images were validated.
- The private CelebA archive and annotations passed their published MD5 checks.
  The balanced, checksum-verified `celeba-core-v1` sample, splits, corruption
  variants, decisions, and provenance are frozen by recorded private hashes.
- The final CelebA pilot has 98 clean and 392 corrupted variants. Its mapped
  model-blind review is valid: 388 answerable and 4 unanswerable, with no
  missing, duplicate, unknown, or non-pilot IDs.
- CelebA final manifests are frozen. POPE final manifests, POPE severities,
  POPE split choices, and experimental memory caps are not frozen.
- M3 inference, M4 analysis, and the main experiment have not started.

## Current milestone

The CelebA portion of M2 is complete and frozen as `celeba-core-v1`. POPE-COCO
final manifests and its human corruption review remain pending, so overall M2
is not complete.

## What is complete, partial, and not started

| Work | Status | Meaning |
| --- | --- | --- |
| M1 model audit | Complete | Eight distinct configurations loaded, matched the request, and passed synthetic inference. |
| POPE-COCO validation | Complete | Real annotations and 500 images were checked. |
| CelebA validation | Complete for M2 | The private 1,200-record `celeba-core-v1` sample and selected images are checksum-verified; evidence hashes are tracked. |
| Final splits and balance | Partial overall | CelebA 98/605/497 pilot/development/test artifacts are frozen privately; POPE final manifests remain pending. |
| CelebA sampler | Complete for M2 | Exact 200/200 yes/no balance for each of three attributes, one question per image, and seed `590` are frozen in `celeba-core-v1`. |
| Corruption pilot | Partial overall | CelebA has four accepted levels and 392 valid decisions; POPE review remains pending. |
| Final memory caps | Not selected | Candidate memory sizes are measured, but experimental caps are not frozen. |
| M3 inference/evaluation | Not started | Only the M1 synthetic smoke test exists. |
| M4 statistics and main results | Not started | There are no dataset accuracy or delta results. |

## Team Drive and private data

Authorized team workspace:
[mixed-precision-visual-shift Google Drive](https://drive.google.com/drive/folders/1qjmMDGkSdEbg287UPCwk-Ct8Qb0UfE2E?usp=drive_link).

The Drive currently contains:

- `artifacts/` — generated private manifests, validation reports, review files,
  hashes, and run summaries;
- `celeba/` — the private CelebA source archive and annotations; and
- `colab notebooks/` — the two notebooks used for the M1 GPU audit and M2
  CelebA processing.

Access is limited to the team email accounts. If the folder does not open,
request access from the owner instead of making a public copy. Do not change the
sharing setting of the CelebA folder or review packages.

The GitHub repository may be private, but that does **not** change the data
rules. Do not commit raw datasets, CelebA images, private manifests, review
packages, access tokens, large artifacts, or model checkpoints. Commit only
code, small non-restricted configurations, counts, and hashes. When Colab needs
to clone the private repository, authenticate with an authorized team GitHub
account or a Colab secret; never paste a token into a notebook cell or Git.

## Plain-language terms used in this project

### Freeze

To **freeze** something means the team has finished the pilot check, accepted
the exact values/files, and will not silently change them during evaluation.
Frozen does not mean the file is technically locked. It means:

- the accepted choice is written in `docs/DECISIONS.md`;
- the exact file is identified by a SHA-256 hash;
- later runs must use that same choice and hash; and
- any change requires a new version name, new hashes, and a new decision entry.

Example: `celeba-core-v1` is frozen. Do not overwrite it. A changed CelebA
sample must be named something like `celeba-core-v2` and reviewed again.

### Manifest

A **manifest** is a small JSONL inventory describing what the experiment will
use. It does not contain the image pixels. A row normally records the image
path/ID, question, yes/no label, group or identity ID, split, and image
checksum. Split manifests tell the runner exactly which records belong to:

- `pilot` — small data used to test feasibility and choose settings;
- `development` — data used to select the best model configuration; or
- `test` — untouched data used only for final evaluation.

All questions for one POPE image stay in one split. All records for one CelebA
identity stay in one split.

### Candidate

The word **candidate** is used in three related ways:

- A **model candidate** is one of the eight audited FP16/NF4 allocations across
  the vision encoder, connector, and language backbone.
- A **severity candidate** is a darkness or blur value being considered during
  the pilot. It is not final until the human answerability review accepts it.
- A **final-candidate manifest** is a regenerated, versioned manifest that has
  passed validation but is waiting for its hashes/decision to be frozen.

Do not confuse a requested model candidate with an audited working
configuration. M1 verified all eight actual configurations on the T4.

### Artifact, checksum, and memory cap

- An **artifact** is any generated output: a manifest, JSON report, prediction
  cache, review page, or run summary.
- A **checksum/hash** is a digital fingerprint. Matching SHA-256 values mean
  teammates used the same file bytes.
- A **memory cap** is the maximum measured model-weight storage allowed in one
  comparison. Every clean/target/mixed selection method must choose from the
  same candidates that fit that cap.

## Fixed choices and choices still pending

### Frozen for CelebA

- Run: `celeba-core-v1`.
- Attributes: `Eyeglasses`, `Wearing_Hat`, and `Smiling`.
- Sampling: 200 yes and 200 no per attribute; one question per image.
- Seeds: sampling `590`, splitting `590`.
- Split: 98 pilot, 605 development, 497 test records.
- Conditions: clean, darkness `0.60`/`0.35`, Gaussian blur `0.5`/`1.0`.
- Human review: 392/392 decisions valid; 388 answerable and 4 unanswerable.
- Evidence: `configs/m2/celeba-core-v1-evidence.json` and the private hashed
  Drive artifacts.

### Still proposed for POPE-COCO

- Variant: `adversarial`.
- Split seed: `590`.
- Possible split: 17 pilot images (102 questions), 399 development images
  (2,394 questions), and 84 test images (504 questions). One image and all six
  questions must stay together.
- Pilot severity candidates: darkness `0.60`, `0.35`, `0.15`; Gaussian blur
  `1.5`, `3`, `6`. These are not accepted yet and must receive a POPE-specific
  visual review.

### Still pending for M3

- final memory caps;
- prompt and image-processing settings;
- strict yes/no parsing rule;
- decoding settings;
- prediction/cache schema implementation; and
- full-run compute estimate.

Measured M1 cap examples are `2,178,324,832` bytes (four candidates),
`3,902,666,464` bytes (six), and `4,493,569,888` bytes (all eight). Choose caps
before looking at comparative test accuracy.

## STEPS for the next teammate

### Step 1 — Get oriented and verify access

1. Clone or pull the private repository using an authorized team GitHub
   account. Do not start from an old notebook copy without checking the commit.
2. Read, in order: `docs/README.md`, this status file,
   `docs/PROJECT_PLAN.md`, and `docs/DECISIONS.md`.
3. Open the team Drive link above and confirm that `artifacts/`, `celeba/`, and
   `colab notebooks/` are visible.
4. Run `git status --short --branch` before editing. Coordinate with the team if
   another person has uncommitted changes in the same working tree.
5. Do not modify or delete
   `artifacts/celeba-core-v1`. Its hashes are already frozen.

### Step 2 — Locate the POPE inputs

The official POPE annotations and 500 referenced COCO images were validated in
an earlier run (we have evidence that Gauri previously checked the POPE data, but the actual POPE/COCO files were not committed to Git), but the next teammate must confirm where those private/local
files are stored. Do not assume they are present merely because this status
says validation succeeded.

Required input facts for one POPE variant are:

- 3,000 questions;
- 500 unique COCO images;
- 1,500 `yes` and 1,500 `no` labels;
- six questions per image; and
- no missing images or duplicate question IDs.

If the source files are not in the team Drive or authorized local storage,
stop and obtain them from the teammate who performed the validation or rerun
the documented official-source acquisition. Do not substitute an unknown copy.

### Step 3 — Accept one POPE variant and create the split

`adversarial` is the current proposed variant. Confirm its source counts and
record acceptance in `docs/DECISIONS.md` before using model accuracy. Then run
the M2 commands from the repository root (replace the example paths):

```bash
python -m pip install -e .

mpvs m2-normalize-pope \
  --annotations /private/path/to/coco_pope_adversarial.json \
  --image-root /private/path/to/coco-val2014 \
  --variant adversarial \
  --output /private/path/to/pope-core-v1/pope-normalized.jsonl

mpvs m2-split \
  --input /private/path/to/pope-core-v1/pope-normalized.jsonl \
  --output-dir /private/path/to/pope-core-v1/splits \
  --seed 590 \
  --pilot-fraction 0.034 \
  --development-fraction 0.798 \
  --test-fraction 0.168

mpvs m2-validate-splits \
  /private/path/to/pope-core-v1/splits/pilot.jsonl \
  /private/path/to/pope-core-v1/splits/development.jsonl \
  /private/path/to/pope-core-v1/splits/test.jsonl
```

The fractions target about 17/399/84 whole images. Record the achieved counts;
whole-image grouping matters more than an exact fraction. The validation report
must show no image, question, group, or checksum leakage.

### Step 4 — Create and review POPE corruption candidates

Create a JSON corruption file with exactly one clean condition and the proposed
POPE darkness/blur candidates. Expand **only the pilot split** first:

```bash
mpvs m2-expand-variants \
  --input /private/path/to/pope-core-v1/splits/pilot.jsonl \
  --corruptions /private/path/to/pope-pilot-corruptions.json \
  --output /private/path/to/pope-core-v1/pilot-variants.jsonl

python scripts/m2_pilot_review.py \
  --variants /private/path/to/pope-core-v1/pilot-variants.jsonl \
  --image-root /private/path/to/coco-val2014 \
  --output-dir /private/path/to/pope-core-v1/pilot-review
```

Open the private review page and judge only whether each question is still
answerable by a human. Do not look at model predictions. Leave a box checked if
the question remains visually answerable; uncheck it when darkness/blur hides
the queried object. Review every corrupted question, export the decisions, and
validate them:

```bash
mpvs m2-validate-answerability \
  --variants /private/path/to/pope-core-v1/pilot-variants.jsonl \
  --decisions /private/path/to/pope-core-v1/pilot-answerability.jsonl
```

Select or reject whole severity levels. Do not delete individual questions to
improve scores. If review choices violate the expected mild-to-strong order,
re-inspect those exact panels and record any correction.

### Step 5 — Freeze `pope-core-v1`

After choosing the accepted POPE levels:

1. Create a non-placeholder corruption configuration under `configs/m2/`.
2. Regenerate the POPE variants using only clean plus accepted levels.
3. Rerun split leakage, variant completeness, checksum, and determinism checks.
4. Save private outputs under a new Drive artifact directory named
   `pope-core-v1`; never overwrite a pilot/placeholder directory.
5. Calculate SHA-256 for the normalized manifest, split summary, corruption
   config, pilot variants, answerability decisions, and run summary.
6. Commit only a small evidence JSON containing counts and hashes.
7. Record the accepted variant, split, severities, evidence, and limitations in
   `docs/DECISIONS.md` and update this file.

Overall M2 is complete only when both `celeba-core-v1` and `pope-core-v1` are
frozen and reproducible by their recorded hashes.

### Step 6 — Begin M3 only after M2 is complete

M3 builds the model runner; it is not the main experiment yet.

1. Implement the fixed prompt, image preprocessing, greedy decoding, and strict
   yes/no parser.
2. Write one prediction JSONL record per question, condition, and audited model
   candidate.
3. Add atomic, resumable caching. Cache keys must include model/processor
   revisions, effective quantization signature, image checksum, corruption,
   question/prompt, and decoding settings.
4. Add tests for cache hits, incompatible keys, invalid answers, failures, and
   resume behavior.
5. Choose common measured memory caps before examining comparative test
   accuracy. Every selection method at a cap uses the same eligible candidates.
6. Run a small disjoint pilot at batch size 1 on the T4. Measure runtime, peak
   GPU memory, invalid-answer rate, and cache/resume correctness.
7. Freeze the prompt, parser, caps, decoding settings, and protocol hash only
   after the pilot passes. Do not run the full experiment before this point.

### Stop and ask for help when

- a required private source file or authorized Drive folder is missing;
- any checksum differs from its frozen value;
- a split leaks an image, identity, group, or question;
- a repeated run produces different manifest or transformed-image bytes;
- the answerability decision file is incomplete or invalid;
- an accepted model candidate fails its audited configuration or inference;
- a cache key could reuse a prediction from different settings; or
- completing the task would require tuning choices using test accuracy.

### End-of-session handoff

Before stopping work, update this file with the exact commands, outputs,
limitations, and next task. Put material protocol choices in
`docs/DECISIONS.md`. State whether work was only implemented, unit-tested,
GPU-tested, or experimentally completed. Never claim completion from synthetic
tests alone.

## Possible future corruption extensions

These are optional extensions, not requirements for the current experiment.
Complete the reproducible darkness/Gaussian-blur experiment first. Report any
later corruption family as exploratory unless a new protocol version promotes
it to a prespecified experiment.

Possible families include:

- geometric shift, defined precisely as affine shear, rotation, or perspective
  distortion rather than the vague term `skewness`;
- occlusion or pixel dropout, defined by a deterministic mask rather than the
  vague term `missing bits`;
- JPEG compression with an explicit encoder/library and quality value; and
- additive sensor-style noise with an explicit distribution, scale, and seed.

Before adding any family:

1. State the scientific reason for it and whether it models acquisition,
   transmission, geometry, or missing visual content. Do not add corruptions
   only because they are easy to generate.
2. Define the transform, parameter units, allowed range, library version,
   transform order, color space, and whether it runs before model-processor
   resizing. A name such as `missing bits` is not reproducible by itself.
3. Reject transforms that silently change the ground-truth answer, crop away
   the queried feature, expose padding artifacts as shortcuts, or affect one
   question for an image differently from another question for that image.
4. For random transforms, derive the random seed from the immutable source
   image/group ID and corruption specification. Store the seed or mask
   signature so repeated runs produce byte-identical pixels.
5. Add explicit `CorruptionSpec` validation and implementation in
   `src/mpvs/corruptions.py`. Include every parameter in the corruption
   signature and therefore in prediction-cache keys.
6. Add synthetic tests for parameter bounds, deterministic pixels, stable
   signatures, complete paired variants, split/group inheritance, and invalid
   configurations. Keep every question and candidate configuration paired on
   the same transformed source image.
7. Pilot candidate severities separately for POPE-COCO and CelebA when image
   size or task semantics change their effect. Use only pilot images and a
   model-blind human answerability review; never tune with development/test
   model accuracy.
8. Select or reject whole severity levels using a written pilot rule. Do not
   remove individual examples because a model answered incorrectly. Record
   accepted values, rejected values, reviewer evidence, and hashes in
   `docs/DECISIONS.md` before evaluation.
9. Re-estimate variant counts, runtime, storage, Colab limits, and statistical
   comparisons. More corruption families multiply inference cost and increase
   the number of reported comparisons; do not delay the core experiment for an
   optional extension.
10. Write extension outputs to a new versioned artifact directory and protocol
    hash. Do not overwrite or mix them into the frozen darkness/blur results.

The smallest safe extension is one precisely defined family with a few
pilot-checked severities. Run it only after M3 caching and the core M4 analysis
are working end to end.

## Possible future CelebA attribute extensions

The core decision remains three attributes (`Eyeglasses`, `Wearing_Hat`, and
`Smiling`) with one selected question per source image. This keeps the image
count equal to the question count, prevents a few faces from dominating the
sample, and makes the first selection experiment easier to interpret.

After the core experiment is reproducible, a new protocol version may test
more visible attributes or multiple attribute questions for the same image.
This could measure whether a precision allocation transfers across different
visual concepts, but it changes the sampling unit and statistical dependence.

Requirements for adding more attributes:

1. Choose attributes that are visually observable and have clear yes/no
   wording. Record source prevalence, identity coverage, mixed labels within an
   identity, and any subjective or ambiguous semantics before inclusion.
2. Avoid adding attributes only to increase question count. Each attribute
   needs enough positive and negative identities for the same documented
   balancing and identity-cap policy.
3. Pilot-check clean-label visibility and corruption answerability separately
   for every new attribute. Do not select attributes using model accuracy or
   test results.
4. Keep the original three-attribute result as the core analysis. Report a
   larger attribute set as an extension or a new protocol version so the
   denominator and task composition are not silently changed.

Requirements for allowing multiple attributes on one image:

1. Add an explicit sampler option such as `max_questions_per_image`; do not
   silently remove the current one-question-per-image rule. Balance each
   attribute and label while also limiting images per identity.
2. Permit the same `original_image_id` only when its records have different
   question/attribute IDs. Continue rejecting exact duplicate questions and
   conflicting image paths, identities, labels, or checksums.
3. Assign the image and its identity to one split before creating questions or
   corruptions. Every question and transformed variant for that image must stay
   in the same pilot/development/test split.
4. Construct clean/target development mixtures by image or identity group, not
   independently by question. Otherwise one image could appear in both visual
   conditions and inflate the effective sample size.
5. Report both unique-image and question counts. Treat questions from the same
   face as correlated; keep identity-clustered uncertainty for CelebA rather
   than treating each question as an independent observation.
6. Include the question ID and attribute in prediction-cache keys while sharing
   the same source-image checksum and corruption signature. Preserve paired
   predictions for every model configuration.
7. Extend synthetic tests for deterministic sampling, per-attribute balance,
   allowed cross-attribute image reuse, forbidden duplicate questions,
   identity leakage, complete variants, and stable cache keys.
8. Repeat the private, model-blind answerability review for every new
   image-question-corruption combination and re-estimate inference cost before
   scaling.

Do not mix this extension into the current `celeba-corruptions-v1` artifacts or
rename the current placeholder run as extended data. Use new versioned
manifests, decisions, protocol hashes, and result tables.

## Detailed work log

## 2026-11-26 — Added teammate-ready access, glossary, and next-step handoff

- Added the authorized team Google Drive link and described its `artifacts/`,
  `celeba/`, and `colab notebooks/` folders.
- Defined `freeze`, `manifest`, the three meanings of `candidate`, `artifact`,
  checksum, split roles, and memory cap in simple project-specific language.
- Replaced the short next-step list with an executable POPE-to-M3 handoff:
  access checks, source-count checks, command templates, human review, freeze
  requirements, M3 implementation order, stop conditions, and end-of-session
  reporting.
- Clarified that a private GitHub repository does not authorize committing raw
  CelebA/COCO data, private manifests, review packages, tokens, checkpoints, or
  large artifacts. Those remain in the team Drive or authorized local storage.
- Documentation only. No source data, manifest, experiment configuration,
  model run, or private artifact changed.

## 2026-11-26 — Split active and passive documentation

- Reduced the frequently updated working set in `docs/` to the status,
  decisions, one merged project-plan/protocol document, and a short
  reading-order index.
- Merged the former implementation plan and experiment protocol into
  `docs/PROJECT_PLAN.md` while preserving separate milestone and scientific
  control sections.
- Moved stable historical/reference material to `passive_docs/`: the proposal
  snapshot, M1 findings, completed CelebA setup guide, earlier M2 validation
  report, M2 technical pipeline, Git workflow, and midterm progress document.
  Added a passive-document index.
- Updated repository links and agent instructions to use
  `docs/live_STATUS.md`; the previously referenced `docs/STATUS.md` did not
  exist. Passive documents do not override the active status, decisions, or
  merged project plan/protocol.
- Documentation organization only. No code, private artifact, manifest,
  experiment setting, or result changed.

## 2026-11-26 — Froze private `celeba-core-v1` M2 artifacts

- Regenerated the accepted 1,200-record CelebA design under the non-placeholder
  run name `celeba-core-v1` with clean plus darkness `0.60`/`0.35` and Gaussian
  blur `0.5`/`1.0`. The sample manifest hash matches the accepted pilot sample.
- Generated 490 pilot variants. Mapped the existing model-blind decisions by
  exact question ID, corruption kind, and numeric value to the final severity
  IDs; validation accepted all 392 decisions with 388 answerable and 4
  unanswerable and no ID/coverage failures.
- Recorded the private sample, split summary, variants, decisions, mapping, and
  run-summary hashes in `configs/m2/celeba-core-v1-evidence.json`. Restricted
  data remains outside Git and was not independently inspected in the repo.
- CelebA M2 is complete. Overall M2 remains open for the POPE final manifest and
  model-blind corruption review.

## 2026-11-26 — Documented future CelebA attribute extensions

- Kept the core final-candidate design at three attributes and one question per
  selected image.
- Documented how a later protocol could add visible attributes or multiple
  questions per image without breaking balance, identity-disjoint splits,
  pairing, cache correctness, model-blind review, or clustered uncertainty.
- Documentation only. The sampler and current CelebA sample were not changed;
  the extension remains outside the core experiment.
- Recorded the current attributes, question wording, 200-per-label rule, seeds,
  split fractions, and achieved 98/605/497 allocation as accepted
  final-candidate choices in `docs/DECISIONS.md`.

## 2026-11-26 — Accepted two CelebA severities per corruption

- User-reported Colab evidence shows that the CelebA archive and two annotation
  files matched their published MD5 values. The private run sampled and
  checksum-verified 1,200 images, then created a validation-passing placeholder
  split with 98 pilot, 605 development, and 497 test records.
- The model-blind pilot review contained all 588 required corrupted decisions:
  no missing, duplicate, unknown, or non-pilot IDs. Its first validated version
  reported 559 answerable and 29 unanswerable variants.
- Accepted for the CelebA final candidate: darkness factors `0.60` and `0.35`,
  plus Pillow Gaussian-blur radii `0.5` and `1.0`. Rejected as whole levels:
  darkness `0.15` (87.76% answerable before correction) and blur `2.0`
  (88.78%). No individual record is removed because of this decision.
- One question caused three monotonicity flags. The reviewer re-inspected it
  and marked all six versions answerable. The corrected private file contains
  561 answerable and 27 unanswerable decisions with no missing/unknown IDs or
  remaining monotonicity violations. Every accepted level is 97/98 (98.98%)
  answerable.
- Added `configs/m2/celeba-corruptions-v1.json` and recorded the decision in
  `docs/DECISIONS.md`. Updated `docs/PROJECT_PLAN.md` to permit up to
  three dataset-specific levels instead of requiring exactly three.
- Added the user-reported private-artifact counts and SHA-256 values to
  `configs/m2/celeba-pilot-evidence-v1.json`; restricted records remain outside
  Git and were not independently inspected in the repository.
- Next: approve the CelebA sample/split choices, generate new versioned final
  manifests with the accepted config, complete the POPE human review, and then
  begin M3 inference/cache implementation.

## 2026-11-26 — Documented optional corruption-extension requirements

- Added a future-only checklist for geometric distortion, deterministic
  occlusion/pixel dropout, JPEG compression, and seeded noise.
- Documented the required protocol, implementation, cache-key, pairing,
  determinism, human-review, compute-budget, and versioning changes.
- No additional corruption was implemented or approved. Darkness and Gaussian
  blur remain the core experiment, and optional extensions remain blocked on a
  reproducible core M3/M4 run.

## 2026-11-25 — CelebA Colab notebook mount-cell repair

- Restored the executable Google Drive mount and directory-initialization code
  in `notebooks/m2_celeba_colab_temp.ipynb`; the cell had been commented out.
- Confirmed `origin/Gauri-Commits` contains `pyproject.toml` and
  `src/mpvs/data/sampling.py`, which the notebook checks before installation.
- Validation: parsed the notebook as JSON, compiled every code cell with
  Python's AST parser, and ran `git diff --check`; all passed.
- The notebook has not been rerun in Colab after this repair. CelebA source
  downloads, checksums, archive contents, sampling, and splits remain pending.

## 2026-11-19 — Placeholder CelebA sampler and low-level handoff

- Added `m2-sample-celeba` with explicit attributes, per-label count, and seed.
- The sampler creates exact yes/no balance per attribute, uses at most one image
  per identity per attribute, selects one question per image, and avoids
  duplicate image bytes when checksums are present.
- Final sampling requires image checksums. Annotation-only testing requires the
  explicit `--allow-missing-image-checksums` flag.
- Added duplicate-image checksum reporting to normalized-record validation.
- Added `m2-verify-images` so an annotation-only sample can be checked after
  extracting only its selected images. This avoids a full 202,599-image
  extraction while still requiring checksums before split creation.
- Added `notebooks/m2_celeba_colab_temp.ipynb`. It keeps the official ZIP in a
  private Drive folder, checks the published MD5 values, inspects all ZIP image
  entries, extracts only the placeholder sample, verifies it, creates trial
  splits and pilot variants, and exports small private artifacts.
- The official Google Drive files later hit their public download quota. The
  notebook now tries the clearly labelled Freie Universität Berlin academic
  mirror after an official-download failure. Every fallback file must match the
  published torchvision MD5 before it is copied to private Drive storage. This
  path is implemented but has not yet been run; the images remain unvalidated.
- Added `passive_docs/M2_CELEBA_SETUP.md` with download, archive, count, checksum,
  normalization, sampling, split, leakage, and visual-review instructions.
- Validation: `conda run -n mpvs-m1 python -m compileall -q src tests` passed;
  `conda run -n mpvs-m1 python -m pytest -q` passed 48 tests; all 23 notebook
  cells parsed and all code cells compiled.
- Status: implemented and synthetic-tested only. CelebA images were not
  downloaded, and placeholder values are not frozen experiment decisions.
- Next: download the authorized images, run normalization with checksums, run
  the sampler with team-approved values, and inspect the generated metadata.

## 2026-10-28 — Gauri M2 real-data validation, Phase A

- Followed Kaushal's M2 handoff runbook. Full report:
  [M2_REAL_DATA_VALIDATION.md](../passive_docs/M2_REAL_DATA_VALIDATION.md).
- POPE-COCO (validated on real data): downloaded the three official POPE COCO
  files and the 500 referenced COCO val2014 images. All variants have 3000
  questions, 500 images, 1500/1500 yes/no, 6 questions per image, no duplicate
  IDs or missing images; the loader defaults matched the schema with no code
  change. Normalization is deterministic.
- CelebA (annotations validated on real data, images not downloaded): the
  official attribute and identity files match the published MD5 values; 202,599
  images, 40 attributes, 10,177 identities, every image has an identity. 615
  identities have mixed Male labels (dataset label noise).
- Trial splits (placeholder seed 590) on POPE adversarial and CelebA
  Eyeglasses: zero leakage, byte-identical on rerun. Trial POPE pilot
  corruptions: 420 variants, completeness valid, 0 pixel mismatches across two
  renders.
- Added `scripts/m2_pilot_review.py`, which renders pilot variants, checks pixel
  determinism, and writes a local review page that exports decisions for
  `m2-validate-answerability`.
- Proposed (awaiting team): POPE `adversarial`; CelebA Eyeglasses /
  Wearing_Hat / Smiling; a per-attribute, identity-capped balanced sampling
  rule; split sizes and seed. Nothing is frozen.
- Not done: CelebA image download, final manifests (Phase B), human
  answerability review (Phase C).
- Next: the team decides the items in section 5 of the report; then implement
  the approved CelebA sampling rule, create final manifests, and run the pilot
  answerability review.

## 2026-10-28 — Gauri M2 review: paired-variant completeness and duplicate-image leakage

- Reviewed `src/mpvs/splits.py` and `src/mpvs/corruptions.py` on `Kaushal-Commits`
  (`cb9feb6`). The existing 17 M2 tests passed. Two validation gaps were found
  with small synthetic probes:
  1. `validate_variant_inheritance` reported `valid: true` when some
     clean/darkness/blur variants were missing, which breaks paired evaluation.
  2. `validate_assigned_records` and `split_records` did not detect or prevent
     the same image bytes (same `source_image_sha256`) appearing under two image
     IDs or two CelebA identities in different splits.
- Fix 1: `validate_variant_inheritance(base, variants, specs=None)` now reports
  `missing_variants` (every base question must have every expected condition),
  `inconsistent_corruption_specs` (one kind/severity_id used with different
  values), and `undeclared_corruption_specs`. The `expand-variants` CLI passes
  the declared specs, so a condition dropped for every question is also caught.
- Fix 2: `split_records` merges groups that share a source-image checksum and
  assigns the merged unit as one indivisible group; the unit's ordering key is
  its smallest member group ID, so data without shared checksums keeps exactly
  the previous assignment. `validate_assigned_records` now reports
  `source_image_checksum_leakage`.
- Added 9 synthetic-fixture tests (`tests/test_corruptions.py`,
  `tests/test_splits.py`). Confirmed 8 of them fail on the previous code; the
  ninth checks that assignments are unchanged when no checksums are shared.
- Command: `PYTHONPATH=src python -m pytest -q` (Python 3.14 venv, macOS, CPU
  only) → 40 passed. Implemented and unit-tested on synthetic data only; not run
  on real POPE-COCO/CelebA files.
- Not changed (open for team discussion): `CorruptionSpec.signature` differs for
  `2` vs `2.0` and accepts `True`; Gaussian blur radius is in pixels, so one
  radius is relatively stronger on CelebA (~178 px wide) than COCO (~640 px).
- Next: run the split/leakage/completeness checks on the real manifests once the
  authorized POPE-COCO and CelebA files are available.

## 2026-10-28 — M2 Python processing architecture

- Added strict normalized record contracts for POPE-COCO and CelebA. POPE
  requires an explicit sampling variant and configurable verified source-field
  names; CelebA requires official-style identity/attribute files and explicit
  attribute questions.
- Added source-image existence/SHA-256 verification, canonical JSONL encoding,
  atomic writes, and immutable manifests that permit identical reruns but reject
  replacement with different content. Normalization and variant sidecars bind
  outputs to source annotation/base-manifest hashes and explicit settings.
- Added deterministic group splitting from a recorded seed. POPE groups by
  original image and CelebA by identity; the validation report includes
  duplicate questions, original-image leakage, identity leakage, achieved
  group/record counts, and yes/no balance.
- Added deterministic Pillow darkness and Gaussian-blur transforms, explicit
  corruption specifications, split/group-preserving variant expansion, and a
  model-independent corrupted-pilot answerability decision contract.
- Added CLI commands for normalization, splitting, leakage validation, variant
  expansion, and answerability validation. Added `passive_docs/M2_PIPELINE.md` as the
  real-data validation handoff.
- Validation: `conda run -n mpvs-m1 python -m pytest -q` passed 31 synthetic/unit
  tests; Python compilation, CLI help, and Git whitespace checks passed. This is
  implemented and synthetic-unit-tested, not real-data validated. No POPE-COCO
  or CelebA source file was accessed and no severity was selected.

## 2026-10-28 — CUDA NF4 audit implementation

- Implemented a CUDA-only fresh-checkpoint candidate loader using the Transformers 4.52.4 bitsandbytes path. Candidate policy is NF4 storage, FP16 compute, uint8 quantized storage, no double quantization, explicit FP16 component-root exclusions, and an explicit FP16 `lm_head` exception.
- Expanded the module audit to record linear/source classes, stored dtype, quantization type/state, compute dtype, double-quantization state, quantized storage dtype, known exceptions, mismatched modules, and unclassified quantized modules. Requested loader settings do not count as validation; `matches_request` comes from the realized modules.
- Expanded memory accounting to include unique tensor storage reachable from bitsandbytes parameter `quant_state` metadata without double-counting parameter or buffer storage.
- Added isolated grid summarization: successful inference candidates are deduplicated by effective signature, sorted by measured total weight storage, and used to enumerate caps admitting at least two distinct configurations.
- Reworked `notebooks/m1_colab_cuda_nf4 kaushal.ipynb` to install the project, run FP16/uniform-NF4/language-NF4 as the feasibility gate, run all eight candidates in fresh subprocesses, preserve failure JSON, summarize M1 acceptance, and download a zip of the ignored artifacts.
- Validation on the Mac development environment: `conda run -n mpvs-m1 python -m pytest -q` passed 14 tests; notebook JSON, Python compilation, wheel construction, unsupported-candidate JSON, and Git whitespace checks passed. This is implemented and unit-tested, not GPU-tested. M1 acceptance remains unmet until the T4 outputs confirm at least two distinct feasible candidates at a measured cap.
- First revised-notebook attempt on a T4 checked out the still-published commit `8d9fee5` and stopped at `pip install -e .` because that revision predates the uncommitted `pyproject.toml` and M1 implementation. This was a publication/infrastructure failure and produced no NF4 evidence. The clone cell now fails early with a clear stale-branch diagnostic when required project files are absent.

## 2026-10-22 — Gauri CUDA smoke-test contribution (historical; superseded below)

- Created `notebooks/m1_colab_cuda_nf4 gauri.ipynb` on `Gauri-Commits`, branched from Omkar's `main`.
- The notebook checks for an L4, clones this branch, installs CUDA-side Python dependencies including bitsandbytes while retaining Colab's CUDA-enabled PyTorch, runs `scripts/preflight.py`, loads the pinned FP16 checkpoint, runs one synthetic inference, and records separate peak allocated/reserved GPU memory for loading and generation.
- The notebook does not construct an NF4 model or claim GPU validation. It must be run on a Colab L4 after the branch is published; no Colab/GPU run has been performed yet.
- Next: publish `Gauri-Commits`, run the notebook on an L4, and retain the generated preflight and smoke-test JSON artifacts outside tracked files.

## M0 prerequisite and M1 Mac slice (2026-10-16)

- Added `passive_docs/M1_MAC_FINDINGS.md` as the consolidated environment, implementation, findings, limitations, command, code-reference, and CUDA-handoff document.
- Created the `mpvs-m1` Conda environment with Python 3.11.17. Added tested direct dependency pins in `environment-macos.yml` and the complete installed inventory in `requirements-macos-lock.txt`; `pip check` reports no broken requirements.
- Resolved `HuggingFaceTB/SmolVLM2-2.2B-Instruct` and its processor to revision `482adb537c021c86670beed01cd58990d01e72e4`. Downloaded the checkpoint into the ignored Hugging Face cache.
- Implemented the model adapter, stable eight-candidate grid, verified component classification, per-module dtype/quantization inventory, effective signatures, unique-storage accounting, atomic JSON output, and explicit NF4 rejection until the CUDA implementation is audited.
- Ran `PYTHONPATH=src conda run -n mpvs-m1 python -m pytest -q`: 7 tests passed. Coverage includes candidate IDs, path boundaries, requested-versus-actual coverage, signatures, unsupported NF4 reporting, shared-storage handling, buffers, and unclassified parameters.
- Ran `conda run -n mpvs-m1 python scripts/preflight.py --output artifacts/preflight-macos-m1.json`: required Mac packages are present; CUDA, NVIDIA GPU, and bitsandbytes are absent.
- Loaded the full FP16 checkpoint on CPU and audited `model.vision_model`, `model.connector`, `model.text_model`, and `lm_head`. Measured unique FP16 parameter plus persistent-buffer storage is 4,493,569,888 bytes. This is implemented and CPU load-tested, not GPU-tested.
- Synthetic inference is not validated on this Mac. Transformers 5.20.1 failed image-token merging and was replaced with 4.52.4. A CPU attempt exposed an FP16 NNPack dtype/backend limitation; after input-dtype correction, the long CPU attempt produced no successful audit replacement. PyTorch reports `mps_built=True` and `mps_available=False`. Do not count this as inference success.
- NF4 construction, real quantization metadata measurement, effective-candidate deduplication, common-cap feasibility, and successful per-candidate inference remain unimplemented/unvalidated. M1 acceptance is not met.

Commands also run: `conda create -n mpvs-m1 python=3.11 pip -y`; dependency installs captured in `environment-macos.yml`; Hugging Face model-info/config queries; `conda run -n mpvs-m1 python -m pip check`; candidate CLI and structured failure-path checks. No dataset, training, calibration, main experiment, or test-set evaluation was run.

## 2026-10-27 — Gauri midterm M0–M2 progress document

- Added `passive_docs/MIDTERM_M0_M2.md`, scoped to M0–M2 as agreed by the team: environment setup (Mac dev environment from `Kaushal-Commits`, Colab L4 environment from this branch), a proposed end-to-end flowchart and M1 audit-gate flowchart, the questions M1 validation will answer and how each outcome changes the plan, the proposed M2 data/corruption pipeline with the leakage and determinism checks to be written, and a risk table covering compute, Colab limits, library compatibility, data access, and schedule.
- Every figure is labelled measured, implemented-not-run, estimated, or proposed. The per-config memory table is Omkar's estimate from `docs/project-deep-dive`; only the FP16 total (4,493,569,888 bytes) is measured (Kaushal, Mac CPU load).
- Documentation only. No code, model download, GPU run, dataset access, or experiment was performed. The Colab notebook has still not been run.
- Next: run `notebooks/m1_colab_cuda_nf4 gauri.ipynb` on an L4 and record its measurements here; then write the M2 synthetic-fixture leakage tests.

## 2026-10-27 — Gauri first GPU run: FP16 smoke test on Colab T4

- Changed the notebook to accept a free-tier T4 as well as an L4 (`aafd02c`) and tag artifact names by GPU.
- First T4 attempt failed at `AutoProcessor.from_pretrained`: `ImportError: Package num2words is required to run SmolVLM processor`. Added `num2words` to the install list and version record (`8b55db6`).
- Second attempt, run from commit `8b55db6` via Colab Runtime → Run all on a Tesla T4: succeeded. Environment: Python 3.13, PyTorch 2.11.0+cu130 (CUDA 13.0), Transformers 4.52.4, Accelerate 1.15.0, bitsandbytes 0.50.2, Pillow 11.3.0, sentencepiece 0.2.2, num2words 0.5.14.
- Measured: FP16 parameters confirmed; load 143.1 s including download; peak allocated/reserved after load 4,504,592,896 / 4,735,369,216 B; one synthetic 384×384 inference (greedy, 4 new tokens) 3.92 s first call; peak allocated/reserved during inference 4,932,262,912 / 5,322,571,776 B; answer `Yes.`.
- Status: FP16 load and synthetic inference are **GPU-tested** on a T4. NF4 construction, quantization audit, real-image memory, per-answer runtime, and accuracy are not tested. Artifact JSON kept outside tracked files (Colab session storage).
- Results added to `passive_docs/MIDTERM_M0_M2.md` §2.1.
- Next: implement selective NF4 replacement and run the component audit on the same Colab GPU; write the M2 synthetic-fixture leakage tests.

## Handoff rule

Every coding session should append what changed, exact validation commands and outcomes, unresolved issues, and the next task. Do not mark GPU-tested or experiment-completed based on synthetic/unit tests.

## Publication preparation (2026-10-05)

- Omkar requested publication to a new repository and supplied `lashkareomkar@gmail.com` for future commits. Updated `passive_docs/GIT_WORKFLOW.md` accordingly and set the repository-local Git identity to `Omkar Lashkare <lashkareomkar@gmail.com>`. The initial commit retains its existing authorship.
- Ran `git status --short --branch` (clean `main` before this documentation change), `git remote -v` (no remote), `git ls-files` (only the research scaffold), and `du -sh . .git` (236K/140K). No raw data, checkpoints, or large artifacts are tracked.
- Ran `gh auth status`: the saved `Omkar38` token initially appeared invalid. Completed `gh auth login -h github.com -p https -w`, then verified the authenticated account with `gh api user --jq .login` (`Omkar38`).
- Omkar created the GitHub repository as public. Added and verified `README.md` through the connected GitHub account, fetched that commit with `git fetch origin main`, merged the two histories with `git merge --allow-unrelated-histories --no-commit origin/main`, then ran `git commit -m 'Merge GitHub README with local project history'` and `git push -u origin main` successfully.
- The repository now contains the existing research scaffold and README. No scientific implementation or experiment was run in this session. Next: run M0 on the experiment machine.

## Scaffold validation (2026-10-05)

- Ran `python3 scripts/preflight.py --output artifacts/preflight.json` successfully in the preparation environment. It reports no available NVIDIA GPU and no installed PyTorch/Transformers/bitsandbytes packages. This environment is not the proposed experiment machine.
- Checked Python syntax, all local Markdown links, the eight unique requested precision tuples, the draft-run gate, and the shared Claude instruction import; all passed.
- Confirmed the repository snapshot omits student IDs.
- No model downloads, GPU inference, or scientific experiments were run.
