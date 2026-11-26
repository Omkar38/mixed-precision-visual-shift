# Project plan and experiment protocol

This is the active implementation and scientific-control document. Read
`live_STATUS.md` first to find the current milestone, then use the relevant
milestone and protocol sections below. Accepted or revised choices belong in
`DECISIONS.md`.

Do not jump from documentation to the full experiment. Build and validate one
small vertical slice at a time, and do not launch the main run until the
required protocol fields are frozen.

## Part I — Milestones

### M0 — Environment and data-access inventory

- Run `python3 scripts/preflight.py --output artifacts/preflight.json`.
- Verify the real GPU, driver/CUDA support, disk space, and access to official
  model/data sources.
- Create an isolated Python environment, verify compatible
  PyTorch/Transformers/bitsandbytes versions, then record exact working
  versions.
- Inspect downloaded dataset schemas/counts; do not download large files
  without confirming storage and applicable access.
- Fill hardware and source fields in a working protocol copy.

**Acceptance:** Reproducible environment instructions, resolved model
revision, available data paths or explicit access blockers. No claims of GPU
compatibility from merely finding installed packages.

### M1 — Model adapter and component-control audit

Suggested modules: `src/mpvs/model.py`, `src/mpvs/quantization.py`,
`src/mpvs/memory.py`.

- Load the FP16 reference and inventory architecture/module paths.
- Implement requested precision assignments using supported APIs or a
  carefully audited replacement method.
- Audit each requested combination, actual tensor storage, quantization
  metadata, exceptions, and effective duplicates.
- Run a very small legal/synthetic image-question smoke test, then the pilot as
  data becomes available.
- Isolate candidate processes if necessary to prevent stale allocations.

**Acceptance:** Machine-readable audit and feasible candidate list; at least
two distinct candidates at a measured cap for a selection comparison. Stop and
report a scientific blocker if independent component control is not achievable.
Do not relabel global quantization as mixed precision.

### M2 — Data and corruption pipeline

Suggested modules: `src/mpvs/data/pope.py`, `src/mpvs/data/celeba.py`,
`src/mpvs/splits.py`, `src/mpvs/corruptions.py`.

- Build normalized records with group IDs and labels.
- Create immutable pilot/development/test manifests.
- Add deterministic transformations and model-independent answerability checks.
- Prespecify the POPE variant and visible CelebA attribute set.
- Add meaningful leakage and determinism tests with synthetic fixtures.

**Acceptance:** No original image or CelebA identity crosses splits; all
questions and transformed variants inherit the original group; manifests and
transforms are reproducible.

The detailed command/reference runbook is passive documentation at
`../passive_docs/M2_PIPELINE.md`. Use it when reproducing M2, but use
`live_STATUS.md` for the current task.

### M3 — Pilot, prediction runner, and protocol freeze

Suggested modules: `src/mpvs/inference.py`, `src/mpvs/cache.py`,
`src/mpvs/cli.py`.

- Build fixed prompting, decoding, strict answer parsing, atomic caching, and
  resume support.
- Run the disjoint 100-question pilot and measure runtime/memory.
- Resolve severity, split-size, mixture, cap, and generation settings.
- Estimate full compute demand; choose the documented fallback if required.
- Pin and hash the frozen protocol.

**Acceptance:** Raw and parsed outputs can be reproduced; resume skips
completed keys without collisions; invalid answers and runtime failures remain
distinguishable; the planned run fits the actual resource budget.

### M4 — Selection and statistical evaluation

Suggested modules: `src/mpvs/selection.py`, `src/mpvs/metrics.py`,
`src/mpvs/bootstrap.py`.

- Select clean/target/mixed winners using development data only.
- Enforce eligibility and deterministic tie-breaking.
- Implement paired cluster bootstrap and separate development selection
  stability.
- Check with hand-computed fixtures, including no feasible cap, equal scores,
  repeated clusters, invalid answers, identical winners, and negative deltas.

**Acceptance:** Exact fixture expectations pass; selector rejects test data;
pairing/grouping remains intact; results include honest denominators and
uncertainty.

### M5 — Main experiments and report

- Run all supported configurations and frozen conditions with resumable
  logging.
- Produce accuracy-memory plots, delta tables, invalid-answer summaries, and
  identifiable component comparisons.
- Record failures, effective sample sizes, environment, protocol hash, and
  limitations.
- Draft the midterm/final report from actual outputs.
- Add optional extensions only after the core is reproducible.

**Acceptance:** Every reported number is traceable to cached predictions,
manifests, audited configurations, and analysis code. A null result still
completes the research objective.

## Part II — Experiment protocol and data contracts

The choices below marked proposed must be checked in the pilot and frozen
before the main run. Proposal commitments are summarized in
`../PROJECT_CONTEXT.md`; material decisions and deviations belong in
`DECISIONS.md`.

### Configuration audit

Enumerate `(vision, connector, language)` precision tuples in a stable order,
with each precision FP16 or NF4. For every candidate, start from the same
original checkpoint and record:

- requested allocation and actual module paths/classes;
- stored weight dtypes, quantized module types, NF4 state, compute dtype, and
  any double-quantization setting;
- unsupported/excepted layers and tied/shared tensors;
- unique weight storage plus quantization metadata and persistent buffers;
- a stable effective-configuration signature; and
- load success, inference success, and failure reason.

Do not count an API flag as verification. Validate selected module coverage and
a small inference run. Keep layer exceptions and compute policy constant where
possible. If all eight configurations are unavailable, use a disclosed
supported subset; do not present a complete factorial analysis.

Measure storage without double-counting shared storage. Record peak allocated
and reserved GPU memory separately after resetting counters and synchronizing
timings. CPU offload changes the execution setup; disable it for the core
single-GPU comparison unless the protocol is explicitly revised.

### Splits, transformations, and mixtures

- Split original image/identity groups before generating variants. Store
  checksums and seeds.
- Pilot groups do not enter development or test. Whole-group assignment takes
  precedence over exact target question counts.
- Choose the POPE variant and visible CelebA attribute set during the pilot.
- Implement darkness as a documented RGB intensity factor below one and
  Gaussian blur with a documented radius/kernel convention. Do not treat
  parameters from different libraries as interchangeable.
- Freeze up to three severity values per corruption and dataset after
  model-independent pilot answerability checks. Fewer levels are allowed when
  stronger candidates make a meaningful share of questions unanswerable. Do
  not assume one blur radius has the same effect at different source-image
  resolutions or describe the transforms as exact ImageNet-C reproductions
  unless that implementation is actually used.
- Freeze transformation order relative to resizing, encoding, and processor
  behavior; use it for every candidate.
- Proposed mixtures reuse one development image-group roster and assign every
  group one condition. For target fraction `alpha` in `{0, .25, .5, .75, 1}`,
  record requested and achieved question/image fractions and use deterministic
  assignments shared across candidates.
- Keep total questions and distinct images constant. Do not count clear and
  corrupted duplicates as extra independent examples.
- Freeze label/attribute proportions or record unavoidable group-allocation
  differences. Test cohorts remain identical for paired comparisons.

### Prediction record

Proposed JSONL fields, one row per question × condition × effective
configuration:

`run_id, dataset, split, original_image_id, identity_id, question_id, question, label, corruption, severity_id, config_id, model_revision, processor_revision, protocol_hash, prompt_hash, raw_answer, parsed_answer, correct, generation_seconds, status`.

- `label` is `yes` or `no`; `parsed_answer` is `yes`, `no`, or `invalid`.
- `identity_id` is required for CelebA and null for POPE.
- Runtime errors have non-success status and no scored correctness. Retry
  consistently or report incompleteness; do not silently count OOM as an
  incorrect answer.
- Keep raw answers so parser changes remain auditable. A parser change creates
  a new analysis version.
- Proposed strict parser: normalize case and surrounding whitespace; accept
  only a single yes/no token with optional final sentence punctuation. Anything
  else is invalid. Pilot-check it against the fixed prompt before freezing.
- Cache keys also cover source-image checksum, transform parameters, package
  versions affecting preprocessing, generation settings, and quantization
  audit signature.
- Write atomically and reject duplicate keys or incompatible cache reuse.

### Selection record

One row per dataset × target corruption/severity × cap × selection policy:

`candidate_set_hash, weight_cap_bytes, selection_policy, target_fraction_requested, target_fraction_achieved, development_manifest_hash, selected_config_id, selected_weight_bytes, development_accuracy, tie_break_applied`.

The selector must not accept test rows. Use a fixed rank for the final
tie-break. All policies at a cap share the same eligible candidates.
Unsupported/OOM exclusions must follow a prespecified feasibility procedure,
not poor test accuracy.

### Analysis

- Score yes/no accuracy over every successfully generated evaluation answer,
  counting invalid text as incorrect.
- For positive-class yes metrics, invalid predictions on positive labels are
  false negatives. Invalid predictions on negative labels are neither true
  negatives nor false positives; report invalid counts separately and define
  zero-denominator handling explicitly.
- Report the full yes/no/invalid contingency table and accuracy denominator.
- Primary delta: `100 × (target-selected target accuracy − clean-selected
  target accuracy)`.
- Mixed improvement: `100 × (mixed-selected target accuracy − clean-selected
  target accuracy)`.
- Quantization's extra corruption drop for configuration `c`:
  `100 × [(A_c(clean) − A_c(target)) − (A_FP16(clean) − A_FP16(target))]`,
  using identical paired test groups.
- Test bootstrap: draw image clusters for POPE or identity clusters for CelebA
  with replacement and include all associated records. Compute paired
  differences with fixed selected configurations. The proposed setting is
  2,000 percentile-bootstrap draws at 95% confidence; freeze its seed/count.
- Development bootstrap: resample development groups, preserve mixture
  assignment, rerun selection, and report selection frequencies separately.
- For component effects, hold other components fixed. Do not interpret
  memory-cap comparisons between arbitrary candidates as causal component
  effects. Incomplete grids require narrower claims.
- Treat per-cell intervals as descriptive unless a multiple-comparison strategy
  was prespecified. Report zero/negative deltas and identical winners.

### Before the full run

Required records are the environment lock, checkpoint revisions, successful
component audit, immutable split checks, frozen transform/prompt/parser rules,
measured eligible caps, resource estimate from the 100-question pilot, and a
working cache-resume smoke test.

Freeze a machine-readable configuration and hash only after every relevant
field is resolved. Preserve earlier pilot artifacts and disclose deviations
instead of overwriting them.

## Working arrangement

Suggested ownership follows the proposal: Kaushal leads M1/M2, Gauri leads
validation and M4, and Omkar leads protocol decisions, analysis, writing, and
integration. This is a proposed work breakdown, not evidence that a task has
started.

When multiple coding agents work simultaneously, use distinct Git worktrees
and branches with explicit file ownership. Prefer sequential handoffs until a
baseline architecture is stable.
