# Implementation plan

Start with M0. Do not jump from documentation to the full experiment.

## M0 — Environment and data-access inventory

- Run `python3 scripts/preflight.py --output artifacts/preflight.json`.
- Verify the real GPU, driver/CUDA support, disk space, and access to official model/data sources.
- Create an isolated Python environment, verify compatible PyTorch/Transformers/bitsandbytes versions, then record exact working versions.
- Inspect downloaded dataset schemas/counts; do not download large files without confirming storage and applicable access.
- Fill hardware and source fields in a working protocol copy.

**Acceptance:** Reproducible environment instructions, resolved model revision, available data paths or explicit access blockers. No claims of GPU compatibility from merely finding installed packages.

## M1 — Model adapter and component-control audit

Suggested modules: `src/mpvs/model.py`, `src/mpvs/quantization.py`, `src/mpvs/memory.py`.

- Load the FP16 reference and inventory architecture/module paths.
- Implement requested precision assignments using supported APIs or a carefully audited replacement method.
- Audit each requested combination, actual tensor storage, quantization metadata, exceptions, and effective duplicates.
- Run a very small legal/synthetic image-question smoke test, then the pilot as data becomes available.
- Isolate candidate processes if necessary to prevent stale allocations.

**Acceptance:** Machine-readable audit and feasible candidate list; at least two distinct candidates at a measured cap for a selection comparison. Stop and report a scientific blocker if independent component control is not achievable. Do not relabel global quantization as mixed precision.

## M2 — Data and corruption pipeline

Suggested modules: `src/mpvs/data/pope.py`, `src/mpvs/data/celeba.py`, `src/mpvs/splits.py`, `src/mpvs/corruptions.py`.

- Build normalized records with group IDs and labels.
- Create immutable pilot/development/test manifests.
- Add deterministic transformations and model-independent answerability checks.
- Prespecify the POPE variant and visible CelebA attribute set.
- Add meaningful leakage and determinism tests with synthetic fixtures.

**Acceptance:** No original image or CelebA identity crosses splits; all questions and transformed variants inherit the original group; reproducible manifests and transforms.

## M3 — Pilot, prediction runner, and protocol freeze

Suggested modules: `src/mpvs/inference.py`, `src/mpvs/cache.py`, `src/mpvs/cli.py`.

- Build fixed prompting, decoding, strict answer parsing, atomic caching, and resume support.
- Run the disjoint 100-question pilot and measure runtime/memory.
- Resolve severity, split-size, mixture, cap, and generation settings.
- Estimate full compute demand; choose the documented fallback if required.
- Pin and hash the frozen protocol.

**Acceptance:** Raw and parsed outputs can be reproduced; resume skips completed keys without collisions; invalid answers and runtime failures remain distinguishable; the planned run fits the actual resource budget.

## M4 — Selection and statistical evaluation

Suggested modules: `src/mpvs/selection.py`, `src/mpvs/metrics.py`, `src/mpvs/bootstrap.py`.

- Select clean/target/mixed winners using development data only.
- Enforce eligibility and deterministic tie-breaking.
- Implement paired cluster bootstrap and separate development selection stability.
- Check with hand-computed fixtures, including no feasible cap, equal scores, repeated clusters, invalid answers, identical winners, and negative deltas.

**Acceptance:** Exact fixture expectations pass; selector rejects test data; pairing/grouping remains intact; results include honest denominators and uncertainty.

## M5 — Main experiments and report

- Run all supported configurations and frozen conditions with resumable logging.
- Produce accuracy-memory plots, delta tables, invalid-answer summaries, and identifiable component comparisons.
- Record failures, effective sample sizes, environment, protocol hash, and limitations.
- Draft the midterm/final report from actual outputs.
- Add optional extensions only after the core is reproducible.

**Acceptance:** Every reported number is traceable to cached predictions, manifests, audited configurations, and analysis code. A null result still completes the research objective.

## Working arrangement

Suggested implementation ownership follows the proposal: Kaushal leads M1/M2, Gauri leads validation and M4, Omkar leads protocol decisions, analysis/reporting, and integration. This is a proposed work breakdown, not a claim that tasks have started.

If both Claude Code and Codex are used simultaneously, use distinct Git worktrees and branches and agree on file ownership. Prefer sequential handoffs until the baseline architecture is established.
