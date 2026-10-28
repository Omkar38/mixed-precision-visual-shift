# Current status

Updated: 2026-10-28.

## Completed

- Read the current proposal's Final tab.
- Created the research context, shared agent instructions, implementation plan, and draft protocol.
- Added a dependency-free environment inventory script.
- Created a local Git repository using Omkar's author and committer identity.
- Published the repository at `https://github.com/Omkar38/mixed-precision-visual-shift`.
- Completed the M1 CUDA/NF4 audit for all eight distinct component allocations
  on a Tesla T4; the machine-readable summary reports acceptance.

## Not completed

- Real-data POPE-COCO/CelebA schema and access validation.
- Real-data split generation, balance review, and leakage validation.
- Pilot selection of corruption severities, split sizes, and final memory caps.
- Inference/evaluation implementation.
- Experimental results or report figures.

## Current milestone

M2 — data and corruption pipeline (implementation complete on synthetic fixtures;
real-data validation pending).

## Next task

Gauri should validate the M2 loaders against authorized POPE-COCO and CelebA
files, record source schemas/checksums/counts, propose explicit dataset choices,
and run the group-leakage, determinism, balance, and pilot-answerability checks.
Do not freeze severities or split sizes from synthetic fixtures.

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
  expansion, and answerability validation. Added `docs/M2_PIPELINE.md` as the
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

- Added `docs/M1_MAC_FINDINGS.md` as the consolidated environment, implementation, findings, limitations, command, code-reference, and CUDA-handoff document.
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

- Added `docs/MIDTERM_M0_M2.md`, scoped to M0–M2 as agreed by the team: environment setup (Mac dev environment from `Kaushal-Commits`, Colab L4 environment from this branch), a proposed end-to-end flowchart and M1 audit-gate flowchart, the questions M1 validation will answer and how each outcome changes the plan, the proposed M2 data/corruption pipeline with the leakage and determinism checks to be written, and a risk table covering compute, Colab limits, library compatibility, data access, and schedule.
- Every figure is labelled measured, implemented-not-run, estimated, or proposed. The per-config memory table is Omkar's estimate from `docs/project-deep-dive`; only the FP16 total (4,493,569,888 bytes) is measured (Kaushal, Mac CPU load).
- Documentation only. No code, model download, GPU run, dataset access, or experiment was performed. The Colab notebook has still not been run.
- Next: run `notebooks/m1_colab_cuda_nf4 gauri.ipynb` on an L4 and record its measurements here; then write the M2 synthetic-fixture leakage tests.

## 2026-10-27 — Gauri first GPU run: FP16 smoke test on Colab T4

- Changed the notebook to accept a free-tier T4 as well as an L4 (`aafd02c`) and tag artifact names by GPU.
- First T4 attempt failed at `AutoProcessor.from_pretrained`: `ImportError: Package num2words is required to run SmolVLM processor`. Added `num2words` to the install list and version record (`8b55db6`).
- Second attempt, run from commit `8b55db6` via Colab Runtime → Run all on a Tesla T4: succeeded. Environment: Python 3.13, PyTorch 2.11.0+cu130 (CUDA 13.0), Transformers 4.52.4, Accelerate 1.15.0, bitsandbytes 0.50.2, Pillow 11.3.0, sentencepiece 0.2.2, num2words 0.5.14.
- Measured: FP16 parameters confirmed; load 143.1 s including download; peak allocated/reserved after load 4,504,592,896 / 4,735,369,216 B; one synthetic 384×384 inference (greedy, 4 new tokens) 3.92 s first call; peak allocated/reserved during inference 4,932,262,912 / 5,322,571,776 B; answer `Yes.`.
- Status: FP16 load and synthetic inference are **GPU-tested** on a T4. NF4 construction, quantization audit, real-image memory, per-answer runtime, and accuracy are not tested. Artifact JSON kept outside tracked files (Colab session storage).
- Results added to `docs/MIDTERM_M0_M2.md` §2.1.
- Next: implement selective NF4 replacement and run the component audit on the same Colab GPU; write the M2 synthetic-fixture leakage tests.

## Handoff rule

Every coding session should append what changed, exact validation commands and outcomes, unresolved issues, and the next task. Do not mark GPU-tested or experiment-completed based on synthetic/unit tests.

## Publication preparation (2026-10-05)

- Omkar requested publication to a new repository and supplied `lashkareomkar@gmail.com` for future commits. Updated `docs/GIT_WORKFLOW.md` accordingly and set the repository-local Git identity to `Omkar Lashkare <lashkareomkar@gmail.com>`. The initial commit retains its existing authorship.
- Ran `git status --short --branch` (clean `main` before this documentation change), `git remote -v` (no remote), `git ls-files` (only the research scaffold), and `du -sh . .git` (236K/140K). No raw data, checkpoints, or large artifacts are tracked.
- Ran `gh auth status`: the saved `Omkar38` token initially appeared invalid. Completed `gh auth login -h github.com -p https -w`, then verified the authenticated account with `gh api user --jq .login` (`Omkar38`).
- Omkar created the GitHub repository as public. Added and verified `README.md` through the connected GitHub account, fetched that commit with `git fetch origin main`, merged the two histories with `git merge --allow-unrelated-histories --no-commit origin/main`, then ran `git commit -m 'Merge GitHub README with local project history'` and `git push -u origin main` successfully.
- The repository now contains the existing research scaffold and README. No scientific implementation or experiment was run in this session. Next: run M0 on the experiment machine.

## Scaffold validation (2026-10-05)

- Ran `python3 scripts/preflight.py --output artifacts/preflight.json` successfully in the preparation environment. It reports no available NVIDIA GPU and no installed PyTorch/Transformers/bitsandbytes packages. This environment is not the proposed experiment machine.
- Checked Python syntax, all local Markdown links, the eight unique requested precision tuples, the draft-run gate, and the shared Claude instruction import; all passed.
- Confirmed the repository snapshot omits student IDs.
- No model downloads, GPU inference, or scientific experiments were run.
