# Current status

Updated: 2026-10-16.

## Completed

- Read the current proposal's Final tab.
- Created the research context, shared agent instructions, implementation plan, and draft protocol.
- Added a dependency-free environment inventory script.
- Created a local Git repository using Omkar's author and committer identity.
- Published the repository at `https://github.com/Omkar38/mixed-precision-visual-shift`.

## Not completed

- Tested/pinned ML environment or verified 8GB GPU availability.
- Dataset acquisition and split preparation.
- Model loading, component quantization audit, or pilot.
- Inference/evaluation implementation.
- Experimental results or report figures.

## Current milestone

M1 — model adapter and component-control audit (in progress).

## Next task

Move the M1 code to the NVIDIA experiment host, validate a CUDA environment with bitsandbytes, then implement and audit fresh-checkpoint NF4 replacement for each requested component. The highest-priority scientific risk remains actual independent quantization control of the vision encoder, connector, and language backbone.

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
