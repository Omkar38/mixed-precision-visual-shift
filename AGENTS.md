# Shared instructions for coding agents

## Read before working

Read `PROJECT_CONTEXT.md`, `docs/STATUS.md`, and `docs/IMPLEMENTATION_PLAN.md`. Before modifying experimental logic, read `docs/EXPERIMENT_PROTOCOL.md` and `docs/DECISIONS.md`. `CLAUDE.md` imports this file so Claude Code and Codex share the same rules.

This is a research implementation, currently at M0. Do not mistake the proposal or example configuration for working software or completed experiments.

## Scientific requirements

- Preserve the core question: transfer of development-selected precision settings under a common measured weight-memory cap.
- Use one primary model, POPE-COCO and CelebA, darkness and Gaussian blur. Optional models/tasks are extensions, not prerequisites.
- Do not train, fine-tune, calibrate weights using development images, add LoRA, or substitute a different quantizer silently.
- Audit actual modules and quantization state. A requested 4-bit configuration is not valid merely because a loader accepted a flag.
- Reconstruct each candidate from the same original checkpoint. Do not quantize a previously quantized model or use dequantization as the FP16 reference.
- Split by original image before corruption; split CelebA by identity as well. Keep all questions and variants in the same split.
- Keep pilot, development, and test groups disjoint. Never select configurations, corruption severities, memory caps, or parser rules using test accuracy.
- Invalid answers count as incorrect. Infrastructure failures are failures to rerun or report, not model answers.
- Use a common candidate set and development sampling budget across clean/target/mixed selection at each cap.
- Preserve paired evaluation and cluster-aware uncertainty. Accept small, negative, or inconclusive results.
- Never fabricate results, citations, checks, performance improvements, professor feedback, or completed tasks.

## Implementation approach

- Complete milestones in order. Build and validate one small vertical slice before bulk inference.
- Choose reasonable implementation details, label them as proposed until pilot-checked, and record material decisions.
- Use Python; start with a simple CLI and modules. Avoid a web UI, databases, orchestration frameworks, or large abstractions unless a concrete need emerges.
- Use the installed model/library version's real API; consult official documentation and inspect module names instead of guessing.
- Pin the working environment after compatibility checks. Record checkpoint and processor revisions, GPU model, software versions, seeds, and config hashes.
- Prefer batch size 1 for the first GPU pilot; do not assume an 8GB device can run every configuration.
- Write meaningful tests for split leakage, pairing, budget eligibility, tie-breaking, invalid-answer handling, cache keys, and bootstrap behavior.
- Tests may use clearly labeled synthetic data. Synthetic success is not research evidence.
- Do not launch the full experiment before the pilot acceptance criteria and frozen protocol are recorded.
- Add resumable prediction caching before scaling. Cache keys must include model/processor revisions, actual quantization configuration, image transformation, prompt, and decoding settings.
- Keep secrets, raw datasets, images, checkpoint files, and large generated outputs outside tracked files.

## Collaboration and commits

For changes made on Omkar's behalf, follow `docs/GIT_WORKFLOW.md`: use his name and verified-in-proposal email for both author and committer; do not use Codex/Claude as the author or append an AI co-author trailer. Do not rewrite existing authors or misattribute another person's work.

Do not let two agents edit the same working tree concurrently. Use separate branches/worktrees with explicit file ownership for parallel work. Read the latest status and Git diff before starting; preserve other people's changes.

Commit coherent, reviewable changes after relevant validation when asked to commit. Never force-push or rewrite shared history without explicit authorization.

## End-of-session handoff

Update `docs/STATUS.md` with changes, exact commands run, outcomes, limitations, and the next actionable task. Record protocol changes in `docs/DECISIONS.md`. Distinguish “implemented,” “unit-tested,” “GPU-tested,” and “experiment completed.” State blockers plainly.
