# Experiment protocol and data contracts

Status: draft implementation specification. Proposal commitments are summarized in PROJECT_CONTEXT.md; choices below marked “proposed” must be checked in the pilot and frozen before the main run.

## Configuration audit

Enumerate (vision, connector, language) precision tuples in a stable order, each precision FP16 or NF4. For each candidate, start from the same original checkpoint. Record:
- Requested allocation and actual module paths/classes.
- Stored weight dtypes, quantized module types, NF4 quantization state, compute dtype, and any double-quantization setting.
- Unsupported/excepted layers and tied/shared tensors.
- Total unique weight storage plus quantization metadata and persistent model buffers under a documented counting convention.
- A stable effective-configuration signature so equivalent candidates are detected.
- Load success, inference success, and failure reason.

Do not count an API flag as verification. Validate selected module coverage and a small inference run. Keep layer exceptions and compute policy constant wherever possible. If all eight configurations are unavailable, use a disclosed supported subset; do not present a complete factorial analysis.

Measure storage without double-counting shared storage. Record peak allocated/reserved GPU memory separately, after resetting peak counters and synchronizing relevant timings. CPU offload changes the execution setup: disable it for the core single-GPU comparison unless the protocol is explicitly revised.

## Splits, transformations, and mixtures

- Split original image/identity groups before generating variants. Store checksums and seeds.
- Pilot groups do not enter development or test. Whole-group assignment takes precedence over exact target question counts.
- Choose the POPE variant and visible CelebA attribute set during the pilot.
- Proposed corruption implementations: multiply RGB intensity by a factor below one for darkness; use a documented Gaussian blur radius/kernel convention. Do not treat parameters from different libraries as interchangeable.
- Freeze three severity values per corruption after model-independent answerability checks on pilot examples. Do not describe these as exact ImageNet-C reproductions unless that implementation is actually used.
- Freeze transformation order relative to resizing, image encoding, and processor behavior; use it for every candidate.
- Proposed mixture construction: reuse one development image-group roster and assign each group one condition. At alpha in {0, .25, .5, .75, 1}, alpha is the intended fraction of target-condition questions. Handle indivisible groups with a prespecified rounding/allocation rule; record achieved question and image fractions. For alpha=0/1 all groups are clean/target. Use deterministic assignments shared across candidates.
- Keep total questions and distinct image counts constant. Never count both clear and corrupted duplicates as extra independent examples in a mixture.
- Freeze label/attribute proportions or record unavoidable differences due to group allocation. Test cohorts are identical for paired comparisons.

## Prediction record (proposed JSONL schema)

One row per question × condition × effective configuration, with:
`run_id, dataset, split, original_image_id, identity_id, question_id, question, label, corruption, severity_id, config_id, model_revision, processor_revision, protocol_hash, prompt_hash, raw_answer, parsed_answer, correct, generation_seconds, status`.

- Label is `yes` or `no`; parsed answer is `yes`, `no`, or `invalid`.
- Identity ID is required for CelebA; null for POPE.
- Runtime errors have non-success status and no scored correctness. Retry consistently or report incompleteness; do not silently count OOM as an incorrect answer.
- Keep raw answers so parser changes are auditable. A parser change creates a new analysis version.
- Proposed strict parser: normalize case and surrounding whitespace; accept only a single yes/no token with optional final sentence punctuation. Anything else is invalid. Check this against the fixed prompt during the pilot, then freeze it.
- Cache keys also cover source image checksum, transform parameters, package versions affecting preprocessing, generation configuration, and quantization audit signature.
- Write atomically and reject duplicate keys or incompatible cache reuse.

## Selection record

One row per dataset × target corruption/severity × cap × selection policy:
`candidate_set_hash, weight_cap_bytes, selection_policy, target_fraction_requested, target_fraction_achieved, development_manifest_hash, selected_config_id, selected_weight_bytes, development_accuracy, tie_break_applied`.

The selector must not accept test rows. Use a fixed rank for the final tie-break. All policies at a cap share exactly the same eligible candidates. Unsupported/OOM exclusions must be based on a prespecified feasibility procedure, not poor test accuracy.

## Analysis

- Score yes/no accuracy over every successfully generated evaluation answer, counting invalid text as incorrect.
- For positive-class yes metrics, invalid predictions on positive labels contribute false negatives. Invalid predictions on negative labels are neither true negatives nor false positives; report the invalid count separately. Define zero-denominator handling explicitly, preferably null with counts.
- Report the full yes/no/invalid contingency table and accuracy denominator so F1 cannot hide invalid outputs.
- Primary delta: 100 × (target-selected target accuracy − clean-selected target accuracy).
- Mixed improvement: 100 × (mixed-selected target accuracy − clean-selected target accuracy).
- Quantization's extra corruption drop for configuration c: 100 × [(A_c(clean) − A_c(target)) − (A_FP16(clean) − A_FP16(target))]. Use identical paired test groups for these terms.
- Test bootstrap: draw image clusters (POPE) or identity clusters (CelebA) with replacement and include all their records, including repeated draws. Compute paired differences with fixed selected configurations. Proposed 2,000 draws and percentile 95% intervals; freeze count and RNG seed.
- Development bootstrap: resample development groups, preserve the chosen mixture-assignment policy, rerun selection, and report selection frequencies separately from conditional test uncertainty.
- For component effects, hold the other components fixed. Do not interpret a memory-cap-constrained comparison between arbitrary candidates as a causal component effect. Incomplete grids require narrower claims.
- Treat per-cell confidence intervals as descriptive unless a multiple-comparison strategy was prespecified. Do not turn the many-cell search into a significance claim.
- Report zero/negative deltas and identical selected configurations.

## Before the full run

Required records: environment lock, checkpoint revisions, successful component audit, immutable split checks, frozen transform/prompt/parser rules, measured eligible caps, resource estimate from the 100-question pilot, and a working cache-resume smoke test.

A protocol can be frozen only after unresolved fields relevant to the run are filled. Freeze a machine-readable configuration and hash it. Preserve earlier pilot artifacts and disclose any deviations instead of overwriting them.
