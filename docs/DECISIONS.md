# Decision log

## Established by the current proposal

- Primary model: SmolVLM2-2.2B.
- Core datasets: one POPE-COCO variant and a visible-attribute CelebA subset.
- Corruptions: darkness and Gaussian blur, up to three severities each.
- Candidate precision choices: FP16/NF4 by component, at most eight configurations.
- No training or development-image weight calibration.
- Clean/target/mixed development selection under common measured weight-memory caps.
- Group-disjoint splits and paired cluster-aware evaluation.
- One 100-question feasibility pilot before scale-up.

## 2026-10-05 — Handoff construction

Created a documentation-first repository based on the current Final tab. No experiment has run. Student IDs, member names, and email addresses are omitted from the repository; responsibilities are recorded by project role.

The implementation protocol adds proposed choices (strict parsing, image-group mixture assignment, bootstrap count) to make engineering concrete. These are explicitly not yet pilot-validated or frozen.

## Open decisions

| Decision | Evidence needed |
| --- | --- |
| Hardware and software versions | Actual environment and compatibility smoke test |
| Exact model/processor revisions | Resolved immutable checkpoint identifiers |
| Component mappings and eight-way feasibility | Per-module audit and successful inference |
| Weight storage convention and candidate caps | Measured storage including metadata/shared tensors |
| POPE variant and CelebA attribute subset | Official annotations and pilot checks |
| Pilot/dev/test sizes | Group counts, label balance, and runtime |
| Darkness/blur severity parameters | Model-independent pilot answerability checks |
| Prompt, resolution, generation, parser | Pilot output inspection without test use |
| Mixture allocation and rounding | Constant sample/image count checks |
| Bootstrap settings and summary weights | Declared analysis plan before test scoring |
| Need for a fallback model | Measured failure/runtime evidence |
| NF4 double quantization on or off | Freeze before the pilot; it shifts every measured cap |
| Blur severity defined in absolute pixels or relative to image size | Cross-domain comparability check between COCO and CelebA |
| Which measured caps are primary | Pilot confirmation without comparative test accuracy |
| Steady-state inference throughput | Pilot measurement; the audit figure is a cold first call |

## 2026-10-16 — Mac M1 development environment

Use a Python 3.11 Conda environment named `mpvs-m1` for the Apple Silicon
development slice. Compatibility testing selected PyTorch 2.14.1,
Transformers 4.52.4, and Accelerate 1.15.0; the initially resolved Transformers
5.20.1 failed SmolVLM image-token merging and was rejected. Direct dependencies
are pinned in `environment-macos.yml`.

Resolve the model and processor to immutable Hugging Face revision
`482adb537c021c86670beed01cd58990d01e72e4`. Verified component roots are
`model.vision_model`, `model.connector`, and `model.text_model`; `lm_head` is
assigned to the language component for inventory purposes.

This environment is for architecture inspection, FP16 loading, accounting, and
unit tests only. PyTorch is built with MPS support but reports MPS unavailable
on this host, and CUDA/bitsandbytes are absent. NF4 candidates are explicitly
unsupported here. FP16 load success is not inference or quantization evidence.

## 2026-10-28 — Validated CUDA NF4 audit policy

Use the installed Transformers 4.52.4 bitsandbytes loader for M1 candidate
construction. For every candidate containing NF4, set quantization type to
NF4, FP16 compute, uint8 quantized storage, and double quantization disabled.
Keep `lm_head` in FP16 as a documented output-head exception. Keep an entire
component in FP16 by passing its verified module root in the loader's skip
list; audit the realized module classes and parameter quantization state after
loading rather than trusting that list. The complete eight-candidate audit at
commit `fdeea24367cf8f64fa9b5b8e9aea9de3beb5eafe` validated this behavior on a
Tesla T4; detailed evidence is recorded in `docs/M1_MAC_FINDINGS.md`.

Run each candidate in a fresh subprocess from the same pinned original
checkpoint revision. A candidate is supported only if every supported linear
module in each requested NF4 component is actually 4-bit, no requested FP16
component is quantized, no unclassified module is quantized, and the synthetic
inference succeeds. This policy is implemented and unit-tested but remains
validated by the T4 audit. This synthetic feasibility result does not establish
task accuracy or freeze the final experimental memory caps.

## 2026-10-28 — Proposed M2 processing architecture

Normalize POPE-COCO and CelebA into one strict yes/no record contract. Require
an explicit POPE variant and explicit CelebA attribute-to-question mappings;
do not guess or merge source schemas. Store portable image-relative paths and
verify source-image SHA-256 values for finalized manifests.

Split one dataset at a time before creating corruptions. Use the original image
as the POPE group and identity as the CelebA group. Order groups by a stable
SHA-256 of dataset, group ID, and a recorded seed, then allocate whole groups
to pilot/development/test. Manifests are canonical, checksummed, and immutable:
an identical rerun is allowed, but different content requires a new path or
protocol version.

Implement darkness as a Pillow RGB brightness factor and Gaussian blur as a
Pillow radius, currently before the model processor. Expand exactly one clean
condition plus explicitly supplied corruption severities. Severity values,
split fractions/seed, the transform stage, POPE variant, CelebA attributes,
and sampling/balance rules remain proposed until real-data and pilot validation.
Answerability decisions must be recorded for corrupted pilot variants using a
model-independent review; model correctness and test accuracy are not inputs.

## 2026-10-28 — Checksum-linked split groups and complete variant sets

Date: 2026-10-28
Decision: Treat groups that share a source-image SHA-256 as one indivisible
split unit, and require every split question to carry every declared
corruption condition before a variant manifest is accepted.
Reason and evidence: The protocol requires splitting by original image (and by
identity for CelebA) and paired evaluation. Synthetic probes showed identical
image bytes under two IDs could land in different splits undetected, and a
variant manifest missing some conditions still validated.
Alternatives considered: Report duplicate-checksum leakage only and let the
user fix inputs by hand; rejected because the split function can prevent it
deterministically at no cost.
Effect on comparability: None for data without shared checksums (assignments
are identical). Manifests that contain duplicates may change assignment and
must be regenerated.
Protocol version/hash: Proposed, pending real-data validation.
Owner: Gauri Chabukswar

## Template for a new decision

Date:
Decision:
Reason and evidence:
Alternatives considered:
Effect on comparability:
Protocol version/hash:
Owner:

## 2026-10-29 — Issues raised by the post-audit review

Four decisions were added to the open list above after reviewing the completed
M1 CUDA audit and the M2 pipeline. They are recorded here because each one can
silently invalidate comparisons if it is settled late or inconsistently.

**Double quantization.** The audited candidates use NF4 with double
quantization off, which costs 127,703,488 bytes of `absmax` metadata on the
fully NF4 candidate — exactly 0.0625 bytes per quantized parameter. Enabling it
would cut that to roughly 16 MB and therefore move every measured cap. The
setting must be frozen before the pilot and recorded with the cap definitions.

**Blur severity units.** `gaussian_blur` values are Pillow radii in absolute
pixels. A CelebA aligned crop is roughly 178 px wide and a COCO image roughly
640 px, so one radius is a far stronger corruption on CelebA. Either define the
radius relative to the short edge, or treat severities as per-domain and never
compare a raw severity across domains. Until this is settled, cross-domain
severity statements are not meaningful.

**Primary caps.** The smallest cap admitting two candidates, 1,587,421,408
bytes, satisfies M1 acceptance but compares two candidates that differ only in
connector precision. The caps admitting four and six candidates
(2,178,324,832 and 3,902,666,464 bytes) vary vision and backbone precision and
are the scientifically useful choices. Not frozen.

**Throughput.** The only measured generation figures, 3.78 to 4.05 seconds per
answer, are cold single calls including warm-up. Depending on steady-state
throughput the full design costs between roughly 26 and 106 GPU-hours.
Measuring this is the pilot's first task.

## 2026-10-29 — Corruption signature normalization

`CorruptionSpec.value` is now coerced to `float` and booleans are rejected.
Previously `2` and `2.0` produced different `signature` digests and `True` was
accepted as a blur radius, so one transform could hold three distinct cache
keys. Cache keys must include the image transformation, so this was a
correctness defect rather than a style issue. Two regression tests cover it.
