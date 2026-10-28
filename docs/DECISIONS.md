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

Created a documentation-first repository based on the current Final tab. No experiment has run. Student IDs and teammates' email addresses were omitted from the repository snapshot; member names and responsibilities were retained.

The implementation protocol adds proposed choices (strict parsing, image-group mixture assignment, bootstrap count) to make engineering concrete. These are explicitly not yet pilot-validated or frozen.

Git commits made on Omkar's behalf use Omkar Lashkare and the email listed in the proposal. GitHub account association remains unverified. Publication defaults to private until Omkar chooses otherwise.

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

## Template for a new decision

Date:
Decision:
Reason and evidence:
Alternatives considered:
Effect on comparability:
Protocol version/hash:
Owner:
