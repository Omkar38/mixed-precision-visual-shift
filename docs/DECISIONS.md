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

## 2026-11-19 — Proposed deterministic CelebA sampling policy

Date: 2026-11-19
Decision: Implement an explicit, seeded CelebA sampler that requests the
attributes and exact per-label count at the command line. Process attributes in
stable sorted order. For each attribute, select the label with fewer available
identities first, rank candidates with SHA-256 of the seed and record identity,
and require exact yes/no balance. Select at most one image per identity for an
attribute and at most one question per source image. Do not select the same
image bytes twice when checksums are available. Require image checksums for a
final sample; allow missing checksums only through an explicit temporary flag.
Reason and evidence: Real CelebA profiling showed strong label imbalance, and
615 identities have mixed `Male` labels. The experiment needs deterministic
balance without allowing one identity or image to dominate an attribute.
Synthetic tests verify exact counts, input-order independence, identity/image
limits, immutable CLI output, missing-checksum rejection, and clear failure
when a requested sample is impossible.
Alternatives considered: Sample every label independently without an identity
limit; rejected because one person could contribute repeatedly. Permit one
image to answer several attributes; rejected for the core sample so question
counts also represent distinct source images. Hard-code the current example
attributes and counts; rejected because those values still require team and
pilot approval.
Effect on comparability: The seed, attributes, per-label count, source-manifest
hash, and sampler version must match across reruns. Changing any of them creates
a new manifest and protocol version.
Protocol version/hash: Proposed and synthetic-tested; pending team approval and
real-image validation.

## 2026-11-19 — Checksum-gated CelebA download fallback

Date: 2026-11-19
Decision: Try the official CelebA Google Drive files first. If Google rejects a
download because of its public quota, the temporary Colab notebook may retrieve
the same named file from the Freie Universität Berlin academic CelebA mirror.
Accept and copy that file into private project storage only when its MD5 exactly
matches the value published for `torchvision.datasets.CelebA`.
Reason and evidence: Google quota failures affected both the image archive and
the attribute annotations. The academic mirror states that it exists for this
quota failure and provides the files used by torchvision. The strict checksum
gate prevents a differently versioned or corrupted file from entering the M2
pipeline.
Alternatives considered: Repeatedly retry Google Drive or require manual
downloads; retained as fallbacks, but they block an otherwise reproducible
Colab run. Use an unchecked community copy; rejected.
Effect on comparability: None when the checksum matches, because the accepted
bytes are identical under the recorded MD5. Record the retrieval URL and hashes
in the run artifacts. The CelebA license agreement still applies.
Protocol version/hash: Operational data-retrieval decision; no experiment
parameter is frozen or changed.
Owner: Team decision pending.

## 2026-11-26 — Accepted CelebA core corruption severities

Date: 2026-11-26
Decision: For the CelebA core experiment, retain two darkness factors (`0.60`
and `0.35`) and two Pillow Gaussian-blur radii (`0.5` and `1.0`). Apply the
deterministic RGB transforms before the model processor, as implemented. Reject
darkness factor `0.15` and blur radius `2.0` as whole severity levels. Use the
versioned specification in `configs/m2/celeba-corruptions-v1.json`. This is a
CelebA decision only; POPE-COCO requires its own pilot because image resolution
and question content differ.
Reason and evidence: A private, model-blind review covered all 588 corrupted
variants from 98 CelebA pilot records. The first validated decision file had no
missing, duplicate, unknown, or non-pilot IDs and reported 559 answerable and
29 unanswerable variants. Before a reviewer consistency correction, the
accepted levels were 97.96%, 98.98%, 98.98%, and 97.96% answerable,
respectively. The rejected darkness `0.15` and blur `2.0` levels were only
87.76% and 88.78% answerable. One question (`celeba::163296.jpg::Smiling`)
produced three monotonicity flags; the reviewer re-inspected that image and
marked all six corrupted versions answerable. The corrected private decision
file has 588 decisions: 561 answerable and 27 unanswerable, with zero missing,
unknown, or monotonicity violations. Every retained level has 97/98 (98.98%)
answerable decisions. Preserve the original decisions, corrected decisions,
correction log, per-severity report, and source hashes in private artifacts.
The repository records the user-reported counts and hashes in
`configs/m2/celeba-pilot-evidence-v1.json` but does not contain or independently
inspect the restricted artifacts themselves.
Alternatives considered: Keep the strongest levels and remove only their
unanswerable records; rejected because it would break complete paired cohorts
and make the evaluated sample severity-dependent. Test replacement levels
`0.25` and `1.5`; deferred because the proposal permits up to three levels and
the two retained levels already passed the pilot. Add new corruption families;
deferred until the core experiment is complete.
Effect on comparability: Every CelebA candidate configuration will use the same
clean cohort and four accepted corrupted cohorts. The rejected levels will not
appear in development or test evaluation. Changing these values, adding a
third level, or applying them at another transform stage creates a new protocol
and artifact version. POPE values are not inferred from this decision.
Protocol version/hash: Accepted CelebA final-candidate decision; the overall
experiment protocol remains unfrozen until final manifests, POPE severities,
generation settings, memory caps, and hashes are recorded.
Owner: Kaushal Shelke, model-blind human reviewer.

## 2026-11-26 — Accepted CelebA core sample and split design

Date: 2026-11-26
Decision: Keep the core CelebA task at three visible attributes:
`Eyeglasses` ("Is this person wearing eyeglasses?"), `Wearing_Hat` ("Is
this person wearing a hat?"), and `Smiling` ("Is this person smiling?").
Select 200 yes and 200 no records per attribute with sampling seed `590`, one
question per selected source image, and the existing identity cap. Split with
seed `590` and requested pilot/development/test fractions `1/12`, `1/2`, and
`5/12`. Treat the observed 98/605/497 records as the accepted final-candidate
allocation, subject to byte-identical regeneration and final manifest hashes.
Reason and evidence: The private real-data run produced 1,200 selected records
with exact per-attribute label balance, checksum-verified selected images, and
a validation-passing identity-disjoint split. The 98-record pilot supported a
complete 588-decision corruption review. One question per image keeps question
and image counts aligned and prevents a few faces from contributing several
attribute questions to the core result.
Alternatives considered: Add more attributes or reuse each image for multiple
questions; deferred as a versioned future extension because those choices
change task composition, effective sample size, sampler validation, mixture
assignment, and clustered analysis. Change the current seeds or force exact
split counts; rejected without evidence, because whole-identity grouping takes
priority over exact target counts.
Effect on comparability: Every model configuration and corruption condition
must use the same selected questions and identity-disjoint split manifests.
Changing attributes, questions, sample counts, seeds, image-reuse policy, or
fractions creates a new protocol version. Report both record and identity
counts.
Protocol version/hash: Accepted final-candidate data design. The main protocol
remains unfrozen until the manifests are regenerated into a non-placeholder
version, their hashes are recorded, and POPE/M3 settings are resolved.
Owner: Kaushal Shelke.

## 2026-11-26 — Frozen private CelebA M2 core artifacts

Date: 2026-11-26
Decision: Treat the private `celeba-core-v1` sample, split summary, five-condition
pilot variants, mapped answerability decisions, mapping record, and run summary
as the frozen CelebA M2 core artifacts identified in
`configs/m2/celeba-core-v1-evidence.json`. Do not overwrite those files. Any
change to the sample, splits, questions, corruption values/IDs, or review
decisions requires a new run name and protocol version.
Reason and evidence: The non-placeholder run reproduced the accepted
1,200-record sample with SHA-256
`582c28314b7f936eef59b0c208043f9c8d76adcfae6fbc3746432f0432b6ce20`.
It generated 490 pilot variants: 98 clean and 392 corrupted. The 392 reviewed
decisions mapped by exact question ID, corruption kind, and numeric value from
the earlier model-blind review. Validation reported 388 answerable and 4
unanswerable decisions, with no missing, duplicate, unknown, or non-pilot IDs.
The variant, decision, mapping, split-summary, and run-summary hashes are
recorded in the evidence JSON. Restricted manifests and images remain private;
the repository records user-reported counts and hashes rather than their
contents.
Alternatives considered: Repeat the same visual review after changing only
severity IDs; rejected as unnecessary because the source questions, image
checksums, transform kinds, and numeric values were unchanged and the mapping
is explicit and hashed. Reuse or rename the placeholder directory; rejected to
preserve provenance.
Effect on comparability: All CelebA M3/M4 runs must reference these exact
private hashes and `configs/m2/celeba-corruptions-v1.json`. A mismatched hash is
an incompatible run. POPE-COCO remains a separate data/severity decision.
Protocol version/hash: CelebA M2 core frozen as `celeba-core-v1`; the overall
experiment protocol remains unfrozen pending POPE and M3 settings.
Owner: Kaushal Shelke.

## Template for a new decision

Date:
Decision:
Reason and evidence:
Alternatives considered:
Effect on comparability:
Protocol version/hash:
Owner:
