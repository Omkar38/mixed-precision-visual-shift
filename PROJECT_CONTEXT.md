# Project context for Claude Code and Codex

## 1. What we are building

We are building a reproducible experiment pipeline to study **Mixed Precision Allocation Under Visual Distribution Shift**.

A vision-language model receives an image and a question, then produces a text answer. We will make different versions of the same pretrained model by storing selected components' supported weights at either 4-bit NF4 or FP16 precision. Lower precision can reduce weight memory, but it may affect accuracy.

We will select a configuration using clear development images, then test it on darker or blurrier images. We will compare that with choosing a configuration using development images from the target condition. Both selection strategies must obey the same weight-memory limit.

We are not training a new model, creating an image-restoration system, or inventing a new quantization algorithm. Our contribution is the controlled evaluation of whether the **choice of precision allocation transfers across image conditions**, and whether mixed development data helps.

This handoff is based on the Final tab of the CSE590 proposal, read on 2026-10-05. See `docs/PROPOSAL_SNAPSHOT.md`. No professor approval or experimental outcome is established by this repository.

## 2. A concrete example

Suppose a clear photo contains a dog. The question is “Is there a dog in the image?” and the label is “yes.”

We create darker and blurred versions of the photo, retaining the label only where the task remains answerable under our prespecified audit. We ask the same question using each supported model configuration.

On development data, configuration A might perform best on clear photos, while B performs best on dark photos. We evaluate A and B on the same held-out dark test photos.

**Hypothetical illustration only:** If A scores 70% and B scores 74%, the clean-selection cost is 4 percentage points. These numbers are not results. If both selection methods choose A, the difference is zero. If B performs worse on the test set, the difference is negative; that outcome must also be reported.

## 3. Model components and candidate configurations

Primary checkpoint: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`.

| Component | Role | Candidate stored-weight precision |
| --- | --- | --- |
| Vision encoder | Converts image content into visual representations | FP16 or NF4 for supported linear layers |
| Connector | Maps visual representations into the language model's representation space | FP16 or NF4 for supported linear layers |
| Language backbone | Processes the question and visual representations to generate an answer | FP16 or NF4 for supported linear layers |

There are at most 2 × 2 × 2 = 8 requested combinations. The exact module paths and coverage must be discovered from the installed checkpoint and library implementation.

“NF4 component” does not mean every tensor is four bits. Embeddings, normalizations, biases, output heads, convolutions, or unsupported layers may remain at higher precision. Record every exception and measure its storage. Global quantization flags alone do not prove independent control of the three components.

Use FP16 as the unquantized reference in this experiment. It is a higher-precision reference relative to NF4, not an FP32 model or a guaranteed accuracy upper bound.

Fallback, if the pilot requires it: reduce the number of severities or use the 500M model named in the proposal. Verify the exact fallback checkpoint before implementation. Treat a model change as a new protocol version, not a silent substitution.

## 4. Research questions

1. Under a fixed weight-memory cap, how much target-test accuracy is lost or gained when the configuration is selected on clean instead of target-condition development images?
2. Do mixtures containing 25%, 50%, or 75% target-condition development examples reduce that difference?
3. When the full supported configuration grid is available, how does changing one component's precision affect clean and shifted performance?

Cross-domain selection is exploratory and must hold image condition fixed while acknowledging different question types. Additional models, iNaturalist, TextVQA/GQA, and additional corruptions are optional extensions.

Do not claim novelty beyond the evidence from the literature review. Related work already studies component sensitivity and robustness under quantization.

## 5. Inputs

- A fixed pretrained checkpoint and processor, with immutable revisions recorded.
- Images, yes/no questions, and ground-truth labels.
- Original image IDs and, for CelebA, identity IDs.
- Disjoint pilot, development, and test manifests.
- Corruption definitions and severity parameters, frozen after the pilot.
- Requested precision allocations and audited actual allocations.
- Measured weight-memory caps, a fixed selection rule, prompts, and decoding settings.

### POPE-COCO

The proposal specifies one prespecified sampling variant, with 3,000 balanced yes/no questions over 500 images for that variant. Obtain the official annotations and corresponding COCO images separately, verify actual counts and schema, and record source checksums. Do not merge variants casually: images/questions may overlap.

All questions for an image must stay in one split. Choose the sampling variant and split sizes before looking at test predictions.

### CelebA

The proposal uses a subset of visible facial attributes such as glasses, converted to yes/no questions. The source dataset has 202,599 images and 40 attributes. Verify downloaded annotations, identity metadata, attribute polarity, and access terms.

Choose visible attributes and balance labels using a documented sampling rule. Split by identity so the same person cannot appear across pilot/development/test. Do not silently rebalance every attribute in a way that produces unreported sampling differences. Keep the images internal under the dataset agreement.

For both datasets, check answerability without conditioning inclusion on model correctness. Pilot checks should choose severity limits. Any later quality exclusions must be model-blind, documented, applied consistently across paired configurations, and frozen before scoring the affected evaluation.

## 6. Processing pipeline

1. Verify hardware, dependency compatibility, dataset availability, and checkpoint access.
2. Audit model components and independently controllable quantization; reject unsupported or duplicate effective configurations.
3. Create immutable group-disjoint split manifests.
4. Apply darkness and Gaussian blur at three pilot-checked severities each, plus clean images: up to seven conditions.
5. Evaluate each supported configuration on development and test conditions using fixed prompts, resolution, decoding, and parsing. Cache raw answers and parsed outcomes.
6. Use development predictions only to select configurations under each common memory cap.
7. Evaluate the selected configurations on the same target test examples; compare accuracy and quantify uncertainty.
8. Generate tables/plots and a report with limitations and reproducibility information.

Keep test labels and aggregate test results out of selection and protocol-tuning code. The runner may cache predictions for every candidate, but the selector must accept development records only.

## 7. Main measurement

For target condition t and memory cap B:

- Eligible candidates: supported configurations whose measured weight storage is at most B and that meet the frozen execution constraints.
- Source-selected configuration: highest clean-development accuracy among eligible candidates.
- Target-selected configuration: highest target-development accuracy among the same candidates.
- Mixed-selected configurations: highest development accuracy for each fixed mixture.
- Tie-break: lower measured weight storage, then a prespecified configuration order.

The primary result is:

**Δ(B,t) = 100 × [target-test accuracy of target-selected configuration − target-test accuracy of source-selected configuration].**

Accuracy in this formula is a fraction. Δ is reported in percentage points. Positive Δ means clean selection cost accuracy in this comparison. Zero and negative values are possible.

Use caps admitting at least two distinct, supported configurations. Choose caps based on measured storage and feasibility before examining comparative test accuracy. Equal eligibility limits do not imply equal realized memory usage; report both the cap and selected configuration's measured storage.

Report each dataset, corruption, severity, and cap separately. Any overall summary needs a prespecified weighting rule. Do not cherry-pick the largest gap as the main finding.

## 8. Fair comparisons and uncertainty

Use the same development question count, distinct-image count, candidate set, prompts, image processing, and decoding settings for clean/target/mixed selection. A practical proposed mixture design assigns either clean or target imagery to each development image group; it must not duplicate clean and corrupt versions to inflate sample size. Freeze the assignment and rounding rule.

Compute paired 95% bootstrap intervals by resampling image groups for POPE and identity groups for CelebA, carrying every associated question/variant with the group. Keep selected configurations fixed within a test bootstrap. Report development-selection stability separately by resampling development groups and rerunning selection.

Additional metrics: accuracy, precision/recall/F1 with explicit invalid-output conventions, invalid-answer rate, configuration selection frequencies, memory/runtime, and accuracy-versus-memory curves.

Compare against uniform supported-layer NF4 and FP16. Show intrinsic FP16 degradation under corruption and the extra degradation associated with quantization. The best configuration on the test set is an explicitly optimistic descriptive reference only; never use it for actual selection.

## 9. Feasibility and outputs

The proposal assumes an available 8GB GPU; its actual model and software support are not yet verified. Begin with a disjoint 100-question pilot, one configuration at a time.

The proposal's illustrative test workload is 2 domains × 500 questions × 8 configurations × 7 conditions = **56,000 generated answers**, excluding development and pilot runs. Actual split sizes must respect whole image/identity groups; 500 is a planning estimate, not permission to split a group.

Report measured model-weight bytes separately from peak GPU inference memory. Activations, KV cache, temporary buffers, and runtime overhead can cause out-of-memory failures even if weight storage fits the cap.

Expected deliverables:
- Module/dtype/quantization audit and supported candidate list.
- Dataset/split manifests and corruption specification.
- Resumable prediction cache with reproducibility metadata.
- Selection table for each dataset, target, cap, and mixture.
- Accuracy differences with cluster-aware uncertainty and selection stability.
- Accuracy-memory plots, component comparisons where identifiable, and a final report.
- Code and permitted split identifiers; no redistribution of restricted images.

## 10. Team and schedule from the proposal

- Evaluation lead: evaluation and validation.
- Pipeline lead: data pipeline and quantization.
- Design lead: experiment design, analysis, and writing.

| Date (2026) | Planned milestone |
| --- | --- |
| October 12 | Data access, label checks, component-control pilot, frozen protocol |
| October 19 | Baselines and main runs |
| October 27 | Midterm report with preliminary results |
| November 9 | Mixture, component, and exploratory cross-domain analyses |
| November 23 | Final draft and reproducibility checks |
| December 10 | Final report, code, and presentation |

These dates are copied from the proposal, not newly verified course deadlines.

## 11. Decisions still requiring pilot evidence

Exact library versions and model revision; usable GPU; module mapping; supported candidate set; actual storage measurements; POPE variant; selected CelebA attributes; group split sizes; severity parameters; image resolution and prompt; parsing rule; development mixture construction; memory caps; bootstrap settings; and fallback necessity.

The example configuration makes these gaps visible. The first agent should resolve them through small verified steps and a decision log, not fill them with invented “final” values.
