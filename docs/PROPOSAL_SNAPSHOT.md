# Proposal snapshot

Source: [CSE590 Project Proposal — Final tab](https://docs.google.com/document/d/1gtlE4NJRYpqTqe1G5gV5SIBgzHL_DG9FJyWphVvco9Q/edit?tab=t.9i03s424878a)

Read on 2026-10-05. Transcribed from the current document; smart chips are rendered as text. Student IDs and contact emails are omitted from this repository copy. This is a proposal, not an experimental result or evidence of professor approval.

CSE590-01 | Project Proposal | Fall 2026

## 1. Project Title

Mixed Precision Allocation Under Visual Distribution Shift

## 2. Group Members

Three group members. Names, student IDs, and email addresses are recorded in the
proposal document and are intentionally omitted from this repository.

## 3. Problem Statement

Background/Context. Quantization lowers memory by storing weights with fewer bits. VLM components (vision encoder, connector, and language backbone) differ in compression sensitivity [1]. Prior work studies quantized-model robustness [2],we study whether the choice of compression setup transfers across image conditions.

Problem Definition. Under the same memory limit, does a setup chosen on clean images still work well on darker or blurrier images? We will measure the cost of clean-data selection, the benefit of mixed selection data, and each component’s precision sensitivity. For example, does a setup chosen on clear photos remain effective on blurred versions?

## 4. Proposed Approach

Idea/Methodology. We will test SmolVLM2-2.2B [6] with up to eight 4-bit NF4/FP16 combinations across its three components, auditing supported linear layers and exceptions [7]. Each setup is quantized once,development data selects among fixed setups without training or weight calibration.

We will split by original image before creating darker and Gaussian-blurred versions at three pilot-checked severities [3]. All questions and variants stay in the same split,CelebA is additionally separated by identity. We will check labels and answerability without filtering by model correctness.

At each measured weight-memory cap with at least two feasible configurations, we will select the best development accuracy using clean-only, target-only, or mixed data (25%, 50%, or 75% target). Sample size, distinct-image count, candidate set, prompts, resolution, and decoding stay fixed. Ties favour lower memory, then a fixed order. Selected setups are evaluated on the same untouched target test set.

Scope & Feasibility. The core uses one model, two domains, and two corruptions. A 100-question pilot will establish memory, runtime, and sample size. At 500 test questions/domain, eight setups and seven image conditions require 56,000 answers, plus development runs. Cached predictions support all caps and resampling. If needed, we will reduce severity count or use the 500M model,extensions follow the core.

## 5. Datasets and Resources

Datasets. POPE-COCO provides 3,000 balanced yes/no object questions over 500 images per sampling variant,we will prespecify one variant [4]. CelebA contains 202,599 images and 40 attributes,we will sample visible attributes such as glasses and balance yes/no labels [5]. Both have public research access. CelebA images remain internal under its research-use agreement,we will release code and split identifiers. Pilot, development, and test groups will be disjoint.

Resources. PyTorch, Transformers, bitsandbytes, image transforms, and NumPy/SciPy on the available 8GB GPU, one setup at a time. Memory caps include quantization metadata,peak inference memory is reported separately. Extra models, iNaturalist, TextVQA/GQA, and additional corruptions are optional extensions requiring separate task and resource checks.

## 6. Plan for Evaluation

Evaluation Metrics. Our main result is Δ = target-selected accuracy − source-selected accuracy on the same target test set, in percentage points. Positive Δ means clean-data selection cost accuracy,zero or negative values are possible. We will also report accuracy, precision/recall/F1, invalid-answer rate (counted incorrect), selection frequencies, and accuracy–memory curves.

Paired 95% bootstrap intervals will resample images (identities for CelebA), retaining all questions and variants. Development resampling will separately assess selection stability. With all eight configurations, matched comparisons will measure component-precision effects under shift,these are not proof of an internal mechanism. Cross-domain selection will be exploratory, holding image condition fixed and acknowledging task differences.

Baseline Comparisons. Clean-only selection, uniform supported-layer 4-bit, and FP16. FP16 measures the model’s own degradation,we will report any additional accuracy drop caused by quantization. It is not an accuracy upper bound. Test-set-best accuracy defines a descriptive, optimistic regret reference, never a selection rule.

Success Criteria. A reproducible estimate of selection-transfer cost with uncertainty and an accuracy–memory comparison. A small or uncertain effect remains valid,conclusions will be limited to the tested models, tasks, and severities.

## 7. Timeline & Milestones

Oct 12, 2026: data access, label checks, component-control pilot, and frozen protocol.

Oct 19, 2026: baseline and main runs,Oct 27, 2026: midterm report with preliminary results.

Nov 9, 2026: mixture, component, and cross-domain analyses,Nov 23, 2026: final draft and reproducibility checks,Dec 10, 2026: final report, code, and presentation.

Responsibilities: evaluation lead — evaluation and validation; pipeline lead — data pipeline and quantization; design lead — experiment design, analysis, and writing.

## 8. References

[1] Shin, H., et al. (2026). Rethinking Small VLM Quantization: From Component-Wise Analysis to Hardware-Aware Edge Deployment. arXiv:2607.08029.

[2] Bouguerra, A., et al. (2025). Can Less Precise Be More Reliable? A Systematic Evaluation of Quantization’s Impact on CLIP Beyond Accuracy. arXiv:2509.21173v1.

[3] Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR.

[4] Li, Y., et al. (2023). Evaluating Object Hallucination in Large Vision-Language Models. EMNLP. POPE code and data.

[5] Liu, Z., et al. (2015). Deep Learning Face Attributes in the Wild. ICCV. CelebA dataset and research-use agreement.

[6] Hugging Face. SmolVLM2-2.2B-Instruct. Official model card.

[7] Hugging Face. Bitsandbytes. Transformers quantization documentation.

# Sources and provenance

The current proposal was read on 2026-10-05 from the [Final tab](https://docs.google.com/document/d/1gtlE4NJRYpqTqe1G5gV5SIBgzHL_DG9FJyWphVvco9Q/edit?tab=t.9i03s424878a). Its snapshot is in PROPOSAL_SNAPSHOT.md. This repository's scientific scope follows that source; newly proposed implementation details are labeled as such.

## Proposal references

1. Shin, H., et al. (2026). *Rethinking Small VLM Quantization: From Component-Wise Analysis to Hardware-Aware Edge Deployment.* [arXiv:2607.08029](https://arxiv.org/abs/2607.08029). Role: related component-wise quantization work.
2. Bouguerra, A., et al. (2025). *Can Less Precise Be More Reliable? A Systematic Evaluation of Quantization's Impact on CLIP Beyond Accuracy.* [arXiv:2509.21173v1](https://arxiv.org/html/2509.21173v1). Role: related robustness evaluation; do not assume CLIP findings automatically apply to generative VLMs.
3. Hendrycks, D., and Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* [arXiv:1903.12261](https://arxiv.org/abs/1903.12261). Role: corruption evaluation precedent, not evidence that arbitrary transforms are exact benchmark reproductions.
4. Li, Y., et al. (2023). *Evaluating Object Hallucination in Large Vision-Language Models.* [POPE repository](https://github.com/RUCAIBox/POPE). Role: questions/annotations and dataset instructions.
5. Liu, Z., et al. (2015). *Deep Learning Face Attributes in the Wild.* [CelebA project and agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Role: attribute data, identities, and usage terms.
6. Hugging Face. [SmolVLM2-2.2B-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct). Role: official checkpoint and inference guidance.
7. Hugging Face. [Bitsandbytes quantization documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes). Role: supported quantization APIs; actual component coverage still requires a runtime audit.

