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

## Implementation documentation checked for this handoff

On 2026-10-05, the official model card, Transformers bitsandbytes documentation, and [Claude Code memory instructions](https://code.claude.com/docs/en/memory) were inspected. The Claude entry file uses the documented relative import of AGENTS.md.

Model/library APIs change. Pin versions that pass the actual pilot; do not copy code from an unversioned documentation page and assume compatibility. Record revision identifiers and access dates when adding new scientific claims. The handoff is not a new exhaustive literature review.
