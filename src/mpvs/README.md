# MPVS implementation

M1 inventories the immutable SmolVLM2 checkpoint, independently controls the
vision/connector/language precision allocation, audits realized NF4 state, and
measures unique weight storage. The complete eight-candidate grid was validated
on a T4; see `docs/M1_MAC_FINDINGS.md`.

M2 provides strict POPE/CelebA normalization, portable image checksums,
deterministic image/identity-group splits, immutable manifests, deterministic
darkness/Gaussian-blur transforms, variant inheritance checks, and a
model-independent pilot answerability review contract. See
`docs/M2_PIPELINE.md` for the real-data handoff and CLI commands.

```bash
conda env create -f environment-macos.yml
conda activate mpvs-m1
python -m pip install -e .
mpvs candidates
mpvs audit \
  --candidate v-fp16_c-fp16_l-fp16 \
  --device cpu \
  --output artifacts/m1-fp16.json
```

Add `--smoke-inference` only on a host with a validated execution backend.
NF4 requests on a non-CUDA host exit nonzero with a structured `unsupported`
record. Generated audits and model caches are intentionally ignored. Use the
Colab notebook for isolated CUDA candidate construction and grid summarization.
`requirements-macos-lock.txt` is the complete tested package inventory;
`environment-macos.yml` is the environment creation entry point.
The editable install makes `mpvs` importable without setting `PYTHONPATH`.
