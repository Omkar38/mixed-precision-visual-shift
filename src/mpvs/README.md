# M1 model audit

The current vertical slice inventories the immutable SmolVLM2 checkpoint, its
vision/connector/language module mapping, stored tensor dtypes, and unique
weight storage. It does not yet implement or validate bitsandbytes NF4.

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
NF4 requests exit nonzero with a structured `unsupported` record until the
separate CUDA/bitsandbytes implementation is audited. Generated audits and
model caches are intentionally ignored.
`requirements-macos-lock.txt` is the complete tested package inventory;
`environment-macos.yml` is the environment creation entry point.
The editable install makes `mpvs` importable without setting `PYTHONPATH`.
