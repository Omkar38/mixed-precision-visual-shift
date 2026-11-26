# CelebA Download, Verification, and Sampling Guide

Date: 2026-11-19

This guide explains the CelebA work one step at a time. The values shown here
are placeholders. They are not frozen experiment settings.

For the storage-saving Google Drive/Colab workflow, use
[`notebooks/m2_celeba_colab_temp.ipynb`](../notebooks/m2_celeba_colab_temp.ipynb).
It keeps the ZIP in private Drive, samples annotations first, extracts only the
selected images into temporary Colab storage, verifies their checksums, creates
placeholder splits and pilot variants, and exports only small private artifacts.
The steps below remain the full local-extraction alternative.

## 1. What files are needed

Download these files from the official CelebA source after accepting its
research-use terms:

```text
img_align_celeba.zip
list_attr_celeba.txt
identity_CelebA.txt
```

Do not download CelebA from an unknown mirror. Do not commit the archive,
images, annotations, or generated manifests to Git.

The repository ignores `data/` and `artifacts/` for this reason.

## 2. Create local directories

Run these commands from the repository root:

```bash
mkdir -p data/celeba
mkdir -p artifacts/m2/celeba-placeholder
```

Place the three downloaded files under `data/celeba/`. A simple layout is:

```text
data/celeba/
  img_align_celeba.zip
  list_attr_celeba.txt
  identity_CelebA.txt
```

## 3. Check the downloaded files before extraction

Confirm that the files exist and are not empty:

```bash
ls -lh data/celeba/
```

Record both MD5 and SHA-256 values:

```bash
md5 data/celeba/img_align_celeba.zip
md5 data/celeba/list_attr_celeba.txt
md5 data/celeba/identity_CelebA.txt

shasum -a 256 data/celeba/img_align_celeba.zip
shasum -a 256 data/celeba/list_attr_celeba.txt
shasum -a 256 data/celeba/identity_CelebA.txt
```

Compare the MD5 values with the values published by the authorized source.
Record the SHA-256 values in the local validation report. A checksum only
proves which file was used. It does not prove that every label is correct.

Test the ZIP archive without extracting it:

```bash
unzip -t data/celeba/img_align_celeba.zip
```

Stop if this command reports a broken archive.

## 4. Extract and count the images

Extract the archive:

```bash
unzip -q data/celeba/img_align_celeba.zip -d data/celeba/
```

The expected image directory is:

```text
data/celeba/img_align_celeba/
```

Count the JPEG files:

```bash
find data/celeba/img_align_celeba -type f -name '*.jpg' | wc -l
```

The expected count is `202599`.

Check the first and last filenames:

```bash
find data/celeba/img_align_celeba -type f -name '*.jpg' | sort | head
find data/celeba/img_align_celeba -type f -name '*.jpg' | sort | tail
```

Do not continue if files are missing or the count is different. First check
whether the ZIP was fully downloaded and extracted into the expected folder.

## 5. Normalize the selected attributes and hash every image

Activate the project environment and install the local package:

```bash
conda activate mpvs-m1
python -m pip install -e .
```

The following command uses placeholder attributes. It reads the real
annotations, confirms that every referenced image exists, and attaches one
SHA-256 checksum to each source image:

```bash
mpvs m2-normalize-celeba \
  --attributes data/celeba/list_attr_celeba.txt \
  --identities data/celeba/identity_CelebA.txt \
  --image-root data/celeba/img_align_celeba \
  --attribute-question 'Eyeglasses=Is this person wearing eyeglasses?' \
  --attribute-question 'Wearing_Hat=Is this person wearing a hat?' \
  --attribute-question 'Smiling=Is this person smiling?' \
  --output artifacts/m2/celeba-placeholder/celeba-normalized.jsonl
```

Do not add `--skip-image-checksums` for a final manifest.

This command creates:

```text
artifacts/m2/celeba-placeholder/celeba-normalized.jsonl
artifacts/m2/celeba-placeholder/celeba-normalized.jsonl.metadata.json
```

The metadata records source hashes, manifest hashes, missing checksums, and
duplicate image-byte groups. Inspect it with:

```bash
python -m json.tool \
  artifacts/m2/celeba-placeholder/celeba-normalized.jsonl.metadata.json | less
```

The important checks are:

```text
validation.valid = true
validation.missing_source_image_checksums = []
validation.conflicting_celeba_image_identities = []
validation.conflicting_image_paths = []
validation.conflicting_source_image_checksums = []
```

Review `validation.duplicate_source_image_checksums` separately. Repeated image
bytes are reported for inspection. The split code keeps checksum-linked groups
together, and the sampler does not select the same bytes twice.

## 6. Create a deterministic balanced sample

The new sampler command is `m2-sample-celeba`.

This placeholder example selects, for each attribute:

- 200 `yes` records;
- 200 `no` records;
- at most one image from an identity for that attribute; and
- only one question for any selected source image.

```bash
mpvs m2-sample-celeba \
  --input artifacts/m2/celeba-placeholder/celeba-normalized.jsonl \
  --attribute Eyeglasses \
  --attribute Wearing_Hat \
  --attribute Smiling \
  --per-label 200 \
  --seed 590 \
  --output artifacts/m2/celeba-placeholder/celeba-sampled.jsonl
```

This produces `1200` records:

```text
3 attributes × 2 labels × 200 records = 1200 records
```

The command requires image checksums by default. This prevents an annotation-
only profile from being mistaken for a final sample.

For temporary annotation-only testing, this flag exists:

```text
--allow-missing-image-checksums
```

Do not use that flag for final manifests.

The sampler writes an immutable manifest and a metadata file. Running the same
command again is allowed and produces the same bytes. Changing the seed or
sample size requires a new output path.

Inspect the result:

```bash
python -m json.tool \
  artifacts/m2/celeba-placeholder/celeba-sampled.jsonl.metadata.json | less
```

Check that:

```text
validation.valid = true
validation.record_count = 1200
each attribute has yes = 200 and no = 200
validation.duplicate_original_image_ids = []
validation.repeated_identity_attribute_pairs = []
validation.duplicate_source_checksum_groups = []
validation.missing_source_image_checksums = []
```

## 7. Create placeholder identity-disjoint splits

The following fractions are examples only:

```bash
mpvs m2-split \
  --input artifacts/m2/celeba-placeholder/celeba-sampled.jsonl \
  --output-dir artifacts/m2/celeba-placeholder/splits \
  --seed 590 \
  --pilot-fraction 0.083333333333 \
  --development-fraction 0.5 \
  --test-fraction 0.416666666667
```

The splitter assigns whole identities. The final question counts may not equal
the requested fractions exactly. Check the achieved counts in:

```text
artifacts/m2/celeba-placeholder/splits/split-summary.json
```

Run the independent leakage check:

```bash
mpvs m2-validate-splits \
  artifacts/m2/celeba-placeholder/splits/pilot.jsonl \
  artifacts/m2/celeba-placeholder/splits/development.jsonl \
  artifacts/m2/celeba-placeholder/splits/test.jsonl
```

The report must show no image, identity, group, question, or checksum leakage.

## 8. Check visible labels by eye

Checksums and tests cannot prove that a face really has the labeled attribute.
A human must inspect a pilot sample.

For each selected attribute and label:

1. Open a small random-looking but deterministic set from the pilot manifest.
2. View the clean image and its question.
3. Record whether the source label is visibly reasonable, unclear, or wrong.
4. Do not use model answers during this review.
5. Report the number reviewed and all disagreements.

Use `scripts/m2_pilot_review.py` after expanding the pilot into clean and
corrupted variants. That page helps inspect the images and export answerability
decisions. It does not decide whether a label is correct automatically.

Do not silently delete a difficult example. Define and record any exclusion
rule before the main experiment.

## 9. When the placeholder run becomes final

Before calling the sample final, the team must approve and record:

- selected attributes and exact question wording;
- records per label and attribute;
- sampling seed;
- split seed and target sizes;
- handling of unclear or incorrect source labels;
- permission for storing or publishing split filenames; and
- accepted pilot corruption severities.

Record these choices in `docs/DECISIONS.md`. Then rerun normalization, sampling,
splitting, and validation into a new versioned output directory. Keep the
placeholder artifacts for audit history; do not overwrite them.
