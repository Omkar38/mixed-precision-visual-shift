# M2 Real-Data Validation (Phase A)

Owner: Gauri Chabukswar. Date: 2026-10-28. Follows Kaushal's M2 handoff runbook.

Status legend: **validated on real data** = the command ran on the real source
files with the result shown; **trial** = run on real data with placeholder
settings only to test the pipeline (nothing frozen); **proposed** = a
recommendation awaiting team approval.

Phase A (inspect and profile) is complete for POPE-COCO and for CelebA
annotations. Phase B (final manifests) and Phase C (pilot answerability) are
**not done**: they require the team decisions in section 5 and, for Phase C, a
human visual review. M2 is not complete.

## 1. Environment

- macOS laptop, CPU only, Python 3.14 virtual environment, `pip install -e .`.
- Branch `Gauri-Commits` = Kaushal's `cb9feb6` + review fix `31ae570`.
- `python -m pytest -q` → 40 passed before any real data was used.
- Raw data lives under the ignored `data/` tree; generated manifests under the
  ignored `artifacts/` tree. Neither is committed.

## 2. POPE-COCO

### 2.1 Sources

| File | Source | SHA-256 |
| --- | --- | --- |
| `coco_pope_random.json` | github.com/AoiDragon/POPE `output/coco/` (last change to that folder: commit `109985c`, 2023-11-16) | `ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994` |
| `coco_pope_popular.json` | same | `72c1a8ad45d0c13514f5f22598261df41d3b533854d29682e924db50ed8aa753` |
| `coco_pope_adversarial.json` | same | `420b3407db1fa9f1187a805dca41cb7b97fd91504e6c2179706188c107fb8ef8` |
| 500 COCO images | `images.cocodataset.org/val2014/` (only the images POPE references, 79 MB) | per-image SHA-256 stored in the ignored manifests |

Files use the `.json` extension but are JSON Lines.

### 2.2 Schema and counts (validated on real data)

- Fields in every row: `question_id`, `image`, `text`, `label`. These match
  the loader defaults, so **no field arguments or code changes were needed**.
- `image` is a bare filename such as `COCO_val2014_<12 digits>.jpg`; the loader
  derives `original_image_id` from the filename stem.

| Variant | Questions | Distinct images | Yes / No | Duplicate question IDs | Questions per image |
| --- | --- | --- | --- | --- | --- |
| random | 3000 | 500 | 1500 / 1500 | 0 | 6 for all 500 |
| popular | 3000 | 500 | 1500 / 1500 | 0 | 6 for all 500 |
| adversarial | 3000 | 500 | 1500 / 1500 | 0 | 6 for all 500 |

- The three variants use the **same 500 images**. Every image has 3 yes and 3
  no questions in every variant.
- These reproduce the proposal's "3,000 balanced questions over 500 images per
  variant".
- All 3000 questions match the template `Is there a/an <object> in the image?`.
- Negative objects: random uses 79 distinct objects, popular 10 (dominated by
  car, dining table, chair), adversarial 57 (dominated by car, dining table,
  person).
- Missing images: 0. All 500 images verified with SHA-256 during normalization.
- Byte-identical images under different IDs: 0 (500 distinct checksums).
- Image properties: 499 RGB, 1 grayscale (`L`, converted to RGB by
  `apply_corruption`); no EXIF orientation tags; width 333–640 (median 640),
  height 234–640 (median 480).
- Determinism: re-running `m2-normalize-pope` for `random` gave the same
  manifest SHA-256 (`e2b1b24d…`).

### 2.3 Proposed POPE variant

**Proposed: `adversarial`.** It is the hardest of the three (negatives are
objects that often co-occur with the scene), so clean accuracy is less likely
to sit at a ceiling and leave no room to measure degradation under darkness
and blur. `random` is the conservative alternative if the pilot shows accuracy
is already too low on clean images. `popular` uses only 10 negative objects,
which narrows what the task measures. Team approval required.

## 3. CelebA

### 3.1 Sources and terms

- Official CelebA Google Drive files (the same file IDs and MD5 values used by
  `torchvision.datasets.CelebA`):

| File | MD5 (matches official) | SHA-256 |
| --- | --- | --- |
| `Anno/list_attr_celeba.txt` | `75e246fa4810816ffd6ee81facbd244c` | `f0e5da289d5ccf75ffe8811132694922b60f2af59256ed362afa03fefba324d0` |
| `Anno/identity_CelebA.txt` | `32bd1bd63d3c78cd57e08160ec5ed1e2` | `c6143857c3e2630ac2da9f782e9c1232e5e59be993a9d44e8a7916c78a6158c0` |

- Terms: the CelebA agreement allows non-commercial research use only and does
  not allow redistributing images. Images and annotation files stay
  internal; only code, aggregate counts and (later) split identifiers may be
  committed. The team should confirm that redistributing split *filenames* is
  allowed before committing final manifests.
- **Images not downloaded yet** (1.4 GB `img_align_celeba.zip`). Image
  existence and checksums are therefore unchecked for CelebA; profiling used
  `--skip-image-checksums`.

### 3.2 Schema and counts (validated on real data)

- Declared images 202,599 = parsed rows 202,599. 40 attributes. Values are only
  `1` and `-1`.
- Identities: 10,177. Every image has an identity. Images per identity:
  min 1, median 21, max 35.
- Loader test: `Eyeglasses=Is this person wearing eyeglasses?` normalized
  202,599 records with `valid: true`, 13,193 yes / 189,406 no, 10,177
  identities.
- **Annotation-noise finding:** 615 identities have images labeled both Male
  and not Male. CelebA attribute labels are per image and can be noisy; this
  is a dataset property, not a loader bug. The proposal already commits to
  "check labels"; Phase C visual review should include a label sanity check.

Per-attribute counts for the candidate visible attributes (all 40 were
profiled; image counts are out of 202,599):

| Attribute | Yes images | No images | Yes % | Identities with ≥1 yes | Identities with ≥1 no |
| --- | --- | --- | --- | --- | --- |
| Eyeglasses | 13,193 | 189,406 | 6.5 | 3,433 | 9,985 |
| Wearing_Hat | 9,818 | 192,781 | 4.8 | 3,321 | 10,158 |
| Smiling | 97,669 | 104,930 | 48.2 | 9,668 | 9,739 |
| Bangs | 30,709 | 171,890 | 15.2 | 4,968 | 10,095 |
| Mustache | 8,417 | 194,182 | 4.2 | 1,435 | 10,153 |
| Bald | 4,547 | 198,052 | 2.2 | 698 | 10,146 |

### 3.3 Proposed CelebA attributes and questions

Proposed (team approval required):

| Attribute | Question |
| --- | --- |
| Eyeglasses | Is this person wearing eyeglasses? |
| Wearing_Hat | Is this person wearing a hat? |
| Smiling | Is this person smiling? |

Reasons: each is a visible, objective property with thousands of identities on
both sides. Excluded: subjective attributes (Attractive, Oval_Face, Big_Nose,
Young, ...); `Blurry`, which is directly confounded with the Gaussian-blur
corruption; and `Pale_Skin`, which is plausibly confounded with darkness.

### 3.4 Proposed sampling and balance rule

The raw labels are very unbalanced (Eyeglasses 6.5% yes), and the current
loader intentionally does not sample. **Proposed rule (not implemented):**

1. For each selected attribute separately, order candidate images by
   SHA-256 of `seed, attribute, filename`.
2. Take images in that order, keeping at most one image per identity per
   attribute, until reaching an equal number of yes and no images.
3. Ask each sampled image only the one attribute question it was sampled for.
4. Split the sampled records by identity with `m2-split` (identity is already
   the CelebA group).

This needs team approval before implementation, as the handoff requires.

## 4. Trial splits and corruptions (trial: placeholder settings, nothing frozen)

### 4.1 POPE split (adversarial, seed 590, fractions 0.02 / 0.81 / 0.17)

Fractions chosen only to mimic the protocol's ~50 pilot and ~500 test questions
per domain.

| Split | Images | Questions | Yes / No |
| --- | --- | --- | --- |
| pilot | 10 | 60 | 30 / 30 |
| development | 405 | 2430 | 1215 / 1215 |
| test | 85 | 510 | 255 / 255 |

`m2-validate-splits`: `valid: true`, and duplicate questions, question split
conflicts, image leakage, group leakage and checksum leakage are all empty.
Two runs produced byte-identical pilot/development/test files.

### 4.2 CelebA split (Eyeglasses, all images, seed 590, 0.1 / 0.45 / 0.45)

Identities 1,018 / 4,580 / 4,579; `valid: true`, zero identity, image and group
leakage; two runs byte-identical. This only tests the splitter at full scale;
the real CelebA split must use the sampled, balanced records from 3.4.

### 4.3 POPE pilot corruptions (exploratory severities)

Exploratory specs: darkness brightness factor 0.6 / 0.35 / 0.15; Gaussian blur
radius 1.5 / 3 / 6 pixels (on ~640 px COCO images).

- `m2-expand-variants`: 60 pilot questions × 7 conditions = 420 variants;
  inheritance and completeness `valid: true`.
- `scripts/m2_pilot_review.py` rendered all 70 image panels twice:
  **0 pixel mismatches**.
- A visual spot check shows the strongest settings make small objects (people,
  traffic lights) hard to see, which is the intended range for review.
- The review page writes a decisions file in the format
  `m2-validate-answerability` accepts (format checked with a synthetic
  all-answerable file; that is **not** a real review).

**Not done:** the human answerability review itself (360 decisions for this
trial pilot). It must be done by eye, without model outputs, and should be
redone on the final pilot manifest after the team approves the split.

## 5. Decisions needed from the team

1. POPE variant: proposed `adversarial` (2.3).
2. CelebA attributes and questions: proposed Eyeglasses, Wearing_Hat,
   Smiling (3.3).
3. CelebA sampling and balance rule (3.4), including how many images per
   attribute.
4. Split sizes. Proposed POPE: about 10 pilot / 405 development / 85 test
   images (60 / 2430 / 510 questions). The development size is a cost question
   (8 configurations × 7 conditions); the team may want a smaller development
   set.
5. Split seed: placeholder 590; the team should pick and record one.
6. Whether the same blur radius is used for both datasets. CelebA aligned
   images are 178×218, about a quarter of COCO's width, so the same radius blurs
   them much more relative to image size.
7. Whether CelebA split filenames may be committed under the agreement.
8. Who downloads the CelebA images (1.4 GB) for checksum verification and
   CelebA pilot review.

## 6. Commands run

```bash
python -m pip install -e .
python -m pytest -q
mpvs m2-normalize-pope --annotations data/pope/coco_pope_VARIANT.json \
  --image-root data/coco-images --variant VARIANT \
  --output artifacts/m2/profile/pope-VARIANT.jsonl   # random, popular, adversarial
mpvs m2-normalize-celeba --attributes data/celeba/list_attr_celeba.txt \
  --identities data/celeba/identity_CelebA.txt \
  --image-root data/celeba/img_align_celeba \
  --attribute-question 'Eyeglasses=Is this person wearing eyeglasses?' \
  --skip-image-checksums --output artifacts/m2/profile/celeba-eyeglasses.jsonl
mpvs m2-split --input artifacts/m2/profile/pope-adversarial.jsonl \
  --output-dir artifacts/m2/profile/trial-split-a --seed 590 \
  --pilot-fraction 0.02 --development-fraction 0.81 --test-fraction 0.17
mpvs m2-validate-splits artifacts/m2/profile/trial-split-a/{pilot,development,test}.jsonl
mpvs m2-split --input artifacts/m2/profile/celeba-eyeglasses.jsonl \
  --output-dir artifacts/m2/profile/trial-celeba-split-a --seed 590 \
  --pilot-fraction 0.1 --development-fraction 0.45 --test-fraction 0.45
mpvs m2-expand-variants --input artifacts/m2/profile/trial-split-a/pilot.jsonl \
  --corruptions artifacts/m2/profile/trial-pilot-corruptions.json \
  --output artifacts/m2/profile/trial-pilot-variants.jsonl
python scripts/m2_pilot_review.py \
  --variants artifacts/m2/profile/trial-pilot-variants.jsonl \
  --image-root data/coco-images --output-dir artifacts/m2/profile/trial-pilot-review
```

Each split and normalization was run twice to check determinism (the second
runs wrote to sibling `-b` / `rerun/` paths). Attribute and label profiles were
computed with short read-only Python snippets over the source files.

## 7. Limitations

- CelebA images are not downloaded, so CelebA image existence, checksums,
  duplicate images and CelebA corruptions are unchecked.
- No final manifests exist; every split above is a trial.
- No answerability decisions have been made.
- No model was run; nothing here uses model accuracy.
