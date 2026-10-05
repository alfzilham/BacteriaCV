# ARCHITECTURE: Computer Vision Based Bacterial Classification System

## 1. Overview
A five-stage linear pipeline. The training stage and the inference stage share the same preprocessing module and backbone, but the training stage additionally uses a lookup table and class weighting.

```
[Microscope image]
      |
      v
[1. Preprocessing] --- resize, normalization, morphological segmentation, watershed, augmentation (train only)
      |
      v
[2. ResNet-50 Backbone] --- frozen, ImageNet pretrained
      |
      v
[3. 2048-d feature vector]
      |          \
      v           v
[4a. Head A]   [4b. Head B]
  softmax, 2     sigmoid, 2
  cell shape     Gram status
      |          |
      v          v
[5. Output: label + confidence + segmentation visualization]
```

At the training stage, the target label is taken from the lookup table [L] before entering the loss.

## 2. Components

### C1. Preprocessing Module
- Input: RGB image of 2048 × 1532 pixels.
- Steps: resize to 224 × 224; intensity normalization; morphological segmentation (erosion, dilation, watershed); augmentation only at the train stage (rotation, flip, zoom, shift).
- Output: 3 × 224 × 224 tensor.
- Dependencies: OpenCV, NumPy, scikit-image.

### C2. Backbone
- ResNet-50 from torchvision, ImageNet weights.
- All parameters are frozen in the first version.
- Output: 2048-dimensional vector after global average pooling.

### C3. Classification Head
- Head A: Linear(2048, 2), softmax activation, weighted cross-entropy loss.
  Two classes: cocci and bacilli. The spiral shape exists in the shape enum as
  a marker for an unpopulated class, because DIBaS contains no spiral species.
- Head B: Linear(2048, 2), sigmoid activation, weighted binary cross-entropy loss.
  **Only the second output column of Head B enters the BCE loss.** The first output
  column is never regularized by the loss; it is only affected by weight
  decay, so its value carries no learned information.
  Consequently, the Head B positive class probability is
  `sigmoid(logit of the second column)`, **not** softmax over two columns. Using
  softmax would combine the trained logit with the untrained logit, which adds
  noise and lowers calibration instead of improving it. This is verified: using
  softmax lowers validation accuracy from 1.0000 to 0.9926 and increases
  low-confidence images on the test data from 0 to 1. The classification threshold uses
  `sigmoid(logit) > 0.5`.
  The implementation is in `train.py` line 599 for the Head B BCE loss and
  `train.py` line 363 for the classification threshold.
- Class weights are computed from frequency on the train data.

### C4. Lookup Table [L]
- Maps 32 DIBaS species labels to (shape, Gram status) pairs.
  Candida albicans is not included because it is excluded from training.
- Used only at the training stage.
- Images whose species label is absent from the table are discarded.
- Species name mapping is stored in `species_map.py` rather than guessed from filenames.
  Six DIBaS archive names contain misspellings, and two different archives
  (`Lactobacillus.jehnsenii` and `Lactobacillus.johnsonii`) refer to the same
  taxonomic species, so both are distinguished through an internal ID.

### C5. Evaluation Module
- Computes macro F1-score and accuracy per head from the confusion matrix.
- Splits data 70:20:10 randomly per image, with five-fold cross-validation.
- The five folds are distributed only within the 70% train data. Validation data (20%) is used for
  early stopping and checkpoint selection, test data (10%) is used only once at the end.
  If the folds included validation data, then validation data would become training data in
  some folds and optimistic bias would appear in the validation metrics.
- Folds are balanced per species rather than globally. A species with 20 train
  images contributes 3/3/3/3/2 to the five folds, while a species with 16 train
  images contributes 4/3/3/3/3. As a result the absolute fold sizes differ, for
  example 104/96/96/96/75. What is preserved is stratification: every species still
  appears in every fold, so cross-fold comparison stays unbiased.

### C6. Output Module
- Produces a label and confidence score per head.
- Produces a segmentation visualization as bacterial area contours over the original image, not individual cell contours. The reason is in SPEC section 8 item 5: segmentation separates bacterial groups, not cells, and object elongation mixes with cell arrangement.

### C7. Web Interface
- Upload an image, run inference, display the result and visualization.
- Backend: FastAPI. Frontend: a single HTML page.

## 3. Data Flow

### 3.1 Training
1. Load images and species labels from DIBaS.
2. Apply lookup table [L] to obtain shape and Gram labels.
3. Discard unmapped images.
4. Preprocess with augmentation.
5. One backbone forward pass per image, save the feature vector to cache.
6. Train both heads on the frozen features. Compute weighted loss, update only the head parameters.
7. Evaluate on validation data, save the best checkpoint.
8. Evaluate test data once at the end, after restoring the best weights.

Steps 5 and 6 are separated because the backbone is frozen. A forward pass at 224 x 224 takes roughly 76 ms per image; running it inside every epoch makes 200 epochs take tens of minutes without changing the feature vectors. The cache key is invalidated by each image's filename, size, and modification time, so changing one source image is enough to rebuild the cache.

### 3.2 Inference
1. Receive an image from the interface.
2. Preprocess without augmentation.
3. Forward pass, then predict both heads.
4. Send the result to the interface.
The lookup table is not used at this stage.

## 4. Directory Structure
```
BacteriaCV/
├── bacteriacv/
│   ├── paths.py                      # folder locations, no absolute paths
│   ├── preprocess.py                 # C1
│   ├── model.py                      # C2, C3
│   ├── label_map.py                  # C4
│   ├── train.py
│   ├── evaluate.py                   # C5
│   ├── infer.py
│   ├── visualize.py                  # C6
│   └── datasets/
│       ├── species_map.py            # 33 DIBaS species to canonical names
│       ├── download.py               # download archives + SHA-256 manifest
│       ├── extract.py                # extraction per species + verification
│       └── build_index.py            # index.csv + leakage check
├── app/
│   ├── main.py                       # C7 backend
│   └── static/
│       ├── index.html                # C7 frontend, markup only
│       ├── main.css                  # styles, industrial brutalist
│       └── main.js                   # interaction and API calls
├── data/
│   ├── raw/
│   │   ├── zips/                     # original ZIP archives
│   │   ├── images/                   # TIFF images per species
│   │   └── zips_manifest.csv         # SHA-256 of each archive
│   ├── unreadable.csv                # damaged images, already damaged in the source archives
│   ├── index.csv                     # path, species, split, fold
│   └── features/                     # backbone feature cache, derived artifacts
├── checkpoints/                      # heads.pt, training_report.json, evaluation.json
├── docs/                             # SPEC, ARCHITECTURE, DESIGN, PRD, CONTEXT
├── scripts/                          # setup_env.ps1, check_env.ps1
├── tests/
└── AGENT.md
```

The structure uses a `bacteriacv` package with the `bacteriacv.` entry point. The reason for using a package rather than a flat `src` folder: the dataset modules depend on each other (`species_map` is used by `download`, `extract`, and `build_index`), and relative imports between packages are tidier than piling modules into one folder.

## 5. Architecture Decisions
| ID | Decision | Reason |
|----|-----------|--------|
| A1 | One backbone, two heads | Matches the Chapter III and IV.4 reports; saves compute |
| A2 | Frozen backbone | Fits on a consumer GPU; reduces overfitting risk |
| A3 | Per-image classification | Matches DIBaS labels; no segmentation needed at inference |
| A4 | Watershed as preprocessing | Reduces overlapping cells without changing labels |
| A5 | PyTorch | Modifying the heads and the dual loss is more explicit |
| A6 | Class weighting, not oversampling | Does not duplicate images; prevents spurious validation results |
| A7 | Folds only within train data | Validation data must stay free of training data so early stopping is not biased |
| A8 | Species names kept in one table | Six DIBaS archive names are misspelled, two archives refer to the same taxonomic species |
| A9 | SHA-256 of each archive recorded | The source server uses an expired TLS certificate, so integrity is proven through hashes |

## 6. Extension Points
- Full fine-tuning: unfreeze the backbone in `model.py`.
- Per-cell classification: add an object cropping stage after watershed, then run inference per object.
- Other datasets: replace `label_map.py` and provide a new loader.

## 7. Error Handling
- Image unreadable: return an error message, do not stop the service.
- Watershed finds no objects: continue with the original image and record a warning.
- Low confidence: display the result with an uncertainty marker rather than hiding it.
- Download failure: retry three times, then raise the error. Temporary downloads are cleaned up
  without masking the original error.
- Image outside the project root: reject explicitly, never write an absolute path into `index.csv`.
- Species with fewer than 15 files: extraction verification failed.

## 8. Data Integrity Notes

The original DIBaS server, `doctoral.matinf.uj.edu.pl`, uses an already expired TLS certificate.
Downloading is only possible without certificate verification. To close that gap, each
archive is hashed with SHA-256 and the result is recorded in `data/raw/zips_manifest.csv`. Those hashes are what
serve as provenance evidence in the report, not the server certificate.

The file count was verified directly against the archives, yielding 692 files. The figure of 660 in the original
paper is an estimate that does not match the archive contents.

Candida albicans is excluded from training. Three damaged files are also discarded, so the
effective data is 669 readable images from 32 species. There are two reasons for the exclusion: fungi are not bacteria,
and fungal cell size makes that class easily separable so the F1-score does not
represent the measured ability.

The three discarded files were already damaged inside the DIBaS archives. The SHA-256 of the files on disk is identical
to the archive entries, so the damage comes from the source, not from the extraction process. The full
list with reasons is recorded in `data/raw/unreadable.csv`.