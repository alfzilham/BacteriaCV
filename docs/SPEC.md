# SPEC: Bacterial Cell Shape and Gram Status Classification System

## 1. Objective
A Computer Vision system that predicts bacterial cell shape (cocci, bacilli, spiral) and Gram status (positive, negative) simultaneously from a single digital microscope image, using one ResNet-50 backbone with two classification heads.

## 2. Scope
### Included in the first version
- Image preprocessing: resize, normalization, morphological segmentation, watershed transform, augmentation.
- Training two heads with a frozen backbone.
- A lookup table mapping 32 species to (shape, Gram status) pairs for training labels. Candida albicans is excluded from training.
- Macro F1-score (primary) and accuracy (secondary) evaluation per head.
- A simple web interface for uploading images and viewing results.

### Not included in the first version
- Full backbone fine-tuning (a later development path).
- Per-cell classification (a later development path).
- Generalization testing on new bacterial strains.
- Integration with a clinical laboratory system.

## 3. Data
- Dataset: DIBaS (Digital Images of Bacterial Species), 692 image files, 33 species, original resolution 2048 × 1532 pixels.
- The file count of 692 comes from direct verification against the archives. The figure of 660 cited in the original paper and other write-ups does not match the archive contents: the number of images per species is not uniform, 21 species have 20 files, 10 species have 23, and Veillonella has 22.
- Three files cannot be read by the image library and are already damaged inside the DIBaS archives: Listeria monocytogenes 0023 at 0 bytes, plus Micrococcus spp 0021 and 0023 with broken TIFF structure. The SHA-256 of the files on disk is identical to the archive entries, so the damage originates from the source. The full list with reasons is in `data/raw/unreadable.csv`.
- Trained species: 32 of 33. Candida albicans is excluded because it is a fungus, not a bacterium, and because fungal cells are 5 to 10 microns in size so this class would be easily separated by the backbone and would raise the F1-score without demonstrating bacterial morphology classification ability. The exclusion is recorded in `bacteriacv/datasets/species_map.py`.
- Effective training data: 669 readable images from 32 species.
- Split: random per image, 70% train, 20% validation, 10% test, with five-fold cross-validation.
- Actual split result: 467 train, 136 validation, 66 test.
- Cell shape: only two classes are populated, namely cocci and bacilli. DIBaS contains no spiral-shaped species, so the spiral class is recorded as unpopulated.
- The five folds are distributed only within the train data. Validation data is used for early stopping, test data is used only once at the end.
- Folds are balanced per species rather than globally, so the absolute size of each fold differs (104/96/96/96/75). Stratification is still preserved because every species appears in every fold.
- A random split does not measure generalization to new strains. The test result only represents performance on strains similar to the train data.
- Training labels are derived from species labels through the lookup table.
- Images whose labels are inconsistent with the lookup table are discarded from training.
- Images that cannot be read by the image library are discarded from the index and recorded in `data/raw/unreadable.csv` with their reasons.

## 4. System Functions
| ID | Requirement | Priority |
|----|-----------|-----------|
| F1 | Accept microscope images in common formats (PNG, JPG, TIFF) | Required |
| F2 | Perform preprocessing according to the specification | Required |
| F3 | Produce a cell shape prediction with a confidence score | Required |
| F4 | Produce a Gram status prediction with a confidence score | Required |
| F5 | Display the segmentation result as bacterial area contours, not individual cells | Required |
| F6 | Train the model with class weighting derived from train data | Required |
| F7 | Compute macro F1-score and accuracy per head | Required |
| F8 | Web interface for upload and result display | Required |

## 5. Outputs
- Cell shape label and confidence score.
- Gram status label and confidence score.
- Segmentation result visualization.

## 6. Success Metrics
- Primary reference: macro F1-score per head.
- Secondary: accuracy per head.
- Comparison references: Talo (2019) 99.2%, Mai & Ishibashi (2021) 98.25%. Direct comparison is only valid on the accuracy metric.
- Report projection: a range of 95% to 99%. These figures are not a success requirement; the real experimental result is what gets reported.

## 7. Class Imbalance Handling
- Class weighting in the loss function.
- Weights are computed only from train data, never from validation or test data.

## 8. Limitations and Constraints
1. Per-image prediction yields one dominant label, so images with mixed cell shapes are not fully represented.
2. A random split does not measure generalization to new strains.
3. Performance depends on image quality and consistency of the Gram staining protocol.
4. The system is an early diagnostic aid, not a replacement for culture and biochemical tests.
5. Morphological segmentation succeeds at separating bacterial groups but not individual cells, because cells touch each other in 100x Gram images. Median object elongation is 1.30 to 1.43 for cocci species and 1.64 to 1.76 for bacilli species under the major/minor axis metric, yet the intraspecies range of 1.03 to 4.19 exceeds the between-group difference of 0.068, so shape classification cannot be validated from segmented objects. Area-based metrics (extent) show no separation at all. This evidence is constrained in two directions: object elongation mixes cell shape with cell arrangements such as clusters, chains, and tetrads, which this method cannot separate. Therefore F5 displays bacterial area contours rather than individual cells, and the visualization panel always states this limitation.
6. The test data contains 66 images from 32 species, roughly two images per species. A macro F1 of 1.000 on the test data means not a single error across 66 images, not that the model is perfect on other data. This figure must be read together with that small test set size.
7. There is one pair of images in DIBaS that are the same microscope field with small differences, namely `Enterococcus.faecalis_0017.tif` in the test split and `Enterococcus.faecalis_0018.tif` in the train split. The Pearson correlation is 0.9949 and the correlation after 64 x 64 normalization is 0.9995. The impact was checked: macro F1-score stays at 1.000 when those images are removed, so over 65 images. `build_index` only rejects path duplication and byte-identical duplication, and does not yet detect near-duplicates. Therefore this measurement is disclosed rather than fixed. The correct fix is near-duplicate detection based on correlation at the build index stage, which is out of scope for the first version.

## 9. Acceptance Criteria

### 9.1 Demo criteria (required, minimum bar)
- The system accepts one microscope image in PNG, JPG, or TIFF format.
- The system displays a cell shape label and confidence score.
- The system displays a Gram status label and confidence score.
- The system displays the segmentation result visualization.
- The web interface can run on a laptop or PC with a consumer GPU.

### 9.2 Report criteria (required for the rubric)
- Macro F1-score and accuracy are computed per head on the test data, from the real experiment.
- The reported figures are experimental results, including when lower than the report projection.
- The limitations in Section 8 are listed in every results report.
- There are no claims of generalization to new strains.

### 9.3 Note
The full pipeline over 669 images is a development target, not a demo requirement.