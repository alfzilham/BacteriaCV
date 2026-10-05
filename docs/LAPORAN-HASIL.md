# Experiment Results Report

All figures in this document come from a real experiment on the DIBaS dataset that has
already been extracted. There are no projected figures. The 95% to 99% projection in
SPEC section 6 is not a success requirement and is not used here.

Execution date: 4 October 2026.

## 1. Configuration Used

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Backbone | ResNet-50, `IMAGENET1K_V2`, frozen | `config.BACKBONE_NAME`, `BACKBONE_WEIGHTS` |
| Feature dim | 2048 | `config.FEATURE_DIM` |
| Head A | `Linear(2048, 2)` softmax, classes `cocci` and `bacilli` | `config.N_SHAPE_CLASSES` |
| Head B | `Linear(2048, 2)` sigmoid, classes `positive` and `negative` | `config.N_GRAM_CLASSES` |
| Augmentation | 4 variants per train image (1 original + 3 rotate/scale/shift/flip) | `config.AUGMENT_VARIANTS` |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 | `config.LEARNING_RATE`, `WEIGHT_DECAY` |
| Batch size | 32 | `config.BATCH_SIZE` |
| Epoch limit | 200 | `config.MAX_EPOCHS` |
| Early stopping | patience 20, min delta 1e-4 | `config.EARLY_STOPPING_PATIENCE`, `MIN_DELTA` |
| Training seed | 1337 | `config.TRAIN_SEED` |
| Index seed | 20260203 | `config.INDEX_SEED` |
| Segmentation resolution | 2048 x 1532 (original) | `config.SEGMENT_SCALE` |
| Upload limit | 20 MB | `config.MAX_UPLOAD_BYTES` |

## 2. Deviations from the Plan and Their Reasons

| What changed | Plan | Reality | Reason |
|----------------|--------|-----------|--------|
| Feature extraction | full `preprocess` | `preprocess_tensor` without segmentation | Segmentation at the original resolution takes 2.5 seconds per image. 2070 feature rows would take 86 minutes without improving any model metric, because ARCHITECTURE section 3 states segmentation is not needed for inference. |
| Visualization panel | Five panels | One panel with a five-stage toggle | DESIGN section 3 rank 3 beats CONTEXT rank 4. Five panels are sent to the browser and displayed one at a time. |
| Object area filtering | `regionprops` per region | `np.unique` plus `np.bincount` | The loop version compares the whole labelled image for each region, 3.76 seconds per image. The resulting mask is identical, only faster. |
| Species label | `strain` | species | DIBaS has no strains, it has species. |
| Number of epochs | 200 | 35, stopped by early stopping | Patience 20, best epoch 15. Running 200 epochs does not improve validation. |

## 3. Data

- 692 TIFF files archived, 689 readable, 669 used after
  `Candida albicans` was excluded.
- Split: 467 train, 136 validation, 66 test.
- After augmentation: 1868 train rows, 136 validation rows, 66 test rows.
- Train distribution after augmentation: `cocci` 576, `bacilli` 1494;
  `positive` 1471, `negative` 599.
- Three images discarded: `Listeria 0023` 0 bytes, `Micrococcus 0021` and `0023`
  broken TIFF. All three were proven already damaged in the source archive through SHA-256.

## 4. Inference Results

| Metric | Head A (shape) | Head B (Gram) |
|--------|-----------------|----------------|
| Validation macro F1, best epoch | 0.9817 | 1.0000 |
| Test macro F1 | 1.0000 | 1.0000 |
| Test accuracy | 1.0000 | 1.0000 |
| Test precision per class | 1.0000 both classes | 1.0000 both classes |
| Test recall per class | 1.0000 both classes | 1.0000 both classes |
| Test support | `cocci` 19, `bacilli` 47 | `positive` 47, `negative` 19 |
| Mean test confidence | 0.9812 | 0.9671 |

Checkpoint: `checkpoints/heads.pt`. Epochs run: 35, best epoch 15,
epoch 1 shape F1 0.9554 and Gram F1 0.8921.

The test data contains 66 images from 32 species, roughly two images per species. A macro
F1 of 1.0000 means not a single error across 66 images. That does not
mean the model is perfect on other data. The margin between images is indeed narrow:
shape confidence 0.9812 and Gram confidence 0.9671, both above 0.8.
Errors on images outside the test data remain very possible.

Gram confidence is computed from `sigmoid(logit)` on the second output column.
Head B was trained as a single-logit binary classifier with `BCEWithLogitsLoss`,
so `sigmoid(logit)` is indeed the positive class probability and the 0.5
threshold applies directly. The first output column never enters the loss. Evidence:
changing that column to 99.999 does not change the loss value at all,
so that column only experiences weight decay and carries no information.
Therefore softmax over the two output columns must not be used for Head B.
Measurement on the 136 validation images actually shows that softmax lowers
accuracy from 1.0000 to 0.9926 and increases low-confidence images from
one to two.

## 5. Segmentation Limitations

Morphological segmentation succeeds at separating bacterial groups but not individual
cells, because cells touch each other in 100x Gram images. Median object elongation
for cocci species is 1.30 to 1.43 and for bacilli species 1.64 to
1.76 under the major/minor axis metric, yet the intraspecies range of 1.03 to
4.19 exceeds the between-group difference of 0.068, so shape classification cannot be
validated from segmented objects. Area-based metrics (extent) show
no separation at all.

This evidence is constrained in two directions. Object elongation mixes
cell shape with cell arrangement in the form of clusters, chains, and tetrads,
which this method cannot separate. Consequently the visualization panel shows bacterial area
contours rather than individual cells, and always writes this limitation into the
panel image itself.

The segmentation constants were not changed after this experiment. `SEGMENT_MIN_PEAK_DISTANCE`
stays at 15. Changing it to widen the gap between groups would chase noise,
because the intraspecies range is already wider than the between-group difference. If the
constants are ever reviewed, the criterion is stability: pick the value that produces
the smallest intraspecies range so the contours on the panel do not keep changing between
images.

## 6. Speed

| Stage | Time |
|-------|-------|
| Read 2048 x 1532 TIFF | 69 ms |
| Resize and normalize | 25 ms |
| Original resolution segmentation | 2,502 ms |
| ResNet-50 forward pass 224 x 224 | 76 ms |
| Extracting 2070 feature rows, once | about 12 minutes |
| 35 epoch training loop on frozen features | under 60 seconds |
| One complete web request | 3.81 seconds |

Without feature caching, 200 epochs would run the backbone 414,000 times. With
cache, the backbone runs 2070 times and the rest is only two Linear layers.

## 7. pytest Output

```
343 passed, 1 warning in 146.48s
```

The only warning is `StarletteDeprecationWarning` from
`fastapi.testclient` suggesting `httpx2`. No tests failed,
were skipped, or were conditionally skipped.

## 8. Conclusion

A macro F1 of 1.0000 on the test data for both heads was achieved on this dataset, with
the caveat of the small test set size and the moderate mean confidence. What remains
unvalidated is shape classification from segmented objects, and that claim cannot be closed by swapping segmentation parameters.