# DIBaS Segmentation Success Criteria

Established before the experiment, 4 October 2026, by the project owner.

## Background

Morphological segmentation is architecture decision A4 and the basis for output F5
in SPEC section 4. Its parameters have not been validated, so they must not be used
as a claim of system capability before passing measurement.

## Parameters Tested

- `min_distance`: 8, 12, 16 (starting from 12)
- Scale: original resolution 2048 x 1532
- Erosion and dilation disk: `disk(5)`, footprint 11 x 11
- `min_object_area`: 300
- Threshold: Otsu on an already blurred grayscale image

## CLASSIFICATION USED

Using the audit-validated `label_map.LOOKUP`, not a new determination:

- Cocci: 9 species
- Bacilli: 23 species

## DIRECT SUCCESS CRITERIA

At `min_distance = 12`, all three of the following conditions must be met:

1. The median elongation across all cocci species must not exceed 1.45
2. The median elongation across all bacilli species must not fall below 1.60
3. The distance between the cocci median and the bacilli median is at least 0.15

Elongation is computed as `axis_major_length / axis_minor_length` from
`skimage.measure.regionprops` on each watershed object. The statistic used is the
median, not the mean, because the distribution is not normal.

## FAILURE CRITERIA

The experiment is declared failed if, after testing `min_distance` at 8, 12, and
16, there is a `min_distance` that meets the three conditions above.

## TIME LIMIT

30 minutes. If that runs out without meeting the criteria, the experiment is stopped.

## FOLLOW-UP DECISION

### If it succeeds

Option (c) proceeds. Segmentation runs at the original resolution for the mask and
visualization, and the 224 resize is only for the model tensor. Constants in `config.py`
are updated to the chosen values.

### If it fails

Segmentation is NOT discarded. The fallback is option (c) fixed with accurate labels:

1. The term "cell contour" in ARCHITECTURE C6 and F5 in SPEC section 4 is changed
   to "bacterial area contour", because what is segmented is bacterial groups, not
   individual cells. This is a separate amendment.
2. SPEC section 8 records the limitation with measured figures: segmentation
   separates bacterial groups but not individual cells, because cells
   touch each other in 100x Gram images.
3. The per-species elongation table goes into the report.

The reason the fallback was chosen rather than discarding segmentation: F5 has Required
status in SPEC section 4, and ARCHITECTURE A3 states segmentation is not needed at
inference so segmentation does not affect model accuracy. Imperfect segmentation with
correct labels still has value; what is damaging is wrong labels.

## NOTE ON STAPHYLOCOCCUS

The grape-like clusters in Staphylococcus are a real morphological unit at
100x magnification, not a segmentation artifact. Larger objects in cluster-forming cocci species do not automatically mean segmentation failed.