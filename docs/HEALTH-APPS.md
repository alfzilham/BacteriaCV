# Health Apps Declaration — BacteriaCVMobile

Answers for the Play Console health apps declaration, with the reasoning.

> **PLACEHOLDER.** Facts below were checked against the source code and the
> committed evaluation report. Items that need an owner decision are marked
> `PLACEHOLDER`.

## Declaration

**BacteriaCVMobile is not a health app and is not a medical device.**

It is submitted under the standard app category. No health app declaration is
claimed, and no health features are requested.

## Why this app is not a health app

### It does not manage, treat, or diagnose a health condition

The app classifies cell morphology in microscope imagery. It reports two labels
about a picture: an approximate cell-shape category and an approximate Gram
stain reaction. It does not assess a person, a symptom, a diagnosis, a
treatment, or a prognosis.

A health app acts on a user's health state. This app acts on a pixel array.

### It has no regulatory clearance

No clearance was obtained or sought from FDA, EU MDR, or any other regulator.
The app is a research demonstration tool.

Consequently the app is not a medical device under any applicable framework, and
it makes no medical claim.

### The mandatory disclaimer is already in the app

The interface states, in the results area:

> This app is not a medical device and does not diagnose, treat, or prevent any
> condition.

## Disclaimers required in the app

The following four statements must be present in the shipped app. This document
records where each one currently lives, so a reviewer can check the artifact.

### 1. Not a medical device

> This app is not a medical device and does not diagnose, treat, or prevent any
> condition.

**Current status: PLACEHOLDER — not yet present in `app/static/`.** A search of
the interface source finds no occurrence of "medical device". This text must be
added to the app before submission.

### 2. Segmentation groups cells

> Segmentation groups cells, it does not isolate individual cells.

**Current status: present, in different words.** The interface already carries
a segmentation limitation block stating that morphological segmentation separates
bacterial groups but not individual cells, because cells touch each other in
100x Gram imagery, together with the measured numbers. The wording differs from
the required sentence above. If the exact wording is mandatory, align it.

Location: `app/static/index.html`, segmentation limitation block, and the
matching translation keys.

### 3. The spiral class is unpopulated

> The spiral class is unpopulated and is excluded from the metrics.

**Current status: PLACEHOLDER — not present as a user-facing statement.** The
fact is enforced in the code: the model is trained on two cell-shape classes,
the `spiral` label exists only as a marker of an unpopulated class, and the
evaluation report states that it is excluded from the metrics. It is not shown
to the user. This statement must be added to the app before submission.

### 4. Results are an aid

> This result is an aid, not a diagnosis.

**Current status: present.** Shown alongside the result whenever confidence is
low, and in the footer information chip.

Location: `app/static/index.html` and `app/static/i18n.js`, key
`low_confidence_warning`; defined in `bacteriacv/config.py` as
`LOW_CONFIDENCE_WARNING`.

## Model validation status

Stated plainly so a reviewer is not left with the impression that the accuracy
figures are clinical.

| Property | Status |
|----------|--------|
| Validation data | 66 held-out images from a public research dataset of 32 bacterial species |
| Clinical validation | **None. No clinical study was performed.** |
| Population studied | Bacterial strains in laboratory microscopy |
| Intended use | Research demonstration and education |
| Not validated for | Clinical decision-making, patient diagnosis, treatment selection |

Accuracy on a laboratory strain dataset does not transfer to clinical
performance on real patient samples. Different sample preparation, different
imaging hardware, and different patient populations are not represented in the
validation set.

## Submission checklist

- [ ] Add disclaimer 1, the not-a-medical-device sentence, to the app.
- [ ] Add disclaimer 3, the unpopulated spiral class statement, to the app.
- [ ] Decide whether disclaimer 2 must match the required wording exactly or
      whether the existing, more detailed wording is acceptable.
- [ ] Confirm the Google Play developer account status.
- [ ] PLACEHOLDER: confirm the target countries.

## Questions for the owner

1. PLACEHOLDER: which of the four disclaimers must be added before submission,
   and who adds them.
2. PLACEHOLDER: whether the existing segmentation wording is acceptable, or
   whether it must be replaced verbatim.
3. PLACEHOLDER: whether to claim any health category at all. This document
   assumes the answer is no.
