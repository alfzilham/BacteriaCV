# Play Store Data Safety — BacteriaCVMobile

Answers for every question in the Play Console Data Safety form, with the
reason behind each answer.

> **PLACEHOLDER.** Every factual claim below has been checked against the
> source code and the committed evaluation report. Answers that depend on facts
> only the owner knows are marked `PLACEHOLDER` and must be confirmed before
> the form is submitted.

Package: `alfzilham.ilham.bacteriacv_mobile`

## Short answer

The app collects nothing, shares nothing, and contains no tracking. Every
question in the form is answered **No** or **No data**.

## Section 1: Data collection and sharing

### Does your app collect or share any of the required user data types?

**No.**

The app has no backend. Inference runs on the device using a model bundled
inside the app. There is no account system, no sign-in, no user identifier, and
no network request in the inference path.

### Is all of the user data collected by your app encrypted in transit?

**Not applicable.** No data is transmitted, so there is nothing in transit.

### Do you provide a way for users to request that their data is deleted?

**Not applicable.** No data is collected or stored beyond a single image held
in memory for the duration of one analysis, which is discarded when the app is
closed.

### Data types collected or shared

| Data type | Collected | Shared | Purpose |
|-----------|-----------|--------|---------|
| Location | No | No | — |
| Personal info | No | No | — |
| Financial info | No | No | — |
| Health and fitness | No | No | — |
| Messages | No | No | — |
| Photos and videos | No | No | — |
| Audio files | No | No | — |
| Files and docs | No | No | — |
| Calendar | No | No | — |
| Contacts | No | No | — |
| App activity | No | No | — |
| Web browsing | No | No | — |
| App info and performance | No | No | — |
| Device or other IDs | No | No | — |

### Justification a reviewer may question

**"Photos and videos — the app analyses images. Why is that No?"**

The app reads an image the user explicitly selects through the system image
picker, processes it in memory, and discards it. That is local processing of a
file the user already has, not collection. Data Safety asks about data leaving
the device or being retained. Neither happens here.

**"App activity — the app shows results. Why is that No?"**

Showing a result to the user who requested it is not collection. No result is
recorded, stored, logged, or transmitted anywhere.

## Section 2: Data security

### Is all of the user data collected by your app encrypted in transit?

**Not applicable.** Nothing is transmitted.

### Do you provide a way for users to request that their data be deleted?

**Not applicable.** Nothing is stored.

### Does your app collect or share data that is not covered by the required
user data types?

**No.** The app has no telemetry, no logging of user content, and no
third-party SDK of any kind.

## Section 3: Data used for tracking

### Does your app collect or share any data for tracking?

**No.**

No advertising, no analytics, no attribution, no crash reporting, no device
fingerprinting, and no advertising identifier of any kind is used. No third-party
SDK is embedded.

## Section 4: Purpose of data collection and sharing

Not applicable, because nothing is collected or shared.

## Section 5: Additional declarations

### Does your app contain or transmit malware?

**No.**

### Does your app respect the rights of users regarding their data?

**Yes.** There is no data to misuse.

### Does your app collect or share data from children?

**No.** The app collects no data from anyone.

### Is your app a health app?

**No.** See `docs/HEALTH-APPS.md` for the full reasoning. In short, the app
presents research findings about cell morphology, it does not manage, treat, or
diagnose a health condition, and it has no regulatory clearance.

## Section 6: Downloadable code

### Does your app download, install, or execute code at runtime?

**No.**

The model weights and all executable code are bundled inside the released APK.
The app contains no dynamic code loading, no remote script evaluation, and no
runtime download of executable code.

## Risks declared honestly

The following are disclosed because a reviewer evaluating a health-adjacent
tool should know them, and because they are stated in the app itself.

1. **The app processes microscope images on the device.** It runs a
   convolutional model over an image the user selects. Nothing leaves the
   device.

2. **This is not a biomedical device and not a diagnostic instrument.** No
   regulatory clearance of any kind was obtained or sought. It is a research
   demonstration tool.

3. **Model validation was on 66 internal test images, not clinical validation.**
   The reported figures come from a held-out split of a public research
   dataset of 32 bacterial species. No clinical study was performed. Accuracy on
   a laboratory dataset does not translate into clinical performance on real
   patient samples.

4. **The spiral class is unpopulated and excluded from the metrics.** The
   training dataset contains no spiral-shaped species, so the model is trained
   on two cell-shape classes. The third class exists in the label enum only as a
   marker and is excluded from every reported metric. The app displays only
   `cocci` and `bacilli`.

5. **Segmentation groups cells; it does not isolate individual cells.** Cells
   touch each other in 100x Gram-stained imagery, so the segmentation stage
   separates bacterial groups rather than individual cells. Measured elongation
   ranges for the detected objects overlap within species more than they differ
   between cell-shape groups, so cell shape is not validated from the
   segmentation output. The app states this limitation in the interface.

6. **Results are an aid, not a diagnosis.** The interface shows confidence
   values alongside every result and states: "This result is an aid, not a
   diagnosis."

## Before submitting this form

Confirm each of the following yourself. None of them can be verified from the
source code.

- PLACEHOLDER: confirm the released APK is built from this repository at the
  expected commit, so that the answers above describe the artifact you actually
  ship.
- PLACEHOLDER: confirm no analytics or crash-reporting SDK is added by the
  build pipeline, gradle configuration, or any release script outside this
  repository.
- PLACEHOLDER: confirm the target audience rating and content rating answers.
