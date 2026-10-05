# DESIGN: Interface Specification and Visual Design

## 1. Scope
This document governs the appearance of the simple web interface (C7 in ARCHITECTURE.md). The technical parts of the Python modules (preprocess, model, label_map, train, evaluate, infer, visualize) remain listed in section 8.

## 2. Colour Palette

| Role | Value | Description |
|-------|-------|-------------|
| Primary | `#050505` | Main text, 2px solid border, section headings, structural accents |
| Secondary | `#333333` | Secondary text, metadata labels |
| Surface | `#FFFFFF` | Result card background, main content panel |
| Sub-surface | `#EAE8E3` | Toggle control background, data preview, scrollbar track |
| Background | `#F4F4F0` | Main page background (matte unbleached paper) |
| Accent | `#D31515` | Main process button, warning accent, active focus, scrollbar thumb on hover |
| High Level | `#005A36` | High confidence level text, 7.57:1 on paper background |
| Medium Level | `#8A5000` | Medium confidence level text, 5.90:1 on paper background |
| Low Level | `#B30000` | Low confidence level text, 6.53:1 on paper background |
| Muted | `#555555` | Faded secondary text, 6.76:1 on paper background |
| Warning Background | `#FFF9E6` | Low confidence warning box background |
| Error Background | `#FFF0F0` | Error message box background |

Main text uses the dark colour (#050505). Text on the `#D31515` background must be white (`#ffffff`).

### 2.1 Colour Accessibility
- Text on `#D31515` must be white (`#ffffff`).
- Status information must not depend on colour alone. Each confidence level must be accompanied by text (High, Medium, Low).
- Text contrast against the background must meet WCAG 2.1 AA (a ratio of at least 4.5:1). Ratios are tested at the implementation stage.
- The scrollbar is a UI component, not text. What applies is WCAG 1.4.11 Non-text Contrast with a 3:1 threshold, not WCAG 1.4.3 which is 4.5:1 and only applies to text. The default thumb `#050505` on the `#EAE8E3` track reaches 16.65:1 and the hover thumb `#D31515` reaches 4.41:1. Both pass the 3:1 threshold. That hover ratio of 4.41:1 is ACCEPTABLE and not a failure: it does not need to be fixed, and it is not required to be raised to 4.5:1.
- The scrollbar track cannot be changed to `#F4F4F0` because that is the page background colour, so the track would become invisible. The track stays `#EAE8E3`.

## 3. Typography
- Structural font: system sans-serif (-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif).
- Telemetry/code font: monospaced (ui-monospace, Consolas, "Courier New", monospace).
- Page title: 28 px, bold (weight 800), sans, uppercase.
- Section heading: 18 px, bold (weight 800), sans, uppercase.
- Subheading / category: 14 px, bold (weight 700), mono, uppercase.
- Body text: 16 px, normal (weight 400), sans, line-height 1.5.
- Caption text: 13 px, medium (weight 500), mono, uppercase.
- Code & numeric metrics: 13 px, semibold (weight 600), mono, tabular-nums.

## 4. Layout

The layout follows the agreed skeleton, a single page with three main areas.

```
+--------------------------------------------------------------+
| [Logo]   Menu       Menu     [Secondary button] [Main button]|  Header
+--------------------------------------------------------------+
| +-------------------------+  +--------------------------+     |
| | Image upload card       |  | Result summary card      |     |
| | [upload area]           |  | [label + confidence]     |     |  Top row
| +-------------------------+  +--------------------------+     |
|                                                             |
| | Section heading          | +----------------------------+  |
| | Caption text             | |                            |  |
| | [Process button] [Choice]| |  Segmentation visualization|  |  Result area
| | Text result summary      | |                            |  |
| |                         | +----------------------------+  |
| |                         | Control panel: view options    |
+--------------------------------------------------------------+
| [Chip] [Chip] [Chip] [Chip] [Chip]                           |  Footer: captions
+--------------------------------------------------------------+
```

### 4.1 Header
- Logo or system name on the left.
- Navigation menu in the middle: Upload, History, About.
- Secondary button and main button on the right.
- The logo comes from `app/static/logo.svg`, the favicon from `app/static/favicon.svg`. Both use the section 2 palette and do not use colours from any reference image. The favicon has no frame because at 16 pixels a black frame turns into a blob.
- The header logo is decorative and uses an empty `alt`, because the system name is already written as text next to it. Otherwise a screen reader would read it twice.
- The language switcher sits in the header next to the existing navigation buttons: two small buttons, EN and ID. It follows the same brutalist rules as the rest of the interface, with no border-radius and no shadow. The selected language is the active one. The choice is stored in `localStorage` under the key `bacteriacv.lang`, and English is the default when that key is absent.

### 4.2 Top Row
- Left: image upload card with a drag-and-drop area and a file picker button.
- Right: result summary card with the label and confidence for Head A and Head B.

### 4.3 Main Area
- Left: section heading, caption text, process button, and brief result text.
- Right: a single segmentation visualization panel of fixed size, and a row of stage toggle buttons below it for switching among the five preprocessing stages, from the original image through the segmentation stage. It stays a single panel, not five separate panels.

### 4.4 Footer
- Five information chips: supported formats, maximum size, model version, data source, and disclaimer.

## 5. Components

| Component | States | Behaviour |
|----------|---------|----------|
| Upload area | normal, hover, dragover, error | accepts PNG, JPG, TIFF up to 20 MB |
| Process button | normal, disabled (no image yet), loading | inactive before an image is selected |
| Result card | empty, loading, filled | shows the label and confidence per head |
| Confidence label | high (>= 0.8), medium (0.6 to 0.79), low (< 0.6) | low shows a warning |
| Visualization panel | empty, loading, filled | contour overlay image, toggle to hide |
| Warning | shown when confidence is low | text: "This result is an aid, not a diagnosis." |
| Footer chip | static | information, not interactive |

## 6. Interaction
1. The user uploads an image through the upload area or by drag-and-drop.
2. The image preview appears in the upload card.
3. The user presses the process button. The button switches to a loading state.
4. The result card and the visualization panel are filled at the same time.
5. If confidence is low, a warning appears above the result card.
6. The user can hide the contours with the toggle.
7. The user can switch the interface language at any time. All static text, result labels, notes, stage titles, error messages, and the filename placeholder change. The current classification result is kept and redrawn in the new language.

## 7. Error Messages
| Condition | Message |
|---------|-------|
| Unsupported format | "File format not supported. Use PNG, JPG, or TIFF." |
| Size exceeds 20 MB | "File size exceeds 20 MB." |
| Image unreadable | "Image cannot be read. Try uploading another file." |
| Segmentation failed | "Segmentation did not succeed. The classification result is still displayed without visualization." |

## 8. Module Technical Parts
The Python modules and their functional contracts stay as in the previous version of DESIGN.md:
- `preprocess.py`: `load_image`, `segment_cells`, `preprocess`.
- `model.py`: `BacteriaNet` with two heads.
- `label_map.py`: `LOOKUP`, `to_targets`, `class_weights`.
- `train.py`: weighted loss, Adam optimizer, early stopping on validation macro F1.
- `evaluate.py`: `evaluate` returns macro F1 and accuracy per head.
- `infer.py`: `predict` returns label, confidence, and overlay.
- `app/main.py`: `POST /predict`, `GET /`, 20 MB size limit.

## 9. Interface Testing
| Type | Coverage |
|-------|---------|
| Visual | conformance with the section 4 layout and the section 2 palette |
| States | every state in section 5 tested separately |
| Accessibility | text contrast and colour limits in section 2.1 |
| Upload | valid file, unsupported file, and file over the limit |
| Language | both languages render every string, and switching preserves the classification result |