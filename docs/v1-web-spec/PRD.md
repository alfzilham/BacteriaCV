# PRD: Cell Shape and Gram Status Classification Prototype

> STATUS: FINAL (version 1.0). Product decisions have been agreed through product-brainstorming sessions.

## 1. Summary
A Computer Vision prototype that accepts a single digital microscope image and displays the cell shape, Gram status, confidence score per head, and a visualization of each preprocessing stage in sequence. Runs on a laptop or PC with a consumer GPU.

## 2. The Problem
Manual bacterial classification is time-consuming and prone to reader error. Meanwhile, students studying the integration of microbiology and Computer Vision struggle to connect Gram staining theory with the actual computational process. This prototype bridges the two.

## 3. Product Goals
1. Help students understand the flow from microscope image to prediction.
2. Provide a prototype that can be run and modified.
3. Show that a single CNN backbone can predict shape and Gram status at the same time.
4. Become a foundation that can be developed for other datasets.

Priority when a conflict arises: report quality and the rubric come first; the prototype only needs to run well enough for a demo.

## 4. Users

| Segment | Role | Main need |
|--------|-------|-----------------|
| Students (primary) | Direct user | Understand the flow, modify the code, get results quickly, compare with the true label |
| Researchers (secondary) | Advanced developer | Clear module structure and extension points |
| Microbiology experts (secondary) | Aid user | Results with clear confidence and a warning when uncertain |

### 4.1 Student Requirement Priority Order
1. Understand the flow (highest): visualization of each preprocessing stage.
2. Modify the code: modules structured according to DESIGN.md section 8.
3. Fast results: covered by the demo criteria.
4. Compare with the true label: a development feature, because the demo interface does not include test data.

## 5. User Stories
1. As a student, I want to see the original image, the resize result, normalization, segmentation, and the final result in sequence, so I can match those steps against the theory in the report.
2. As a student, I want to see a specific caption on a stage that failed, so I know where the problem occurred.
3. As a student, I want to upload an image and get the shape and Gram labels with their confidence, so I can use it for assignments.
4. As a student, I want to open and change the code per module, so I can experiment.
5. As a researcher, I want to add a new dataset or architecture through documented extension points, so development does not require changing the whole system.
6. As a microbiology expert, I want to see a warning when confidence is low, so I do not rely on an uncertain result.

## 6. Requirements
Referring to SPEC.md section 4, plus the following UX requirements:

| ID | Requirement | Priority |
|----|-----------|-----------|
| U1 | A numbered panel displays each preprocessing stage in sequence | Required |
| U2 | "next" and "back" buttons to step through the stages | Required |
| U3 | A failed stage is marked with a specific caption and a short reason | Required |
| U4 | A short caption for each stage, matching the theory in the report | Required |
| U5 | A warning when confidence is below 0.6 | Required |
| U6 | Grad-CAM or layer activation visualization | Not included in the first version |
| U7 | Comparing results with the true label from the dataset | Development |

## 7. Success Criteria

### 7.1 Demo (minimum bar, agreed)
The system can accept one image and display the shape label, Gram label, confidence per head, and the preprocessing stage panel.

### 7.2 Report
- Primary reference: macro F1-score per head on the test data, from the real experiment.
- Secondary: accuracy per head.
- The 95% to 99% projection from the literature is a comparison, not a requirement.
- Non-functional: single-image inference on a consumer GPU is measured and reported; no threshold is imposed before measuring.

### 7.3 User Experience
- Students can explain the order of the preprocessing stages after using the prototype, tested through a short session with at least three students (method and results must be documented).

## 8. Out of Scope
According to SPEC.md section 2, plus:
- Visualization of CNN layer activations (U6).
- Comparison with the true label in the interface (U7).

## 9. Risks

| Risk | Impact | Mitigation |
|--------|--------|----------|
| Random split overstates performance | results do not reflect new strains | recorded as a limitation, external testing as development |
| Per-image prediction loses mixed images | incomplete labels | recorded as a limitation |
| Misuse as a diagnosis | clinical risk | warning in the interface, limitation in SPEC |
| Watershed fails on dense images | poor segmentation | the stage is marked failed, the flow is still displayed (U3) |
| The stage panel is too technical for beginners | low comprehension | short caption per stage (U4) |

## 10. Decisions Already Agreed
| Decision | Source |
|-----------|--------|
| Conflict priority: the report comes first | product-brainstorming session |
| Demo bar: one image, labels, confidence, visualization | product-brainstorming session |
| Primary user: students | product-brainstorming session |
| Student requirement order: flow, code, fast results, true label | product-brainstorming session |
| Sequential preprocessing stage display, without Grad-CAM | product-brainstorming session |
| Failed stage displayed with a specific caption | product-brainstorming session |