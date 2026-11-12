# CONTEXT: Project Background

> STATUS: FINAL (version 1.0). The users and goals have been validated through product-brainstorming sessions.

## 1. Project Origin
This project is an implementation of case study A from a Biology course assignment, with the report title "Conceptual Design of a Computer Vision Based Computing System for Classifying Bacterial Cell Shape and Gram Status from Digital Microscopic Images". The report was written by Alfiz Ilham (260411101100105), Computer Engineering Study Program, Universitas Syiah Kuala, 2026.

## 2. The Problem
Identifying bacteria through Gram staining depends on the skill of the reader. Samuel et al. (2016) recorded an error rate of 0.4% to 2.7%, with 24% of the errors coming from readers.

On the education side, students studying the integration of microbiology and Computer Vision often struggle to connect staining and morphology theory with the actual computational process.

## 3. Users
- **Students (primary):** study the integration of microbiology and Computer Vision, and modify the code for assignments.
- **Researchers (secondary):** want to develop a multi-head architecture on other datasets.
- **Microbiology experts (secondary):** potential users of an early diagnostic aid.

Requirement details are in PRD.md sections 4 and 5.

## 4. Goals and Priorities
The project goals are combined: pass the assignment with high quality, prove the concept, build a prototype that can be developed further, and understand the integration of microbiology with Computer Vision.

Priority when a conflict arises: report quality and the rubric come first; the prototype only needs to run well enough for a demo.

## 5. Out of Scope
- Clinical use.
- Species identification (only shape and Gram status).
- A replacement for culture and biochemical tests.

## 6. Terms
| Term | Meaning |
|---------|------|
| Cocci | round-shaped cell |
| Bacilli | rod-shaped cell |
| Spiral | helical or curved cell |
| Gram-positive | thick cell wall with peptidoglycan, appears purple |
| Gram-negative | thin cell wall with an outer membrane, appears pink |
| DIBaS | public dataset of 692 files from 33 species; 669 readable images from 32 species are used after Candida albicans is excluded and three damaged files are discarded |
| Lookup table | mapping of species labels to (shape, Gram) pairs |
| Multi-head | one backbone with several classification heads |

## 7. Main Sources
- Talo, M. (2019). arXiv:1912.08765
- Mai, D.-T., & Ishibashi, K. (2021). Electronics, 10(23), 3005
- Cabeen, M. T., & Jacobs-Wagner, C. (2005). Nature Reviews Microbiology, 3(8), 601-610
- Samuel, L. P., et al. (2016). Journal of Clinical Microbiology, 54(6), 1442-1447
- Zieliński, B., et al. (2017). PLoS ONE, 12(9), e0184554
- The complete list is in the report, References section.

## 8. Decisions Already Agreed
See SPEC.md sections 2 through 9 and PRD.md section 10. Summary: random split per image, macro F1-score as the primary reference, class weighting, per-image classification, and a simple web interface with a sequential preprocessing stage panel.