# P1 Code Demonstration Notebooks

This folder contains a compact demonstration set for the P1 defense. It is intentionally separate from the formal `experiments/phase_*` folders so that the presentation demo does not change the official experiment status.

## Demo order

1. `01_phase04_minimal_baseline.ipynb`  
   Minimal Phase 04 baseline using the Phase 03 no-performance dataset, subject-wise GroupKFold, and conventional ML models.

2. `02_basic_hdc_prototype_demo.ipynb`  
   Basic HDC prototype classifier demo using feature binding, bundling, class prototypes, and cosine-style similarity.

3. `03_hdc_variants_comparison.ipynb`  
   Small HDC variant comparison: Vanilla Prototype HDC, OnlineHD-style update, Multi-centroid HDC, and HDC+OnlineHD hybrid.

## Important boundary

These notebooks are P1 feasibility demonstrations. They are not the final HDC experiment suite. The final research results should still be produced later in the formal Phase 04-10 folders with complete tuning, repeated validation, and full reporting.
