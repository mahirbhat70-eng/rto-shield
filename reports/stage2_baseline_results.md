# Stage 2 Baseline Results

> Default 0.5 threshold is a placeholder. Real operating threshold
> will be chosen in a later stage via the cost matrix.

| Model | Split | Flag Rate | Precision | Recall | F1 | PR-AUC | PR-AUC Provenance | ROC-AUC | Prec @ Rule Recall |
|-------|-------|-----------|-----------|--------|----|--------|-------------------|---------|-------------------|
| Rule Baseline | val | 0.3628 | 0.2638 | 0.4751 | 0.3393 | 0.2311 | implied (closed-form, binary scorer) | N/A | N/A |
| Rule Baseline | test | 0.3557 | 0.2717 | 0.4887 | 0.3493 | 0.2339 | implied (closed-form, binary scorer) | N/A | N/A |
| Logistic Regression | val | 0.0100 | 0.5497 | 0.0272 | 0.0519 | 0.3406 | sklearn | 0.6756 | 0.3299 |
| Logistic Regression | test | 0.0097 | 0.5793 | 0.0283 | 0.0541 | 0.3434 | sklearn | 0.6854 | 0.3408 |
