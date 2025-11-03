# MODEL CARD – ML2 Fraud Classifier

**Intended Use**: Academic competition (Dacon ML2), binary fraud detection.  
**Data**: Dacon-provided train/test CSVs (private).  
**Algorithm**: CatBoostClassifier (k-fold ensemble).

## Factors
- **Inputs**: claim metadata, driver/vehicle info, engineered ratios.
- **Metrics**: macro F1 on public LB; OOF macro F1 for internal.
- **Limitations**: class imbalance, potential dataset shift.

## Ethical Considerations
- False positives may flag legitimate claims; threshold is tuned using LB feedback—use caution for real-world deployment.
