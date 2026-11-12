# Week 8 — Unsupervised Learning & Interpretability

## Deliverables
- `clustering_pca.py`: preprocessing, K-Means candidates (k=2..6), silhouette/elbow metrics, PCA projection and segment profiles.
- `interpretability.py`: holdout permutation importance, aggregated XGBoost built-in importance, SHAP summary plot for the frozen Week 7 model.
- `INSIGHTS.md`: business-language cluster profiles, PCA interpretation, importance comparison and SHAP explanation.
- `outputs/`: generated plots, CSVs, metrics, and the copied Week 7 fitted pipeline.
- `data/raw/Telco-Customer-Churn.csv`: dataset used by both scripts.

## Run
Install dependencies and execute:
```bash
pip install -r requirements.txt
python clustering_pca.py
python interpretability.py
```

The scripts expect to be run from this directory (or via their own file location). `interpretability.py` loads `outputs/final_churn_pipeline.joblib`; it does not retrain that model.

## Leakage and methodology notes
- `Churn` and `customerID` are excluded from K-Means/PCA inputs. Churn is only used afterward to profile segments.
- Permutation importance uses the untouched 25% stratified Week 7 holdout split (`random_state=42`) and the complete frozen pipeline.
- SHAP uses transformed holdout features and the existing XGBoost classifier.
