# Week 8 — Customer Segmentation & Model Interpretability

## Scope and method

This analysis uses the Telco Customer Churn dataset (7,043 unique customer records). `customerID` was excluded. **`Churn` was excluded from the K-Means and PCA input matrices** and was joined back only after clustering to measure each segment's observed churn rate.

Numerical columns were median-imputed and standardized. Categorical columns were mode-imputed and one-hot encoded. K-Means was tested for k = 2, 3, 4, 5, and 6. The chosen k was the candidate with the highest silhouette score. The same transformed feature matrix was projected to two principal components for visualization.

## Choosing the number of clusters

| k | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 68,651.1 | 0.2353 |\n| 3 | 55,588.4 | 0.2383 |\n| 4 | 50,815.3 | 0.2304 |\n| 5 | 47,872.9 | 0.1960 |\n| 6 | 45,698.3 | 0.1854 |\n
The selected solution is **k = 3**, because it had the highest tested silhouette score (0.2383). Inertia decreases as k increases, as expected, but the silhouette comparison favors three groups. The score is modest rather than close to 1, so the segments overlap and should be treated as useful summaries, not perfectly separated customer types.

## Cluster profiles in business language

### Cluster 0 — Customers without an internet service

- Customers: **1,526**
- Average tenure: **30.5 months**
- Average monthly charge: **$21.08**
- Average total charges: **$665.22**
- Observed churn rate (profile only; not a clustering input): **7.4%**
- All customers in this segment have `InternetService = No`; 41.8% have a two-year contract. The segment has the lowest monthly charges and lowest observed churn.\n### Cluster 1 — Higher-risk, month-to-month internet customers

- Customers: **3,189**
- Average tenure: **15.3 months**
- Average monthly charge: **$67.86**
- Average total charges: **$1019.68**
- Observed churn rate (profile only; not a clustering input): **44.0%**
- 87.0% are on month-to-month contracts; 49.3% use electronic check. This segment has the highest observed churn and the shortest average tenure, making it a sensible retention-priority group.\n### Cluster 2 — Longer-tenure, contract-oriented internet customers

- Customers: **2,328**
- Average tenure: **56.9 months**
- Average monthly charge: **$89.15**
- Average total charges: **$5068.03**
- Observed churn rate (profile only; not a clustering input): **15.1%**
- 75.2% have one- or two-year contracts; 57.3% have tech support and 54.4% have online security. They have the longest tenure and lower churn than the month-to-month segment, despite higher monthly charges.\n
## PCA visualization and interpretation

The first two principal components explain **27.2%** and **18.1%** of the variance respectively (combined **45.3%**).

- **PC1** is most associated with monthly charges, total charges, and tenure on the positive side, while the no-internet-service indicators load in the opposite direction. It can be read cautiously as a broad service/charge intensity axis.
- **PC2** loads positively on tenure and total charges (and two-year contracts), and negatively on month-to-month contracts and several “no” support/security indicators. It appears to contrast longer-established/longer-contract customers with shorter-term or less-supported profiles.

These are mixtures of many encoded variables, not direct business measurements. PCA component signs can be flipped without changing the solution, and the first two components omit about 54.7% of total variance. See `outputs/pca_loadings.csv` for the largest loadings.

## Feature importance comparison for the frozen Week 7 model

Permutation importance was calculated on the **untouched 25% stratified holdout set** using the full fitted Week 7 pipeline and average precision as the scoring metric (10 repeats). The Task 7 model was loaded from `outputs/final_churn_pipeline.joblib` and **was not retrained**.

| Feature | Built-in importance (aggregated) | Holdout permutation importance |
|---|---:|---:|
| Contract | 0.4518 | 0.1175 |\n| tenure | 0.0188 | 0.0942 |\n| TotalCharges | 0.0127 | 0.0347 |\n| MonthlyCharges | 0.0122 | 0.0215 |\n| TechSupport | 0.0366 | 0.0190 |\n| OnlineSecurity | 0.0464 | 0.0158 |\n| InternetService | 0.1600 | 0.0147 |\n| PaperlessBilling | 0.0259 | 0.0082 |\n| PaymentMethod | 0.0394 | 0.0076 |\n
Both approaches identify **Contract** as the strongest feature. They disagree on several other variables: built-in importance ranks `InternetService` and `OnlineSecurity` near the top, while holdout permutation importance ranks `tenure`, `TotalCharges`, and `MonthlyCharges` higher. Built-in tree importance measures how the fitted trees used split features; it can favor features with many possible split points or correlated one-hot columns. Permutation importance instead measures the drop in holdout score when a source column is shuffled, so it better reflects reliance for this evaluation set. Correlated features can share or mask importance, so neither ranking is a causal explanation.

## SHAP: top churn drivers in plain English

The SHAP TreeExplainer summary plot was calculated on 1,000 deterministically sampled rows from the Week 7 holdout set, using the frozen XGBoost classifier after the fitted preprocessing transform. Mean absolute SHAP values were aggregated from one-hot columns back to their original source features.

The top three aggregated drivers are:

1. **Contract** (mean absolute SHAP = 0.941): month-to-month contract status pushes model output toward higher churn risk, while longer contracts generally push it lower.
2. **Tenure** (0.370): lower tenure tends to push predictions toward churn; longer-standing customers tend to push predictions away from churn.
3. **MonthlyCharges** (0.281): higher monthly charges tend to push predictions toward churn in this model.

These values describe how the trained model's output is influenced by features for the sampled predictions. They do **not** prove that changing a contract, tenure, or price will cause churn to change. SHAP contributions are model explanations, not causal effects.

## Reproducibility and limitations

Run from this project directory:

```bash
python clustering_pca.py
python interpretability.py
```

The clustering uses one-hot encoded categorical variables with Euclidean distance. This can give high-cardinality or multi-level categorical blocks substantial influence; customerID was removed, but one-hot geometry remains a limitation. A future extension could compare feature-block weighting or a mixed-data distance such as Gower distance. The silhouette score is modest, and the clusters are descriptive rather than ground-truth customer classes.

The interpretability analysis uses the same 25% stratified split and random seed as Week 7 (`random_state=42`). Permutation importance is computed only on holdout rows. SHAP is computed from the frozen model; no test labels are used to retrain or tune it.
