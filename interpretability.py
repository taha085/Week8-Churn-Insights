"""Week 8: Explain the frozen Week 7 churn Pipeline.

Permutation importance is calculated on the untouched stratified holdout set
using raw input columns and the complete fitted Pipeline. The Week 7 model is
loaded and never retrained here.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "outputs" / "final_churn_pipeline.joblib"
OUT = ROOT / "outputs"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["Churn"]).drop_duplicates("customerID").copy()
    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn", "customerID"])
    return X, y


def main():
    OUT.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = joblib.load(MODEL_PATH)
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]

    # Holdout permutation importance uses the full pipeline and original columns.
    perm = permutation_importance(
        model, X_test, y_test, scoring="average_precision",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=1
    )
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values("permutation_importance_mean", ascending=False)
    perm_df.to_csv(OUT / "permutation_importance.csv", index=False)

    # Built-in XGBoost gain/weight importance at encoded-column level,
    # aggregated back to the original source feature for fair comparison.
    encoded_names = preprocessor.get_feature_names_out()
    encoded_importance = classifier.feature_importances_
    def source_name(encoded):
        name = encoded.split("__", 1)[-1]
        # OneHotEncoder names use the original column name as prefix.
        for col in X.columns:
            if name == col or name.startswith(col + "_"):
                return col
        return name
    built = pd.DataFrame({
        "encoded_feature": encoded_names,
        "importance": encoded_importance,
    })
    built["feature"] = built["encoded_feature"].map(source_name)
    built_agg = built.groupby("feature", as_index=False)["importance"].sum()
    built_agg = built_agg.sort_values("importance", ascending=False)
    built_agg.to_csv(OUT / "builtin_feature_importance.csv", index=False)

    compare = built_agg.merge(perm_df, on="feature", how="outer").fillna(0)
    compare["builtin_rank"] = compare["importance"].rank(ascending=False, method="min").astype(int)
    compare["permutation_rank"] = compare["permutation_importance_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    compare["rank_difference"] = compare["builtin_rank"] - compare["permutation_rank"]
    compare.sort_values("permutation_rank").to_csv(
        OUT / "importance_comparison.csv", index=False
    )

    # SHAP TreeExplainer operates on the already-preprocessed holdout matrix.
    Xt_test = preprocessor.transform(X_test)
    if hasattr(Xt_test, "toarray"):
        Xt_test = Xt_test.toarray()
    Xt_test = np.asarray(Xt_test)
    feature_names = list(encoded_names)
    # Bound runtime while retaining a deterministic, representative holdout sample.
    rng = np.random.default_rng(RANDOM_STATE)
    n_sample = min(1000, len(Xt_test))
    sample_idx = np.sort(rng.choice(len(Xt_test), size=n_sample, replace=False))
    X_shap = Xt_test[sample_idx]
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_matrix = np.asarray(shap_values[-1])
    else:
        shap_matrix = np.asarray(shap_values)
        if shap_matrix.ndim == 3:
            # Newer SHAP versions can return (rows, features, classes).
            shap_matrix = shap_matrix[:, :, -1]
    shap.summary_plot(
        shap_matrix, X_shap, feature_names=feature_names,
        show=False, max_display=20
    )
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary_plot.png", dpi=180, bbox_inches="tight")
    plt.close()

    shap_mean = np.abs(shap_matrix).mean(axis=0)
    shap_df = pd.DataFrame({
        "encoded_feature": feature_names,
        "mean_abs_shap": shap_mean,
    }).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(OUT / "shap_feature_importance_encoded.csv", index=False)

    # Aggregate one-hot encoded SHAP contributions to source columns.
    shap_df["feature"] = shap_df["encoded_feature"].map(source_name)
    shap_agg = shap_df.groupby("feature", as_index=False)["mean_abs_shap"].sum()
    shap_agg.sort_values("mean_abs_shap", ascending=False).to_csv(
        OUT / "shap_feature_importance.csv", index=False
    )

    metrics = {
        "holdout_rows": int(len(X_test)),
        "holdout_average_precision": float(average_precision_score(
            y_test, model.predict_proba(X_test)[:, 1]
        )),
        "holdout_roc_auc": float(roc_auc_score(
            y_test, model.predict_proba(X_test)[:, 1]
        )),
        "permutation_scoring": "average_precision",
        "permutation_repeats": 10,
        "shap_rows": int(n_sample),
        "model_retrained": False,
        "holdout_split": "25% stratified; random_state=42, matching Week 7",
    }
    (OUT / "interpretability_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Top permutation importance:\n", perm_df.head(10).to_string(index=False))
    print("\nTop aggregated SHAP:\n", shap_agg.sort_values(
        "mean_abs_shap", ascending=False).head(10).to_string(index=False))
    print("\nTop built-in importance:\n", built_agg.head(10).to_string(index=False))
    print("\nMetrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
