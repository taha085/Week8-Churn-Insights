"""Week 8: K-Means customer segmentation and PCA visualization.

Target Churn is excluded from every clustering/PCA input. It is joined back
only after fitting to profile segment churn rates.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
OUT = ROOT / "outputs"
RANDOM_STATE = 42
CANDIDATE_K = [2, 3, 4, 5, 6]


def load_data():
    df = pd.read_csv(DATA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["Churn"]).drop_duplicates("customerID").copy()
    return df


def main():
    OUT.mkdir(exist_ok=True)
    df = load_data()
    # Neither Churn nor customerID is permitted in the unsupervised input.
    X = df.drop(columns=["Churn", "customerID"])
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    Xt = prep.fit_transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=np.float32)

    rows, fitted = [], {}
    for k in CANDIDATE_K:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(Xt)
        sil = silhouette_score(
            Xt, labels, sample_size=min(2000, len(Xt)),
            random_state=RANDOM_STATE
        )
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(sil),
        })
        fitted[k] = (km, labels)

    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "k_selection_metrics.csv", index=False)
    # Select the best silhouette among the tested candidates, not by inspection.
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    best_model, labels = fitted[best_k]
    result = df.copy()
    result["cluster"] = labels
    result.to_csv(OUT / "customer_clusters.csv", index=False)

    # Elbow and silhouette evidence.
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(scores["k"], scores["inertia"], marker="o")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("K-Means inertia")
    ax1.set_title("Elbow analysis")
    ax1.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "elbow_method.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scores["k"], scores["silhouette"], marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette comparison")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "silhouette_scores.png", dpi=160)
    plt.close(fig)

    # PCA is fitted on the same scaled feature matrix, never on Churn.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(Xt)
    pca_df = pd.DataFrame({
        "PC1": pcs[:, 0], "PC2": pcs[:, 1], "cluster": labels,
        "Churn": df["Churn"].to_numpy()
    })
    pca_df.to_csv(OUT / "pca_coordinates.csv", index=False)
    loadings = pd.DataFrame({
        "feature": prep.get_feature_names_out(),
        "PC1_loading": pca.components_[0],
        "PC2_loading": pca.components_[1],
    })
    loadings.to_csv(OUT / "pca_loadings.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap="tab10",
                         s=13, alpha=.65)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"Customer clusters projected onto first two PCs (k={best_k})")
    ax.grid(alpha=.2)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(OUT / "pca_cluster_plot.png", dpi=160)
    plt.close(fig)

    # Business-readable cluster profile; churn is only overlaid after clustering.
    profile = result.groupby("cluster").agg(
        customers=("customerID", "size"),
        avg_tenure=("tenure", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_total_charges=("TotalCharges", "mean"),
        churn_rate=("Churn", "mean"),
        senior_citizen_rate=("SeniorCitizen", "mean"),
    ).reset_index()
    profile["churn_rate"] *= 100
    profile["senior_citizen_rate"] *= 100
    profile.to_csv(OUT / "cluster_profiles.csv", index=False)

    metadata = {
        "selected_k": best_k,
        "selection_rule": "Highest silhouette score among k=2,3,4,5,6",
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_total_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "input_features": X.columns.tolist(),
        "target_excluded_from_clustering": True,
        "customer_id_excluded": True,
        "n_customers": int(len(df)),
    }
    (OUT / "clustering_summary.json").write_text(json.dumps(metadata, indent=2))
    print(scores.to_string(index=False))
    print("\nSelected k:", best_k)
    print(profile.to_string(index=False))
    print("\nPCA variance:", metadata["pca_explained_variance_ratio"])


if __name__ == "__main__":
    main()
