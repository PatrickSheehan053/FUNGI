"""
FUNGI v6 -- Spatial Niching (Part C: Bridge to TuRBO)

Extracts a diverse set of anchor coordinates from the Phase 1 Sobol results
by clustering the top-performing graphs in 6D hyperparameter space and
selecting one local champion per cluster.

This prevents TuRBO from receiving a cloud of redundant near-duplicates
and ensures the refinement phase explores the entire Pareto front.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def extract_anchors(df_results, top_fraction=0.05, n_clusters=50,
                    random_seed=42):
    """Identifies spatially diverse anchor points for TuRBO initialization.

    Pipeline:
    1. Filter out shattered graphs.
    2. Slice the top fraction by utopian loss.
    3. Standardize the 6D coordinates (critical: lambda and kappa operate
       on vastly different scales).
    4. K-Means clustering in standardized space.
    5. Extract the single best graph per cluster (lowest loss).

    Parameters
    ----------
    df_results : pd.DataFrame
        Merged Phase 1 results with columns: beta, gamma, delta, kappa,
        k_core, lambda, utopia_loss, is_shattered.
    top_fraction : float
        Fraction of surviving graphs to retain (e.g., 0.05 = top 5%).
    n_clusters : int
        Number of spatial niches (K-Means clusters).
    random_seed : int
        Random seed for K-Means reproducibility.

    Returns
    -------
    anchor_coords : np.ndarray, shape (n_clusters, 6)
        Un-standardized 6D coordinates of the local champions.
    anchor_losses : np.ndarray, shape (n_clusters,)
        Utopian loss values of the local champions.
    cluster_summary : pd.DataFrame
        Per-cluster statistics for logging.
    """
    feature_cols = ["beta", "gamma", "delta", "kappa", "k_core", "lambda"]

    # Step 1: Remove shattered graphs
    df_viable = df_results[df_results["is_shattered"] == 0].copy()

    if len(df_viable) == 0:
        raise ValueError("No viable (non-shattered) graphs found in Phase 1 results.")

    # Step 2: Top-slice by utopian loss
    n_elite = max(int(len(df_viable) * top_fraction), n_clusters + 1)
    df_elite = df_viable.nsmallest(n_elite, "utopia_loss").copy()

    print(f"Spatial Niching: {len(df_elite):,} elite graphs selected "
          f"(top {top_fraction * 100:.1f}% of {len(df_viable):,} survivors).")

    # Step 3: Standardize 6D coordinates
    X_raw = df_elite[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Step 4: K-Means clustering
    actual_k = min(n_clusters, len(df_elite) - 1)
    if actual_k < n_clusters:
        print(f"  WARNING: Only {len(df_elite)} elite graphs available. "
              f"Reducing clusters from {n_clusters} to {actual_k}.")

    kmeans = KMeans(n_clusters=actual_k, random_state=random_seed, n_init=10)
    df_elite["cluster"] = kmeans.fit_predict(X_scaled)

    # Step 5: Extract one local champion per cluster
    anchor_records = []
    for cluster_id in range(actual_k):
        cluster_df = df_elite[df_elite["cluster"] == cluster_id]
        if len(cluster_df) == 0:
            continue
        best_row = cluster_df.loc[cluster_df["utopia_loss"].idxmin()]
        anchor_records.append(best_row)

    df_anchors = pd.DataFrame(anchor_records)
    anchor_coords = df_anchors[feature_cols].values
    anchor_losses = df_anchors["utopia_loss"].values

    # Build cluster summary
    cluster_summary = df_elite.groupby("cluster").agg(
        n_members=("utopia_loss", "count"),
        best_loss=("utopia_loss", "min"),
        mean_loss=("utopia_loss", "mean"),
    ).reset_index()

    print(f"  Extracted {len(anchor_coords)} anchor coordinates across "
          f"{actual_k} spatial niches.")
    print(f"  Loss range of anchors: [{anchor_losses.min():.4f}, "
          f"{anchor_losses.max():.4f}]")

    return anchor_coords, anchor_losses, cluster_summary
