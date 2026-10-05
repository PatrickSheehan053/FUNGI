"""
FUNGI v9.0 — Spatial Niching

v9.0: feature_cols updated to reflect new 6D parameter space.
  gamma removed, psi added.
v10.0: nu (RDF exponent) added as 7th feature column.

Note: as of refinement.py v3.0 (region-capped archive search), Phase 5 does
its own internal basin detection directly on the Phase 3 results and does not
consume this module's anchor_coords/cluster_summary output. This module still
runs and writes its output to disk, but is no longer a functional dependency
of the refinement step.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def extract_anchors(df_results, top_fraction=0.05, n_clusters=50, random_seed=42):
    """
    Identifies spatially diverse anchor points for Phase 5 refinement.

    v9.0 change: feature_cols updated from v7.1 (removed gamma, added psi).
    v10.0 change: nu (RDF exponent) added as 7th feature column.
    """
    feature_cols = ["beta", "delta", "kappa", "k_core", "lambda", "psi", "nu"]

    df_viable = df_results[df_results["is_shattered"] == 0].copy()

    if len(df_viable) == 0:
        raise ValueError("No viable (non-shattered) graphs found in Phase 3 results.")

    n_elite = max(int(len(df_viable) * top_fraction), n_clusters + 1)
    df_elite = df_viable.nsmallest(n_elite, "utopia_loss").copy()

    print(f"Spatial Niching: {len(df_elite):,} elite graphs selected "
          f"(top {top_fraction * 100:.1f}% of {len(df_viable):,} survivors).")

    X_raw = df_elite[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    actual_k = min(n_clusters, len(df_elite) - 1)
    if actual_k < n_clusters:
        print(f"  WARNING: Only {len(df_elite)} elite graphs. "
              f"Reducing clusters to {actual_k}.")

    kmeans = KMeans(n_clusters=actual_k, random_state=random_seed, n_init=10)
    df_elite["cluster"] = kmeans.fit_predict(X_scaled)

    anchor_records = []
    for cluster_id in range(actual_k):
        cluster_df = df_elite[df_elite["cluster"] == cluster_id]
        clean_cluster = cluster_df.dropna(subset=["utopia_loss"])
        if len(clean_cluster) == 0:
            continue
        best_row = clean_cluster.loc[clean_cluster["utopia_loss"].idxmin()]
        anchor_records.append(best_row)

    df_anchors = pd.DataFrame(anchor_records)
    anchor_coords = df_anchors[feature_cols].values
    anchor_losses = df_anchors["utopia_loss"].values

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