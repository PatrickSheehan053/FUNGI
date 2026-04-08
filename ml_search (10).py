# src/ml_search.py
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor

def get_agnostic_bounds(n_genes):
    """
    Dynamically scales parameter bounds based on the input graph size.

    v4.1 changes (Fix 4 — Hybrid Approach B+C):
    - Lambda bounds widened from [n*0.002, n*0.009] to [n*0.001, n*0.020].
      The old tight bounds locked VCC into avg degree ~7 (below percolation
      threshold for directed PPR propagation). Wide bounds let the GWCC
      percolation penalty in the loss function guide the optimizer to the
      minimum viable density per dataset, rather than hardcoding it.
    - k_core reinterpreted as average degree of the topological core
      (linear scaling: target_edges = n * k_core) instead of density
      fraction (quadratic: n^2 * k_core). Bounds updated accordingly.
    """
    # Lambda: wide bounds, optimizer finds minimum viable density via GWCC penalty
    lam_lower = n_genes * 0.001   # floor: avg degree ~1 (allows very sparse)
    lam_upper = n_genes * 0.020   # ceiling: avg degree ~20 (generous upper bound)

    # k_core: now means "average degree of the topological core"
    # Old bounds [0.01, 0.05] with n^2 scaling gave core edges ~ [n^2*0.01, n^2*0.05]
    # For VCC (3406): [116k, 580k] edges. For Echoes (5127): [263k, 1.3M] edges.
    # New linear scaling: target_edges = n * k_core.
    # k_core in [30, 150] gives VCC [102k, 511k], Echoes [154k, 769k].
    # Ratio between datasets is now 1.5x (linear) not 2.3x (quadratic).
    k_core_lower = 30.0
    k_core_upper = 150.0

    # [beta, gamma, delta, kappa, k_core, lambda]
    lower = np.array([1.0, 0.0, 0.0, 0.02, k_core_lower, lam_lower])
    upper = np.array([5.0, 2.5, 1.0, 0.08, k_core_upper, lam_upper])

    return lower, upper

def generate_latin_hypercube(n_genes, n_samples=12000):
    """Generates uniformly distributed points across the dynamic 6D parameter space."""
    lower_bounds, upper_bounds = get_agnostic_bounds(n_genes)

    sampler = LatinHypercube(d=6)
    sample = sampler.random(n=n_samples)

    # Scale samples to our dynamic bounds
    scaled_samples = lower_bounds + sample * (upper_bounds - lower_bounds)
    return scaled_samples

def train_targeted_models(df_results, elite_fraction=0.05):
    """
    Trains the RF Regressor on all data to predict Utopia Loss,
    and the GMM to map the multi-valley elite parameter space.
    """
    features = ['beta', 'gamma', 'delta', 'kappa', 'k_core', 'lambda']

    # 1. Train Random Forest REGRESSOR on ALL data
    X_all = df_results[features]
    y_all = df_results['utopia_loss']

    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_reg.fit(X_all, y_all)

    # 2. Extract the "Elite" runs for the GMM
    n_elite = int(len(df_results) * elite_fraction)
    df_elite = df_results.nsmallest(n_elite, 'utopia_loss')
    X_elite = df_elite[features]

    # 3. Dynamic Multi-Valley Discovery using BIC
    lowest_bic = np.inf
    best_gmm = None
    best_n = 1

    for n_components in range(1, 11):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=42,
            reg_covar=1e-5
        )
        gmm.fit(X_elite)
        bic = gmm.bic(X_elite)

        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
            best_n = n_components

    print(f"GMM Auto-Discovery: Found {best_n} distinct Utopian Valleys based on BIC.")

    return best_gmm, rf_reg

def generate_targeted_coordinates(gmm, rf_reg, lower_bounds, upper_bounds,
                                  target_loss_threshold, n_needed=8000):
    """
    Proposes points via GMM using rejection sampling to enforce bounds,
    and accepts them only if the RF Regressor predicts a safe Utopia Loss.
    """
    accepted_points = []
    batch_size = 5000

    print("Initiating Targeted Search Generation via Surrogate RF...")
    while len(accepted_points) < n_needed:
        # 1. Propose a large batch of points from the GMM
        proposals, _ = gmm.sample(batch_size)

        # 2. Rejection Sampling to prevent boundary spikes
        valid_mask = np.all((proposals >= lower_bounds) & (proposals <= upper_bounds), axis=1)
        valid_proposals = proposals[valid_mask]

        if len(valid_proposals) == 0:
            continue

        # 3. Predict the Utopia Loss using the Surrogate RF
        predicted_loss = rf_reg.predict(valid_proposals)

        # 4. Fast Vectorized Acceptance
        safe_proposals = valid_proposals[predicted_loss <= target_loss_threshold]
        accepted_points.extend(safe_proposals)

    final_coordinates = np.array(accepted_points)[:n_needed]
    print(f"Successfully generated {len(final_coordinates)} targeted coordinates.")
    return final_coordinates


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train_tri_model_surrogate(df_ledger, feature_cols, target_col='utopia_loss', elite_percentile=0.15):
    """
    Trains the 3-part AI Surrogate Architecture:
    1. RF Classifier (learns the shatter boundaries)
    2. RF Regressor (learns the Utopian gradients)
    3. GMM (learns the Elite 6D covariance)
    """
    print("Training Phase 3B Tri-Model Surrogate Architecture...")

    # --- 1. RF Classifier ---
    X_all = df_ledger[feature_cols].values
    y_clf = df_ledger['is_shattered'].values

    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_clf.fit(X_all, y_clf)
    print(f"  RF Classifier trained on all {len(X_all):,} runs to map shatter cliffs.")

    # --- 2. RF Regressor ---
    # Only train on graphs that survived to map the true valley
    df_survivors = df_ledger[df_ledger['is_shattered'] == 0]
    X_surv = df_survivors[feature_cols].values
    y_reg = df_survivors[target_col].values

    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf_reg.fit(X_surv, y_reg)
    print(f"  RF Regressor trained on {len(X_surv):,} survivors to map Utopian gradients.")

    # --- 3. GMM ---
    # Dynamic 15% cutoff of the survivors
    cutoff_idx = int(len(df_survivors) * elite_percentile)
    df_elite = df_survivors.sort_values(target_col, ascending=True).head(cutoff_idx)
    X_elite = df_elite[feature_cols].values

    best_gmm = None
    best_bic = np.inf
    best_n = 1

    # Auto-tune Gaussian components
    for n_components in range(1, 6):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(X_elite)
        bic = gmm.bic(X_elite)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n_components

    print(f"  GMM mapped Top {elite_percentile*100}% ({len(X_elite):,} elite graphs) using {best_n} components.")

    return best_gmm, rf_clf, rf_reg


def generate_smart_targeted_coordinates(gmm, rf_clf, rf_reg, lower_bounds, upper_bounds, n_hallucinate=150000, n_needed=8000, min_survival_prob=0.90):
    """
    Proposes coordinates via GMM, filters by strict bounds,
    rejects high-risk configurations, and sorts the rest by predicted Utopian Loss.
    """
    print(f"\nInitiating Targeted Search: Generating {n_hallucinate:,} coordinates...")

    # 1. Generate massive surplus
    proposals, _ = gmm.sample(n_hallucinate)

    # 2. Strict Boundary Enforcement
    valid_mask = np.all((proposals >= lower_bounds) & (proposals <= upper_bounds), axis=1)
    valid_proposals = proposals[valid_mask]
    print(f"  -> {len(valid_proposals):,} survived Global Boundary constraints.")

    # 3. Predict Survival Probability
    # rf_clf.predict_proba returns array of shape (n_samples, 2): [prob_0, prob_1]
    # We want prob_0 (probability of is_shattered == 0)
    survival_probs = rf_clf.predict_proba(valid_proposals)[:, 0]
    safe_mask = survival_probs >= min_survival_prob
    safe_proposals = valid_proposals[safe_mask]
    print(f"  -> {len(safe_proposals):,} survived Structural Filter (Survival Confidence >= {min_survival_prob*100}%).")

    if len(safe_proposals) < n_needed:
        raise ValueError(f"Only {len(safe_proposals)} passed safety checks, but {n_needed} are needed. Increase n_hallucinate or lower min_survival_prob.")

    # 4. Predict Utopian Loss
    predicted_losses = rf_reg.predict(safe_proposals)

    # 5. Rank and Select
    sort_idx = np.argsort(predicted_losses)
    best_proposals = safe_proposals[sort_idx]
    best_losses = predicted_losses[sort_idx]

    # Slice the top 8k
    final_coordinates = best_proposals[:n_needed]

    print(f"  Successfully extracted Top {n_needed:,} Targeted Coordinates.")
    print(f"  -> Predicted Utopian Loss Range of Elite 8k: [{best_losses[0]:.4f} to {best_losses[n_needed-1]:.4f}]")

    return final_coordinates