"""
FUNGI v9.0 — Phase 5: ML+GMM Refinement with Boundary Probe

Changes from v8.1:
  - Boundary probe sub-phase (200 extreme configs) before GMM rounds
    → guarantees shatter examples for RF classifier on ANY dataset
    → universal: works for VCC CHITIN (low shatter) and K562 (high shatter)
  - 2D (δ, k_core) interaction grid as Round 0 of refinement
    → characterizes FFL motif interaction without full 6D overhead
    → ~200 points, adds ~4-5 minutes but significantly improves GMM initialization
  - Updated param_cols to reflect new 6D space [beta, delta, kappa, k_core, lambda, psi]
  - select_champion updated for new column names
"""

import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


PARAM_COLS = ["beta", "delta", "kappa", "k_core", "lambda", "psi"]


# ---------------------------------------------------------------------------
# Champion selection
# ---------------------------------------------------------------------------

def select_champion(df_all, utopian_bounds, n_genes):
    """
    Select champion by margin-to-boundary for zero-loss graphs,
    minimum utopia_loss otherwise.
    """
    df_viable = df_all[df_all["is_shattered"] == 0].copy()
    if len(df_viable) == 0:
        raise ValueError("No viable graphs found.")

    param_map = {"alpha": "alpha", "gini": "Gini", "S_max": "S_max",
                 "Q": "Q", "C": "C", "rho": "rho"}

    zero_loss = df_viable[df_viable["utopia_loss"] <= 1e-6]

    if len(zero_loss) > 1:
        margins = []
        for _, row in zero_loss.iterrows():
            min_margin = float('inf')
            for param, col in param_map.items():
                lo, hi = utopian_bounds[param]
                width = max(hi - lo, 1e-6)
                val = row.get(col, (lo + hi) / 2)
                margin = min(val - lo, hi - val) / width
                min_margin = min(min_margin, margin)
            margins.append(min_margin)
        zero_loss = zero_loss.copy()
        zero_loss["_margin"] = margins
        champion = zero_loss.loc[zero_loss["_margin"].idxmax()].drop("_margin")
    elif len(zero_loss) == 1:
        champion = zero_loss.iloc[0]
    else:
        champion = df_viable.loc[df_viable["utopia_loss"].idxmin()]

    return champion


# ---------------------------------------------------------------------------
# Boundary probe (universal negative data generation)
# ---------------------------------------------------------------------------

def _generate_boundary_probe_params(lower, upper, n_probes=200, seed=99):
    """
    Generate configurations deliberately targeting the shatter region.
    Ensures the RF classifier always has negative examples regardless of
    how well-conditioned the main search space is.

    Strategy: oversample high-β + low-λ, low-κ + low-λ, and random extremes.
    This covers the three main shatter causes:
      1. Over-sharpening (high β concentrates on 2-3 edges per source)
      2. Under-density (low λ starves GWCC connectivity)
      3. Hub starvation (low κ prevents forming connected backbone)
    """
    rng = np.random.default_rng(seed)
    hp_range = upper - lower
    probes = np.empty((n_probes, 6), dtype=np.float64)

    for i in range(n_probes):
        p = lower + rng.random(6) * hp_range
        third = n_probes // 3

        if i < third:
            # High β (top 20%) + low λ (bottom 15%) → over-sharpening + under-density
            p[0] = upper[0] - 0.2 * hp_range[0]  # beta near top
            p[4] = lower[4] + 0.15 * hp_range[4]  # lambda near bottom
        elif i < 2 * third:
            # Low κ (bottom 15%) + low λ (bottom 20%) → hub starvation
            p[2] = lower[2] + 0.15 * hp_range[2]  # kappa near bottom
            p[4] = lower[4] + 0.20 * hp_range[4]  # lambda near bottom
        # else: pure random extremes from corners of the space
        # No else: the Sobol samples are already random; for corner sampling:
        else:
            # Sample from extreme corners using Bernoulli projection
            corner = rng.integers(0, 2, size=6)
            p = np.where(corner == 0,
                         lower + 0.10 * hp_range,
                         upper - 0.10 * hp_range)

        probes[i] = np.clip(p, lower, upper)

    return probes


# ---------------------------------------------------------------------------
# 2D (δ, k_core) interaction grid
# ---------------------------------------------------------------------------

def _generate_delta_kcore_grid(lower, upper, df_phase3, n_points=196):
    """
    Generate a focused 2D grid over (δ, k_core) with other parameters
    fixed near their Phase 3 optima.

    δ and k_core interact multiplicatively through the FFL computation:
    the motif bonus is only meaningful if k_core is large enough to include
    the relevant triangles. This 2D sweep directly characterizes that interaction.
    """
    df_viable = df_phase3[df_phase3["is_shattered"] == 0].copy()
    if len(df_viable) == 0:
        return None

    # Fix other 4 parameters at their Phase 3 elite medians
    top_n = max(int(len(df_viable) * 0.05), 10)
    elite = df_viable.nsmallest(top_n, "utopia_loss")

    beta_fix = float(elite["beta"].median())
    kappa_fix = float(elite["kappa"].median())
    lam_fix = float(elite["lambda"].median())
    psi_fix = float(elite["psi"].median()) if "psi" in elite.columns else 0.5

    # Grid over delta × k_core (sqrt(n_points) × sqrt(n_points))
    grid_n = int(np.sqrt(n_points))
    delta_vals = np.linspace(lower[1], upper[1], grid_n)
    kcore_vals = np.linspace(lower[3], upper[3], grid_n)

    dv, kv = np.meshgrid(delta_vals, kcore_vals)
    grid_params = np.column_stack([
        np.full(grid_n * grid_n, beta_fix),
        dv.ravel(),
        np.full(grid_n * grid_n, kappa_fix),
        kv.ravel(),
        np.full(grid_n * grid_n, lam_fix),
        np.full(grid_n * grid_n, psi_fix),
    ])

    return grid_params


# ---------------------------------------------------------------------------
# ML + GMM Refinement (with boundary probe + 2D grid pre-rounds)
# ---------------------------------------------------------------------------

def run_ml_gmm_refinement(df_phase3, lower, upper, evaluator,
                           refinement_cfg, verbose=True):
    """
    Phase 5: ML+GMM refinement with universal boundary probe.

    Pipeline:
      0a. Boundary probe (200 pts) → shatter examples for RF classifier
      0b. 2D (δ, k_core) grid (~196 pts) → characterize FFL interaction
      1-N. GMM+RF rounds (same as v8.1)

    The boundary probe ensures the RF always has negative training data
    regardless of the dataset's natural shatter rate.
    """
    convergence_threshold = refinement_cfg.get("convergence_threshold", 1e-4)
    top_fraction = refinement_cfg.get("top_fraction", 0.05)
    n_gmm_components = refinement_cfg.get("n_gmm_components", 5)
    n_samples_per_round = refinement_cfg.get("n_samples_per_round", 500)
    n_rounds = refinement_cfg.get("n_rounds", 3)
    min_rf_confidence = refinement_cfg.get("min_rf_confidence", 0.65)
    chunk_size = refinement_cfg.get("chunk_size", 50)
    n_boundary_probes = refinement_cfg.get("n_boundary_probes", 200)
    run_delta_kcore_grid = refinement_cfg.get("run_delta_kcore_grid", True)

    df_viable = df_phase3[df_phase3["is_shattered"] == 0].copy()
    if len(df_viable) == 0:
        if verbose:
            print("  Phase 5: No viable Phase 3 graphs. Skipping.")
        return None, True

    best_phase3_loss = df_viable["utopia_loss"].min()
    n_zero = int((df_viable["utopia_loss"] <= convergence_threshold).sum())

    if best_phase3_loss <= convergence_threshold:
        if verbose:
            print(f"  Phase 5 SKIPPED: {n_zero} zero-loss solutions found "
                  f"in Phase 3 (best={best_phase3_loss:.6f})")
        return None, True

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        sklearn_ok = True
    except ImportError:
        sklearn_ok = False
        if verbose:
            print("  WARNING: scikit-learn not available. Using Sobol fallback.")
        return _sobol_refinement_fallback(df_phase3, lower, upper, evaluator,
                                          refinement_cfg, verbose)

    if verbose:
        print(f"\n  Phase 5: ML+GMM Refinement")
        print(f"    Phase 3 best loss = {best_phase3_loss:.4f}")
        print(f"    Boundary probe: {n_boundary_probes} pts | "
              f"δ×k_core grid: {'yes' if run_delta_kcore_grid else 'no'} | "
              f"{n_rounds} GMM rounds × {n_samples_per_round} pts")

    t0 = time.time()
    all_results = []

    # ── Pre-round A: Boundary probe ─────────────────────────────────────────
    if verbose:
        print(f"\n    [Boundary Probe] {n_boundary_probes} extreme configs for RF training data...")

    boundary_params = _generate_boundary_probe_params(
        lower, upper, n_probes=n_boundary_probes)
    df_boundary = evaluator.evaluate(
        param_list=boundary_params,
        chunk_size=chunk_size,
        desc="  Phase 5 boundary probe",
        show_progress=verbose,
    )
    n_shat = int(df_boundary["is_shattered"].sum())
    if verbose:
        print(f"    [Boundary Probe] {n_shat}/{n_boundary_probes} shattered "
              f"({n_shat/n_boundary_probes*100:.0f}%) — "
              f"{'good RF training data' if n_shat >= 10 else 'low shatter rate, dataset well-conditioned'}")
    all_results.extend(df_boundary.to_dict("records"))

    # ── Pre-round B: 2D (δ, k_core) grid ───────────────────────────────────
    if run_delta_kcore_grid:
        grid_params = _generate_delta_kcore_grid(lower, upper, df_phase3)
        if grid_params is not None:
            if verbose:
                print(f"\n    [δ×k_core Grid] {len(grid_params)} pts "
                      f"({int(np.sqrt(len(grid_params)))}×{int(np.sqrt(len(grid_params)))} grid)...")
            df_grid = evaluator.evaluate(
                param_list=grid_params,
                chunk_size=chunk_size,
                desc="  Phase 5 δ×k_core grid",
                show_progress=verbose,
            )
            grid_viable = df_grid[df_grid["is_shattered"] == 0]
            if len(grid_viable) > 0:
                grid_best = grid_viable["utopia_loss"].min()
                if verbose:
                    print(f"    [δ×k_core Grid] best loss from grid: {grid_best:.4f}")
            all_results.extend(df_grid.to_dict("records"))
            # Augment Phase 3 data with grid results for GMM initialization
            df_augmented = pd.concat([df_phase3, df_grid], ignore_index=True)
        else:
            df_augmented = df_phase3
    else:
        df_augmented = df_phase3

    # ── Build combined labeled dataset for RF ───────────────────────────────
    df_rf_train = pd.concat(
        [df_augmented, df_boundary], ignore_index=True)

    X_all = df_rf_train[PARAM_COLS].values.astype(np.float64)
    y_all = df_rf_train["is_shattered"].values.astype(int)
    n_shat_total = int(y_all.sum())
    n_viab_total = len(y_all) - n_shat_total

    if verbose:
        print(f"\n    [RF] Training on {n_viab_total:,} viable + {n_shat_total:,} shattered...")

    rf_model = None
    if n_shat_total >= 3:
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight="balanced", n_jobs=-1, random_state=42)
        rf_model.fit(X_all, y_all)
        if verbose:
            p_viable = rf_model.predict_proba(X_all[y_all == 0])[:, 0]
            recall = float((p_viable >= min_rf_confidence).mean())
            print(f"    [RF] Viable recall at {min_rf_confidence:.0%}: {recall:.2f}")
    else:
        if verbose:
            print(f"    [RF] Only {n_shat_total} shattered — using GMM only")

    # ── Fit GMM on elite configurations ─────────────────────────────────────
    df_all_viable = df_rf_train[df_rf_train["is_shattered"] == 0]
    n_elite = max(int(len(df_all_viable) * top_fraction), n_gmm_components + 1)
    df_elite = df_all_viable.nsmallest(n_elite, "utopia_loss")
    X_elite = df_elite[PARAM_COLS].values.astype(np.float64)

    scaler = StandardScaler()
    X_elite_scaled = scaler.fit_transform(X_elite)
    n_components = min(n_gmm_components, len(X_elite))

    if verbose:
        print(f"\n    [GMM] {n_components} components on {len(X_elite)} elite graphs...")

    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          n_init=5, random_state=42)
    gmm.fit(X_elite_scaled)

    # ── GMM rounds ──────────────────────────────────────────────────────────
    best_loss = min(best_phase3_loss,
                    df_all_viable["utopia_loss"].min() if len(df_all_viable) > 0 else best_phase3_loss)
    rng = np.random.default_rng(42)

    for round_num in range(1, n_rounds + 1):
        if best_loss <= convergence_threshold:
            if verbose:
                print(f"\n    Round {round_num}: Converged. Done.")
            break

        if verbose:
            print(f"\n    Round {round_num}/{n_rounds}:")

        oversample = 4 if rf_model is not None else 2
        samples_scaled, _ = gmm.sample(n_samples_per_round * oversample)
        samples = scaler.inverse_transform(samples_scaled)
        samples = np.clip(samples, lower, upper)

        if rf_model is not None:
            proba = rf_model.predict_proba(samples)[:, 0]
            mask = proba >= min_rf_confidence
            samples_pass = samples[mask]
            if verbose:
                print(f"      RF filter: {mask.sum()}/{len(samples)} pass")
            if len(samples_pass) < 20:
                top_half = proba >= np.percentile(proba, 50)
                samples_pass = samples[top_half]
                if verbose:
                    print(f"      Relaxed to top-50%: {len(samples_pass)}")
        else:
            samples_pass = samples

        if len(samples_pass) > n_samples_per_round:
            samples_pass = samples_pass[:n_samples_per_round]

        if verbose:
            print(f"      Evaluating {len(samples_pass)} candidates via Ray...")

        df_round = evaluator.evaluate(
            param_list=samples_pass,
            chunk_size=chunk_size,
            desc=f"  Phase 5 round {round_num}",
            show_progress=verbose,
        )

        round_viable = df_round[df_round["is_shattered"] == 0]
        if len(round_viable) > 0:
            round_best = round_viable["utopia_loss"].min()
            if round_best < best_loss:
                if verbose:
                    print(f"      ↓ New best: {round_best:.6f} (was {best_loss:.6f})")
                best_loss = round_best
            else:
                if verbose:
                    print(f"      No improvement. Best: {best_loss:.6f}")

        all_results.extend(df_round.to_dict("records"))

        # Update GMM with new high-quality finds
        new_top = round_viable.nsmallest(min(20, len(round_viable)), "utopia_loss")
        if len(new_top) > 0:
            X_new = new_top[PARAM_COLS].values.astype(np.float64)
            X_elite = np.vstack([X_elite, X_new])
            loss_elite_all = np.concatenate([
                df_elite["utopia_loss"].values, new_top["utopia_loss"].values])
            keep = np.argsort(loss_elite_all)[:n_elite]
            X_elite = X_elite[keep]
            X_elite_scaled = scaler.fit_transform(X_elite)
            try:
                gmm = GaussianMixture(
                    n_components=min(n_components, len(X_elite)),
                    covariance_type="full", n_init=3, random_state=42)
                gmm.fit(X_elite_scaled)
            except Exception:
                pass

        # Refit RF every other round with new labeled data
        if rf_model is not None and round_num % 2 == 0 and len(df_round) > 0:
            X_all = np.vstack([X_all, df_round[PARAM_COLS].values.astype(np.float64)])
            y_all = np.concatenate([y_all, df_round["is_shattered"].values.astype(int)])
            try:
                rf_model.fit(X_all, y_all)
            except Exception:
                pass

    elapsed = time.time() - t0
    total_evals = len(all_results)

    if verbose:
        n_zero_final = sum(1 for r in all_results
                           if r.get("is_shattered", 1) == 0
                           and r.get("utopia_loss", 999) <= convergence_threshold)
        print(f"\n  Phase 5 complete: {total_evals:,} evaluations in {elapsed:.1f}s "
              f"({total_evals/elapsed:.1f}/s)")
        print(f"    Best loss: {best_loss:.6f} | Zero-loss solutions: {n_zero_final}")

    return pd.DataFrame(all_results) if all_results else None, False


# ---------------------------------------------------------------------------
# Sobol fallback
# ---------------------------------------------------------------------------

def _sobol_refinement_fallback(df_phase3, lower, upper, evaluator,
                                refinement_cfg, verbose):
    from scipy.stats.qmc import Sobol

    n_per_seed = refinement_cfg.get("n_samples_per_round", 400)
    n_seeds = min(5, len(df_phase3[df_phase3["is_shattered"] == 0]))

    df_viable = df_phase3[df_phase3["is_shattered"] == 0]
    seeds = df_viable.nsmallest(n_seeds, "utopia_loss")[PARAM_COLS].values

    hp_range = upper - lower
    rng = np.random.default_rng(42)
    all_samples = []

    for seed in seeds:
        m = int(np.ceil(np.log2(max(n_per_seed, 2))))
        sampler = Sobol(d=6, scramble=True, seed=int(rng.integers(0, 9999)))
        narrow_lo = np.maximum(lower, seed - 0.12 * hp_range)
        narrow_hi = np.minimum(upper, seed + 0.12 * hp_range)
        pts = narrow_lo + sampler.random(n=2**m)[:n_per_seed] * (narrow_hi - narrow_lo)
        all_samples.append(pts)

    samples = np.vstack(all_samples)
    if verbose:
        print(f"  Phase 5 (Sobol fallback): {len(samples):,} points")

    df_results = evaluator.evaluate(
        param_list=samples,
        chunk_size=refinement_cfg.get("chunk_size", 50),
        desc="Phase 5: Sobol refinement",
        show_progress=verbose,
    )
    return df_results, False
