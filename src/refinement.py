"""
FUNGI — Phase 5: ML+GMM Refinement with Basin-Aware Density Search

Pipeline overview:
  Pre-rounds : boundary probe + delta×k_core interaction grid (batched, single Ray call)
  Loss mode  : exhaustion-based GMM+RF search minimizing utopia loss
  Expansion  : if zero-loss pool is small, grows it before switching modes
  Density mode: DBSCAN detects distinct zero-loss basins in 6D hyperparameter space;
                each basin gets a quality-proportional round and sample budget;
                high-lambda bias steers each round toward the dense frontier;
                re-detection pass finds new basins after primary search completes
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
    df_viable = df_all[df_all["is_shattered"] == 0].copy()
    if len(df_viable) == 0:
        raise ValueError("No viable graphs found.")

    param_map = {"alpha": "alpha", "gini": "Gini", "S_max": "S_max",
                 "Q": "Q", "C": "C", "rho": "rho"}

    df_viable["utopia_loss"] = pd.to_numeric(
        df_viable["utopia_loss"], errors="coerce").fillna(999.0)
    df_viable["n_edges"] = pd.to_numeric(
        df_viable["n_edges"], errors="coerce").fillna(0).astype(int)

    zero_loss = df_viable[df_viable["utopia_loss"] <= 1e-6]

    if len(zero_loss) > 1:
        champion_pos = int(zero_loss["n_edges"].idxmax())
    elif len(zero_loss) == 1:
        champion_pos = int(zero_loss.index[0])
    else:
        margins = []
        for _, row in df_viable.iterrows():
            min_margin = float('inf')
            for param, col in param_map.items():
                lo, hi = utopian_bounds[param]
                width = max(hi - lo, 1e-6)
                val = row.get(col, (lo + hi) / 2)
                margin = min(val - lo, hi - val) / width
                min_margin = min(min_margin, margin)
            margins.append(min_margin)
        df_viable = df_viable.copy()
        df_viable["_margin"] = margins
        champion_pos = int(df_viable["_margin"].idxmax())

    return df_viable.loc[champion_pos]


# ---------------------------------------------------------------------------
# Pre-round generators
# ---------------------------------------------------------------------------

def _generate_boundary_probe_params(lower, upper, n_probes=200, seed=99):
    rng = np.random.default_rng(seed)
    hp_range = upper - lower
    probes = np.empty((n_probes, 6), dtype=np.float64)
    third = n_probes // 3

    for i in range(n_probes):
        p = lower + rng.random(6) * hp_range
        if i < third:
            p[0] = upper[0] - 0.2 * hp_range[0]
            p[4] = lower[4] + 0.15 * hp_range[4]
        elif i < 2 * third:
            p[2] = lower[2] + 0.15 * hp_range[2]
            p[4] = lower[4] + 0.20 * hp_range[4]
        else:
            corner = rng.integers(0, 2, size=6)
            p = np.where(corner == 0,
                         lower + 0.10 * hp_range,
                         upper - 0.10 * hp_range)
        probes[i] = np.clip(p, lower, upper)

    return probes


def _generate_delta_kcore_grid(lower, upper, df_phase3, n_points=196):
    df_viable = df_phase3[df_phase3["is_shattered"] == 0].copy()
    if len(df_viable) == 0:
        return None

    top_n = max(int(len(df_viable) * 0.05), 10)
    elite = df_viable.nsmallest(top_n, "utopia_loss")

    beta_fix  = float(elite["beta"].median())
    kappa_fix = float(elite["kappa"].median())
    lam_fix   = float(elite["lambda"].median())
    psi_fix   = float(elite["psi"].median()) if "psi" in elite.columns else 0.5

    grid_n = int(np.sqrt(n_points))
    delta_vals = np.linspace(lower[1], upper[1], grid_n)
    kcore_vals = np.linspace(lower[3], upper[3], grid_n)
    dv, kv = np.meshgrid(delta_vals, kcore_vals)

    return np.column_stack([
        np.full(grid_n * grid_n, beta_fix),
        dv.ravel(),
        np.full(grid_n * grid_n, kappa_fix),
        kv.ravel(),
        np.full(grid_n * grid_n, lam_fix),
        np.full(grid_n * grid_n, psi_fix),
    ])


# ---------------------------------------------------------------------------
# DBSCAN basin detection
# ---------------------------------------------------------------------------

def _detect_basins(df_zero, param_cols, eps=0.6, min_samples=3):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    df_zero = df_zero.copy()
    df_zero["utopia_loss"] = pd.to_numeric(
        df_zero["utopia_loss"], errors="coerce").fillna(999.0).astype(float)
    df_zero["n_edges"] = pd.to_numeric(
        df_zero["n_edges"], errors="coerce").fillna(0).astype(int)

    if len(df_zero) < min_samples:
        return [df_zero]

    X = df_zero[param_cols].values.astype(np.float64)
    X_scaled = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    unique_labels = [l for l in np.unique(labels) if l >= 0]

    if len(unique_labels) == 0:
        return [df_zero]

    basins = [df_zero[labels == label].copy() for label in unique_labels]

    noise_mask = labels == -1
    if noise_mask.any():
        scaler = StandardScaler().fit(X)
        centroids = np.array([
            scaler.transform(b[param_cols].values.astype(np.float64)).mean(axis=0)
            for b in basins
        ])
        noise_X = scaler.transform(X[noise_mask])
        for i, nx in enumerate(noise_X):
            nearest = int(np.argmin(np.linalg.norm(centroids - nx, axis=1)))
            basins[nearest] = pd.concat(
                [basins[nearest],
                 df_zero.iloc[np.where(noise_mask)[0][i]].to_frame().T],
                ignore_index=True)

    return basins


# ---------------------------------------------------------------------------
# Sobol fallback
# ---------------------------------------------------------------------------

def _sobol_refinement_fallback(df_phase3, lower, upper, evaluator,
                                refinement_cfg, verbose):
    from scipy.stats.qmc import Sobol

    n_per_seed = refinement_cfg.get("n_samples_per_round", 400)
    df_viable = df_phase3[df_phase3["is_shattered"] == 0]
    n_seeds = min(5, len(df_viable))
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
        print(f"  Refinement (Sobol fallback): {len(samples):,} points")

    return evaluator.evaluate(
        param_list=samples,
        chunk_size=refinement_cfg.get("chunk_size", 50),
        desc="  Refinement: Sobol fallback",
        show_progress=verbose,
    ), False


# ---------------------------------------------------------------------------
# Main refinement function
# ---------------------------------------------------------------------------

def run_ml_gmm_refinement(df_phase3, lower, upper, evaluator,
                           refinement_cfg, verbose=True):
    # ── Config reads ──────────────────────────────────────────────────────
    convergence_threshold      = float(refinement_cfg.get("convergence_threshold", 0.0))
    density_epsilon            = float(refinement_cfg.get("density_mode_epsilon", 1e-6))
    top_fraction               = refinement_cfg.get("top_fraction", 0.05)
    n_gmm_components           = refinement_cfg.get("n_gmm_components", 5)
    n_samples_per_round        = refinement_cfg.get("n_samples_per_round", 500)
    min_rf_confidence          = refinement_cfg.get("min_rf_confidence", 0.65)
    chunk_size                 = refinement_cfg.get("chunk_size", 50)
    n_boundary_probes          = refinement_cfg.get("n_boundary_probes", 200)
    run_delta_kcore_grid       = refinement_cfg.get("run_delta_kcore_grid", True)
    exhaustion_mode            = refinement_cfg.get("exhaustion_mode", True)
    n_rounds_fixed             = refinement_cfg.get("n_rounds", 3)
    patience                   = refinement_cfg.get("patience", 2)
    density_rounds_per_basin   = refinement_cfg.get("density_rounds_after_zero", 5)
    density_patience           = refinement_cfg.get("density_patience", 1)
    min_edge_improvement       = int(refinement_cfg.get("min_edge_improvement", 50))
    min_zero_for_density       = int(refinement_cfg.get("min_zero_for_density", 30))
    basin_expansion_rounds     = int(refinement_cfg.get("basin_expansion_rounds", 3))
    dbscan_eps                 = float(refinement_cfg.get("dbscan_eps", 0.6))
    dbscan_min_samples         = int(refinement_cfg.get("dbscan_min_samples", 3))
    basin_ceiling_gap_fraction = float(refinement_cfg.get("basin_ceiling_gap_fraction", 0.08))
    redetection_patience       = int(refinement_cfg.get("redetection_patience", 1))

    df_viable = df_phase3[df_phase3["is_shattered"] == 0].copy()
    if len(df_viable) == 0:
        if verbose:
            print("  Refinement: No viable graphs from expansive search. Skipping.")
        return None, True

    df_viable["utopia_loss"] = pd.to_numeric(
        df_viable["utopia_loss"], errors="coerce").fillna(999.0)
    df_viable["n_edges"] = pd.to_numeric(
        df_viable["n_edges"], errors="coerce").fillna(0).astype(int)

    best_phase3_loss = float(df_viable["utopia_loss"].min())
    n_zero_phase3 = int((df_viable["utopia_loss"] <= density_epsilon).sum())

    density_mode = best_phase3_loss <= convergence_threshold
    expansion_rounds_remaining = 0
    in_expansion = False

    if density_mode and n_zero_phase3 < min_zero_for_density:
        if verbose:
            print(f"\n  Refinement: Expansive search found {n_zero_phase3} zero-loss solutions "
                  f"(need {min_zero_for_density}). Running {basin_expansion_rounds} "
                  f"expansion round(s) first.")
        expansion_rounds_remaining = basin_expansion_rounds
        density_mode = False
        in_expansion = True

    if verbose:
        mode_label = "density maximization" if density_mode else "loss minimization"
        stop_label = (f"exhaustion (patience={patience})"
                      if exhaustion_mode else f"fixed {n_rounds_fixed} rounds")
        print(f"\n  ── Refinement Search  [{mode_label}] [{stop_label}]")
        print(f"     Expansive search best loss: {best_phase3_loss:.6f} | "
              f"Zero-loss solutions: {n_zero_phase3:,}")

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        if verbose:
            print("  WARNING: scikit-learn not available. Using Sobol fallback.")
        return _sobol_refinement_fallback(df_phase3, lower, upper, evaluator,
                                          refinement_cfg, verbose)

    t0 = time.time()

    # ── Batch pre-rounds ──────────────────────────────────────────────────
    boundary_params = _generate_boundary_probe_params(
        lower, upper, n_probes=n_boundary_probes)
    grid_params = None
    if run_delta_kcore_grid:
        grid_params = _generate_delta_kcore_grid(lower, upper, df_phase3)

    pre_parts = [boundary_params]
    boundary_end = len(boundary_params)
    grid_end = boundary_end
    if grid_params is not None:
        pre_parts.append(grid_params)
        grid_end = boundary_end + len(grid_params)

    pre_round_all = np.vstack(pre_parts)

    if verbose:
        print(f"\n     [Pre-rounds] {len(pre_round_all)} configs "
              f"(boundary: {boundary_end}"
              + (f", δ×k_core grid: {len(grid_params)}" if grid_params is not None else "")
              + ") — single parallel batch")

    df_pre = evaluator.evaluate(
        param_list=pre_round_all, chunk_size=chunk_size,
        desc="  Refinement pre-rounds", show_progress=verbose)

    df_boundary = df_pre.iloc[:boundary_end].reset_index(drop=True)
    n_shat = int(df_boundary["is_shattered"].sum())
    if verbose:
        print(f"     [Boundary probe] {n_shat}/{n_boundary_probes} shattered "
              f"({n_shat / n_boundary_probes * 100:.0f}%) — "
              + ("RF has negative training data" if n_shat >= 3
                 else "dataset well-conditioned, RF runs GMM-only"))

    df_grid = None
    if grid_params is not None:
        df_grid = df_pre.iloc[boundary_end:grid_end].reset_index(drop=True)
        grid_viable = df_grid[df_grid["is_shattered"] == 0]
        if verbose and len(grid_viable) > 0:
            grid_best = pd.to_numeric(
                grid_viable["utopia_loss"], errors="coerce").min()
            print(f"     [δ×k_core grid] best loss: {grid_best:.6f}")

    # ── RF training set ───────────────────────────────────────────────────
    df_augmented = pd.concat(
        [df_phase3] + ([df_grid] if df_grid is not None else []),
        ignore_index=True)
    df_rf_train = pd.concat([df_augmented, df_boundary], ignore_index=True)

    X_all = df_rf_train[PARAM_COLS].values.astype(np.float64)
    y_all = df_rf_train["is_shattered"].values.astype(int)
    n_shat_total = int(y_all.sum())
    n_viab_total = len(y_all) - n_shat_total

    if verbose:
        print(f"\n     [RF classifier] {n_viab_total:,} viable + {n_shat_total:,} shattered")

    rf_model = None
    if n_shat_total >= 3:
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight="balanced", n_jobs=-1, random_state=42)
        rf_model.fit(X_all, y_all)
        if verbose:
            p_viable = rf_model.predict_proba(X_all[y_all == 0])[:, 0]
            recall = float((p_viable >= min_rf_confidence).mean())
            print(f"     [RF classifier] Viable recall @ {min_rf_confidence:.0%}: {recall:.2f}")
    else:
        if verbose:
            print(f"     [RF classifier] Insufficient shattered examples — GMM only")

    # ── Inner helpers ─────────────────────────────────────────────────────

    def _cast_df(df):
        df = df.copy()
        df["utopia_loss"] = pd.to_numeric(
            df["utopia_loss"], errors="coerce").fillna(999.0).astype(float)
        df["n_edges"] = pd.to_numeric(
            df["n_edges"], errors="coerce").fillna(0).astype(int)
        return df

    def _fit_gmm_on_pool(df_pool, mode_is_density):
        df_pool = _cast_df(df_pool)
        if mode_is_density:
            zero_pool = df_pool[df_pool["utopia_loss"] <= density_epsilon].copy()
            if len(zero_pool) == 0:
                zero_pool = df_pool.nsmallest(
                    max(int(len(df_pool) * top_fraction), n_gmm_components + 1),
                    "utopia_loss")
            n_el = max(int(len(zero_pool) * max(top_fraction, 0.20)),
                       n_gmm_components + 1)
            n_top = max(n_el // 2, n_gmm_components)
            n_margin = max(n_el - n_top, 1)
            df_top = zero_pool.nlargest(n_top, "n_edges")
            df_margin = zero_pool.nsmallest(n_margin, "utopia_loss")
            df_el = pd.concat([df_top, df_margin]).drop_duplicates()
        else:
            n_el = max(int(len(df_pool) * top_fraction), n_gmm_components + 1)
            df_el = df_pool.nsmallest(n_el, "utopia_loss")

        X_el = df_el[PARAM_COLS].values.astype(np.float64)
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_el)
        n_comp = min(n_gmm_components, len(X_el))
        gm = GaussianMixture(n_components=n_comp, covariance_type="full",
                             n_init=5, random_state=42)
        gm.fit(X_sc)
        return gm, sc, X_el, df_el

    def _sample_from_gmm(gmm, scaler, n_want, basin_centroid=None,
                         high_lambda_bias=0.15):
        oversample = 4 if rf_model is not None else 2
        samples_scaled, _ = gmm.sample(n_want * oversample)
        samples = scaler.inverse_transform(samples_scaled)
        samples = np.clip(samples, lower, upper)

        if basin_centroid is not None:
            lambda_idx = PARAM_COLS.index("lambda")
            n_bias = max(1, int(n_want * high_lambda_bias))
            rng_bias = np.random.default_rng(int(time.time() * 1000) % (2**31))
            bias_samples = np.tile(basin_centroid, (n_bias, 1))
            anchor_lambda = basin_centroid[lambda_idx]
            bias_samples[:, lambda_idx] = np.linspace(
                anchor_lambda, upper[lambda_idx], n_bias)
            jitter = rng_bias.normal(0, 0.02, size=bias_samples.shape)
            jitter[:, lambda_idx] = 0.0
            bias_samples = np.clip(bias_samples + jitter, lower, upper)
            samples = np.vstack([samples, bias_samples])

        if rf_model is not None:
            proba = rf_model.predict_proba(samples)[:, 0]
            mask = proba >= min_rf_confidence
            samples_pass = samples[mask]
            if len(samples_pass) < 10:
                top_half = proba >= np.percentile(proba, 50)
                samples_pass = samples[top_half]
        else:
            samples_pass = samples

        return samples_pass[:n_want] if len(samples_pass) > n_want else samples_pass

    def _run_basin_search(zero_pool, existing_basin_centroids=None,
                          is_redetection=False):
        zero_pool = _cast_df(zero_pool)
        basins = _detect_basins(zero_pool, PARAM_COLS,
                                eps=dbscan_eps, min_samples=dbscan_min_samples)

        if verbose:
            print(f"\n     [Basin detection] {len(basins)} basin(s) from "
                  f"{len(zero_pool):,} zero-loss solutions")

        current_best_edges = int(zero_pool["n_edges"].max()) if len(zero_pool) > 0 else 0
        edge_floor = current_best_edges * (1.0 - basin_ceiling_gap_fraction)

        scaler_rep = StandardScaler()
        X_zero = zero_pool[PARAM_COLS].values.astype(np.float64)
        scaler_rep.fit(X_zero if len(X_zero) > 1 else np.vstack([X_zero, X_zero]))

        new_states = []
        skipped = 0

        for b_idx, basin_df in enumerate(basins):
            basin_df = _cast_df(basin_df)
            basin_best_edges = int(basin_df["n_edges"].max())
            centroid = basin_df[PARAM_COLS].values.astype(np.float64).mean(axis=0)
            centroid_norm = scaler_rep.transform(centroid.reshape(1, -1))[0]

            if existing_basin_centroids and len(existing_basin_centroids) > 0:
                dists = [np.linalg.norm(centroid_norm - ec)
                         for ec in existing_basin_centroids]
                if min(dists) < dbscan_eps:
                    skipped += 1
                    continue

            if basin_best_edges < edge_floor:
                if verbose:
                    gap_pct = ((current_best_edges - basin_best_edges)
                               / max(current_best_edges, 1) * 100)
                    print(f"       Basin {b_idx + 1}: skipped "
                          f"(best={basin_best_edges:,}, "
                          f"{gap_pct:.1f}% below champion, "
                          f"threshold={basin_ceiling_gap_fraction*100:.0f}%)")
                skipped += 1
                continue

            quality_score = ((basin_best_edges / max(current_best_edges, 1))
                             * np.log1p(len(basin_df)))
            max_quality = np.log1p(len(zero_pool))
            quality_frac = quality_score / max(max_quality, 1e-6)

            rounds_budget = max(1, int(np.round(
                density_rounds_per_basin * quality_frac)))
            rounds_budget = min(rounds_budget, density_rounds_per_basin)
            if is_redetection:
                rounds_budget = min(rounds_budget, redetection_patience)

            samples_budget = max(100, int(n_samples_per_round * quality_frac))
            samples_budget = min(samples_budget, n_samples_per_round)

            gm, sc, X_el, _ = _fit_gmm_on_pool(basin_df, mode_is_density=True)
            global_idx = (len(existing_basin_centroids)
                          if existing_basin_centroids else 0) + len(new_states)

            lambda_idx = PARAM_COLS.index("lambda")
            lambda_pct = ((centroid[lambda_idx] - lower[lambda_idx])
                          / max(upper[lambda_idx] - lower[lambda_idx], 1e-6) * 100)

            new_states.append({
                "idx":               global_idx,
                "centroid_norm":     centroid_norm,
                "df":                basin_df,
                "gmm":               gm,
                "scaler":            sc,
                "X_elite":           X_el,
                "best_edges":        basin_best_edges,
                "rounds_left":       rounds_budget,
                "samples_per_round": samples_budget,
                "no_improve":        0,
                "exhausted":         False,
                "solutions_found":   len(basin_df),
                "is_redetection":    is_redetection,
            })

            if verbose:
                print(f"       Basin {global_idx + 1}: "
                      f"{len(basin_df)} solutions, "
                      f"best={basin_best_edges:,} edges, "
                      f"budget={rounds_budget} round(s) × {samples_budget} samples, "
                      f"λ={centroid[lambda_idx]:.2f} ({lambda_pct:.0f}% of range) "
                      f"δ={centroid[PARAM_COLS.index('delta')]:.2f} "
                      f"β={centroid[PARAM_COLS.index('beta')]:.2f}"
                      + (" [re-detection]" if is_redetection else ""))

        if skipped > 0 and verbose:
            print(f"       Skipped {skipped} basin(s) "
                  f"(below gap threshold or already covered)")

        if len(new_states) > 1 and verbose:
            print(f"\n       Inter-basin distances (normalized 6D):")
            for i in range(len(new_states)):
                for j in range(i + 1, len(new_states)):
                    d = np.linalg.norm(new_states[i]["centroid_norm"]
                                       - new_states[j]["centroid_norm"])
                    print(f"         Basin {new_states[i]['idx']+1} ↔ "
                          f"Basin {new_states[j]['idx']+1}: {d:.3f}")

        return new_states

    def _search_basin_list(basin_list, round_num_ref, density_round_ref):
        round_num = round_num_ref
        density_round = density_round_ref

        while True:
            active = [s for s in basin_list if not s["exhausted"]]
            if not active:
                break

            for state in list(active):
                if state["rounds_left"] <= 0 or state["exhausted"]:
                    state["exhausted"] = True
                    continue

                density_round += 1
                round_num += 1
                state["rounds_left"] -= 1
                patience_limit = (redetection_patience
                                  if state["is_redetection"] else density_patience)

                if verbose:
                    tag = "re-detection" if state["is_redetection"] else "density"
                    print(f"\n     Round {round_num} [{tag} — basin "
                          f"{state['idx']+1}, "
                          f"rounds left: {state['rounds_left']+1}, "
                          f"no-improve: {state['no_improve']}/{patience_limit}, "
                          f"samples: {state['samples_per_round']}]:")

                # Anchor high-lambda bias to the basin's best-edges solution
                best_row = state["df"].loc[state["df"]["n_edges"].idxmax()]
                best_anchor = best_row[PARAM_COLS].values.astype(np.float64)

                samples_pass = _sample_from_gmm(
                    state["gmm"], state["scaler"],
                    state["samples_per_round"],
                    basin_centroid=best_anchor)

                if verbose:
                    print(f"       Evaluating {len(samples_pass)} candidates...")

                df_round = _cast_df(evaluator.evaluate(
                    param_list=samples_pass, chunk_size=chunk_size,
                    desc=f"  Refinement density r{density_round} b{state['idx']+1}",
                    show_progress=verbose).copy())

                df_round["basin_idx"] = state["idx"]
                all_results.extend(df_round.to_dict("records"))

                round_viable = df_round[df_round["is_shattered"] == 0]
                round_zero = (round_viable[round_viable["utopia_loss"] <= density_epsilon]
                              if len(round_viable) > 0 else pd.DataFrame())

                if len(round_zero) > 0:
                    round_best_edges = int(round_zero["n_edges"].max())
                    edge_gain = round_best_edges - state["best_edges"]

                    if edge_gain >= min_edge_improvement:
                        if verbose:
                            print(f"       ↑ Basin {state['idx']+1}: "
                                  f"{round_best_edges:,} edges (+{edge_gain:,})")
                        state["best_edges"] = round_best_edges
                        state["no_improve"] = 0
                        state["df"] = pd.concat(
                            [state["df"], round_zero], ignore_index=True)
                        state["solutions_found"] += len(round_zero)
                        try:
                            gm, sc, X_el, _ = _fit_gmm_on_pool(
                                state["df"], mode_is_density=True)
                            state["gmm"] = gm
                            state["scaler"] = sc
                            state["X_elite"] = X_el
                        except Exception:
                            pass
                    else:
                        state["no_improve"] += 1
                        if verbose:
                            gain_str = (f"+{edge_gain}" if edge_gain > 0
                                        else "no change")
                            print(f"       No meaningful improvement "
                                  f"({gain_str} edges, threshold={min_edge_improvement}). "
                                  f"Streak: {state['no_improve']}/{patience_limit}")
                else:
                    state["no_improve"] += 1
                    if verbose:
                        print(f"       No zero-loss solutions found. "
                              f"Streak: {state['no_improve']}/{patience_limit}")

                if state["no_improve"] >= patience_limit:
                    state["exhausted"] = True
                    if verbose:
                        print(f"       Basin {state['idx']+1} exhausted "
                              f"(best: {state['best_edges']:,} edges).")

        return round_num, density_round

    # ── Accumulate viable data ────────────────────────────────────────────
    df_all_viable = _cast_df(df_rf_train[df_rf_train["is_shattered"] == 0].copy())
    all_results = df_pre.to_dict("records")
    best_loss = (float(df_all_viable["utopia_loss"].min())
                 if len(df_all_viable) > 0 else best_phase3_loss)
    consecutive_no_improve = 0
    round_num = 0
    basin_states = []

    # ── Initial GMM ───────────────────────────────────────────────────────
    gmm_loss, scaler_loss, X_elite_loss, df_elite_loss = _fit_gmm_on_pool(
        df_all_viable, mode_is_density=False)

    if verbose and not density_mode:
        print(f"\n     [GMM] {min(n_gmm_components, len(df_elite_loss))} components "
              f"on {len(df_elite_loss)} elite graphs (loss mode)")

    # ── Loss / expansion loop ─────────────────────────────────────────────
    while not density_mode:
        round_num += 1

        if in_expansion:
            if expansion_rounds_remaining <= 0:
                in_expansion = False
                zero_count = int((df_all_viable["utopia_loss"] <= density_epsilon).sum())
                if zero_count > 0:
                    if verbose:
                        print(f"\n     Expansion complete: {zero_count} zero-loss solutions. "
                              f"Switching to density mode.")
                    density_mode = True
                    break
                else:
                    if verbose:
                        print(f"\n     Expansion complete: no zero-loss. Continuing.")
            else:
                expansion_rounds_remaining -= 1
        else:
            if exhaustion_mode:
                if consecutive_no_improve >= patience:
                    if verbose:
                        print(f"\n     Patience exhausted.")
                    break
            else:
                if round_num > n_rounds_fixed:
                    break

        if verbose:
            tag = "expansion" if in_expansion else "loss"
            streak = (f", no-improve: {consecutive_no_improve}/{patience}"
                      if not in_expansion and exhaustion_mode else "")
            print(f"\n     Round {round_num} [{tag}{streak}]:")

        samples_pass = _sample_from_gmm(gmm_loss, scaler_loss, n_samples_per_round)
        if verbose:
            print(f"       Evaluating {len(samples_pass)} candidates...")

        df_round = _cast_df(evaluator.evaluate(
            param_list=samples_pass, chunk_size=chunk_size,
            desc=f"  Refinement round {round_num}", show_progress=verbose))

        round_viable = df_round[df_round["is_shattered"] == 0]
        all_results.extend(df_round.to_dict("records"))

        if len(round_viable) > 0:
            df_all_viable = pd.concat([df_all_viable, round_viable], ignore_index=True)
            round_best = float(round_viable["utopia_loss"].min())
            total_zero = int((df_all_viable["utopia_loss"] <= density_epsilon).sum())

            if total_zero >= min_zero_for_density:
                if verbose:
                    print(f"       Zero-loss pool sufficient "
                          f"({total_zero} >= {min_zero_for_density}). "
                          f"Switching to density mode.")
                density_mode = True
                break
            elif total_zero > 0:
                if verbose:
                    print(f"       Zero-loss hit ({total_zero} solutions, "
                          f"need {min_zero_for_density}). Growing pool...")
                best_loss = min(best_loss, round_best)
                consecutive_no_improve = 0
            elif round_best < best_loss - 1e-8:
                if verbose:
                    print(f"       ↓ New best: {round_best:.6f}")
                best_loss = round_best
                consecutive_no_improve = 0
            else:
                consecutive_no_improve += 1
                if verbose:
                    print(f"       No improvement. Best: {best_loss:.6f} "
                          f"(streak: {consecutive_no_improve})")

            new_top = round_viable.nsmallest(min(20, len(round_viable)), "utopia_loss")
            if len(new_top) > 0:
                X_new = new_top[PARAM_COLS].values.astype(np.float64)
                X_elite_loss = np.vstack([X_elite_loss, X_new])
                try:
                    scaler_loss = StandardScaler()
                    gmm_loss = GaussianMixture(
                        n_components=min(n_gmm_components, len(X_elite_loss)),
                        covariance_type="full", n_init=3, random_state=42)
                    gmm_loss.fit(scaler_loss.fit_transform(X_elite_loss))
                except Exception:
                    pass
        else:
            consecutive_no_improve += 1

        if rf_model is not None and round_num % 2 == 0 and len(df_round) > 0:
            X_all = np.vstack([X_all,
                               df_round[PARAM_COLS].values.astype(np.float64)])
            y_all = np.concatenate([y_all,
                                    df_round["is_shattered"].values.astype(int)])
            try:
                rf_model.fit(X_all, y_all)
            except Exception:
                pass

    # ── Density mode ──────────────────────────────────────────────────────
    density_round = 0
    if density_mode:
        df_all_viable = _cast_df(df_all_viable)
        zero_pool = df_all_viable[
            df_all_viable["utopia_loss"] <= density_epsilon].copy()

        basin_states = _run_basin_search(zero_pool, is_redetection=False)
        all_basin_centroids = [s["centroid_norm"] for s in basin_states]

        round_num, density_round = _search_basin_list(
            basin_states, round_num, density_round)

        if verbose:
            print(f"\n     [Re-detection pass] Scanning accumulated zero-loss pool...")

        full_zero = pd.DataFrame(
            [r for r in all_results
             if r.get("is_shattered", 1) == 0
             and float(r.get("utopia_loss", 999.)) <= density_epsilon])

        if len(full_zero) > 0:
            full_zero = _cast_df(full_zero)
            new_basin_states = _run_basin_search(
                full_zero,
                existing_basin_centroids=all_basin_centroids,
                is_redetection=True)

            if new_basin_states:
                if verbose:
                    print(f"     Found {len(new_basin_states)} new qualifying basin(s).")
                basin_states.extend(new_basin_states)
                round_num, density_round = _search_basin_list(
                    new_basin_states, round_num, density_round)
            else:
                if verbose:
                    print(f"     No new qualifying basins found.")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    total_evals = len(all_results)
    n_zero_final = sum(
        1 for r in all_results
        if r.get("is_shattered", 1) == 0
        and float(r.get("utopia_loss", 999.)) <= density_epsilon)

    if verbose:
        print(f"\n  ── Refinement complete: {total_evals:,} evaluations in "
              f"{elapsed:.1f}s ({total_evals / elapsed:.1f}/s)")
        if density_mode and basin_states:
            all_best = max(s["best_edges"] for s in basin_states)
            print(f"     Basins searched: {len(basin_states)} | "
                  f"Zero-loss solutions: {n_zero_final:,} | "
                  f"Densest zero-loss: {all_best:,} edges")
            print(f"     Per-basin summary:")
            for s in basin_states:
                tag = " [re-detection]" if s.get("is_redetection") else ""
                print(f"       Basin {s['idx']+1}: "
                      f"best={s['best_edges']:,} edges, "
                      f"{s['solutions_found']} solutions{tag}")
        else:
            print(f"     Best loss: {best_loss:.6f} | "
                  f"Zero-loss solutions: {n_zero_final:,}")

    return pd.DataFrame(all_results) if all_results else None, False


# ---------------------------------------------------------------------------
# Diverse cohort selection
# ---------------------------------------------------------------------------

def select_diverse_cohort(df_all, utopian_bounds, n_genes, cohort_size=5,
                          max_loss_multiplier=3.0):
    df_viable = df_all[df_all["is_shattered"] == 0].copy()
    if len(df_viable) == 0:
        raise ValueError("No viable graphs found.")

    df_viable["utopia_loss"] = pd.to_numeric(
        df_viable["utopia_loss"], errors="coerce").fillna(999.0)
    df_viable["n_edges"] = pd.to_numeric(
        df_viable["n_edges"], errors="coerce").fillna(0).astype(int)

    topo_map = [
        ("alpha", "alpha"), ("gini", "Gini"), ("S_max", "S_max"),
        ("Q", "Q"), ("C", "C"), ("rho", "rho"),
    ]

    topo_mat = np.zeros((len(df_viable), 6), dtype=np.float64)
    for col_idx, (bound_key, df_col) in enumerate(topo_map):
        lo, hi = utopian_bounds[bound_key]
        col_range = max(hi - lo, 1e-6)
        vals = df_viable[df_col].values.astype(np.float64)
        topo_mat[:, col_idx] = (vals - lo) / col_range

    df_viable = df_viable.reset_index(drop=True)
    champion_loss = float(df_viable["utopia_loss"].min())
    zero_candidates = df_viable[df_viable["utopia_loss"] <= 1e-6]

    if len(zero_candidates) > 1:
        champion_pos = int(zero_candidates["n_edges"].idxmax())
    else:
        champion_pos = int(df_viable["utopia_loss"].idxmin())

    if champion_loss > 1e-6:
        max_loss = champion_loss * max_loss_multiplier
    else:
        max_loss = float(df_viable["utopia_loss"].quantile(0.15))

    eligible_mask = df_viable["utopia_loss"].values <= max_loss
    eligible_indices = np.where(eligible_mask)[0].tolist()
    if champion_pos not in eligible_indices:
        eligible_indices.append(champion_pos)

    selected_positions = [champion_pos]
    selected_topo = [topo_mat[champion_pos].copy()]

    for _ in range(min(cohort_size - 1, len(eligible_indices) - 1)):
        best_min_dist = -1.0
        best_pos = None
        for pos in eligible_indices:
            if pos in selected_positions:
                continue
            candidate = topo_mat[pos]
            min_dist = min(float(np.linalg.norm(candidate - s))
                           for s in selected_topo)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = pos
        if best_pos is None:
            break
        selected_positions.append(best_pos)
        selected_topo.append(topo_mat[best_pos].copy())

    cohort = df_viable.iloc[selected_positions].copy()
    cohort = cohort.reset_index(drop=True)
    cohort["cohort_rank"] = list(range(1, len(selected_positions) + 1))
    cohort["is_champion"] = [True] + [False] * (len(selected_positions) - 1)
    return cohort
