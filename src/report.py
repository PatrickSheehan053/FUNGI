"""
FUNGI — src/report.py
─────────────────────
Post-run diagnostic report generator.

Produces a single self-contained Markdown report covering every stage of a
FUNGI run: input data, diagnostic calibration, prefiltering, biological
priors, structural scoring, search trajectory, cohort overview, champion
topology deep-dive, hub/target analysis, perturbation coverage, DEG
reachability, edge composition, and a full configuration snapshot.

Call generate_report() at the end of a notebook run or from FUNGI_express.py.

Usage (notebook):
    from report import generate_report
    generate_report(
        run_name        = RUN_NAME,
        cfg             = cfg,
        adata           = adata,
        graph_df        = graph_df,          # champion edge list DataFrame
        cohort          = cohort,
        utopian_bounds  = utopian_bounds,
        loss_weights    = loss_weights,
        diagnostic_report = diagnostic_report,
        experimental_df = experimental_df,
        df_expansive    = df_expansive,
        df_refinement   = df_refinement,
        sources_arr     = sources_arr,
        targets_arr     = targets_arr,
        W_arr           = W_arr,
        source_pert_impact = source_pert_impact,
        chi_prior       = chi_prior,
        rho_prior       = rho_prior,
        total_gate      = total_gate,
        er_diagnostics  = er_diagnostics,    # None if SCBER disabled
        pert_efficiency_map = pert_efficiency_map,
        alpha_md        = alpha_md,
        alpha_stab      = alpha_stab,
        output_dir      = OUTPUT_ROOT / 'phase6_champion',
    )
"""

import gc
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# DEG reachability (self-contained, no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deg_reachability(adata, graph_df, cfg):
    """
    Computes per-perturbation DEG reachability against the champion graph.

    For each perturbation gene g, identifies its statistically significant
    DEGs via Wilcoxon rank-sum test, then measures what fraction of those
    DEGs are reachable via directed paths from g in the champion graph.

    Returns a DataFrame with one row per perturbation gene and a summary dict.
    """
    pert_col   = cfg["input"]["perturbation_column"]
    ctrl_label = cfg["input"]["control_label"]
    pval_thr   = cfg["diagnostics"].get("de_pval_threshold", 0.05)
    lfc_thr_raw = cfg["diagnostics"].get("de_lfc_threshold", 0.25)

    is_metacell = cfg["input"].get("is_metacell", False)
    mc_pool     = cfg["input"].get("metacell_pooling_factor", None)
    if is_metacell and mc_pool and mc_pool > 1:
        lfc_thr = lfc_thr_raw / np.sqrt(mc_pool)
    else:
        lfc_thr = lfc_thr_raw

    # Run Wilcoxon DE
    adata_sub = adata.copy()
    sc.tl.rank_genes_groups(
        adata_sub, groupby=pert_col, reference=ctrl_label,
        method="wilcoxon", use_raw=False,
        n_jobs=cfg["diagnostics"].get("n_jobs", 8))

    pert_genes = [g for g in adata.obs[pert_col].unique() if g != ctrl_label]
    gene_set   = set(adata.var_names)

    # Build reachability index from graph
    graph_genes = set(graph_df["Regulator"]) | set(graph_df["Target"])
    adj = {}
    for _, row in graph_df.iterrows():
        adj.setdefault(row["Regulator"], set()).add(row["Target"])

    def _bfs_reach(start, adj):
        visited = set()
        queue   = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for nb in adj.get(node, []):
                if nb not in visited:
                    queue.append(nb)
        visited.discard(start)
        return visited

    records = []
    total_degs_global = 0
    total_reachable_global = 0

    for gene in pert_genes:
        in_graph = gene in graph_genes and gene in adj

        try:
            df_de = sc.get.rank_genes_groups_df(adata_sub, group=gene)
            sig = df_de[
                (df_de["pvals_adj"] < pval_thr) &
                (df_de["logfoldchanges"].abs() > lfc_thr)
            ]
            deg_set = set(sig["names"].values) & gene_set
        except Exception:
            deg_set = set()

        n_degs = len(deg_set)
        total_degs_global += n_degs

        if not in_graph or n_degs == 0:
            records.append({
                "perturbation":        gene,
                "in_graph":            in_graph,
                "n_degs":              n_degs,
                "n_reachable":         0,
                "fraction_reachable":  0.0,
            })
            continue

        reachable = _bfs_reach(gene, adj)
        n_reach   = len(deg_set & reachable)
        total_reachable_global += n_reach

        records.append({
            "perturbation":        gene,
            "in_graph":            True,
            "n_degs":              n_degs,
            "n_reachable":         n_reach,
            "fraction_reachable":  n_reach / n_degs if n_degs > 0 else 0.0,
        })

    df_reach = pd.DataFrame(records).sort_values(
        "fraction_reachable", ascending=False).reset_index(drop=True)

    in_graph_mask = df_reach["in_graph"]
    in_graph_df   = df_reach[in_graph_mask & (df_reach["n_degs"] > 0)]
    n_in           = int(in_graph_mask.sum())
    n_out          = len(df_reach) - n_in
    mean_reach     = float(in_graph_df["fraction_reachable"].mean()) if len(in_graph_df) > 0 else 0.0
    median_reach   = float(in_graph_df["fraction_reachable"].median()) if len(in_graph_df) > 0 else 0.0

    total_degs_in_graph = int(df_reach[in_graph_mask]["n_degs"].sum())

    summary = {
        "n_in_graph":            n_in,
        "n_not_in_graph":        n_out,
        "mean_reachability":     mean_reach,
        "median_reachability":   median_reach,
        "total_degs":            int(df_reach["n_degs"].sum()),
        "total_degs_in_graph":   total_degs_in_graph,
        "total_reachable":       total_reachable_global,
        "global_reachability":   (total_reachable_global / total_degs_in_graph
                                  if total_degs_in_graph > 0 else 0.0),
        "lfc_threshold_used":    round(lfc_thr, 4),
    }

    del adata_sub
    gc.collect()
    return df_reach, summary


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hr(char="─", width=72):
    return char * width

def _section(title):
    bar = "═" * 72
    return f"\n{bar}\n  {title}\n{bar}\n"

def _subsection(title):
    return f"\n{'─' * 72}\n  {title}\n{'─' * 72}\n"

def _bar(value, max_value, width=30, char="█"):
    n = int(round(value / max(max_value, 1) * width))
    return char * n + "░" * (width - n)

def _pct(value, total):
    if total == 0:
        return "—"
    return f"{value / total * 100:.1f}%"

def _tick(inside):
    return "✓" if inside else "✗"

def _fmt_bounds(lo, hi):
    return f"[{lo:.4f}, {hi:.4f}]"

def _reachability_histogram(df_in_graph, bins=10):
    lines = []
    edges = np.linspace(0, 1, bins + 1)
    counts, _ = np.histogram(df_in_graph["fraction_reachable"].values, bins=edges)
    max_count = max(counts.max(), 1)
    for i in range(bins):
        lo_b = edges[i]
        hi_b = edges[i + 1]
        n    = counts[i]
        bar  = "█" * int(round(n / max_count * 30))
        lines.append(f"  {lo_b:.1f}–{hi_b:.1f}  {bar:<30s}  {n:4d}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    run_name,
    cfg,
    adata,
    graph_df,
    cohort,
    utopian_bounds,
    loss_weights,
    diagnostic_report,
    experimental_df       = None,
    df_expansive          = None,
    df_refinement         = None,
    sources_arr           = None,
    targets_arr           = None,
    W_arr                 = None,
    source_pert_impact    = None,
    chi_prior             = None,
    rho_prior             = None,
    total_gate            = None,
    er_diagnostics        = None,
    pert_efficiency_map   = None,
    alpha_md              = 0.5,
    alpha_stab            = 0.3,
    output_dir            = None,
    graphs_to_report      = None,   # list of cohort_rank ints; None = champion only
):
    """
    Generate a full diagnostic report for a completed FUNGI run.

    Writes {run_name}_report.md to output_dir (default: phase6_champion/).
    Returns the report as a string.
    """
    if output_dir is None:
        output_dir = Path("data/output/phase6_champion")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if graphs_to_report is None:
        graphs_to_report = [1]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    W = lines.append   # shorthand

    # ── Cover ─────────────────────────────────────────────────────────────────
    W(f"# FUNGI Diagnostic Report")
    W(f"\n**Run name:** `{run_name}`  ")
    W(f"**Generated:** {timestamp}  ")
    W(f"**Config:** `{cfg.get('_config_path', 'fungi_config.yaml')}`")
    W(f"\n---\n")

    # ── 1. Run Identity ───────────────────────────────────────────────────────
    W(_section("1 · Run Identity"))

    champion_row = cohort[cohort["is_champion"]].iloc[0]
    n_genes      = int(diagnostic_report.get("N_GENES", len(adata.var_names)
                                             if adata is not None else 0))
    W(f"| Field | Value |")
    W(f"|---|---|")
    W(f"| Run name | `{run_name}` |")
    W(f"| Timestamp | {timestamp} |")
    W(f"| Input graph | `{Path(cfg['input']['graph_path']).name}` |")
    W(f"| Expression data | `{Path(cfg['input']['sc_data_path']).name}` |")
    W(f"| Genes (HVGs) | {n_genes:,} |")
    if adata is not None:
        W(f"| Cells / metacells | {adata.n_obs:,} |")
    W(f"| Is metacell | {cfg['input'].get('is_metacell', False)} |")
    if cfg['input'].get('is_metacell') and cfg['input'].get('metacell_pooling_factor'):
        W(f"| Metacell pooling factor | {cfg['input']['metacell_pooling_factor']} |")
    W(f"| Perturbation column | `{cfg['input']['perturbation_column']}` |")
    W(f"| Control label | `{cfg['input']['control_label']}` |")
    W(f"| Champion utopia loss | {champion_row['utopia_loss']:.6f} |")
    W(f"| Champion edges | {int(champion_row['n_edges']):,} |")

    # ── 2. Input Data Summary ─────────────────────────────────────────────────
    W(_section("2 · Input Data Summary"))

    if adata is not None:
        pert_col   = cfg["input"]["perturbation_column"]
        ctrl_label = cfg["input"]["control_label"]
        all_groups = adata.obs[pert_col].unique()
        n_ctrl     = int((adata.obs[pert_col] == ctrl_label).sum())
        n_pert_cells = int((adata.obs[pert_col] != ctrl_label).sum())
        n_perts    = int(sum(1 for g in all_groups if g != ctrl_label))
        W(f"**Expression matrix:** {adata.n_obs:,} cells × {adata.n_vars:,} genes\n")
        W(f"| | Count |")
        W(f"|---|---|")
        W(f"| Control cells | {n_ctrl:,} |")
        W(f"| Perturbed cells | {n_pert_cells:,} |")
        W(f"| Unique perturbation targets | {n_perts:,} |")

    W(f"\n**Parent GRN:**\n")
    if experimental_df is not None:
        exp_cols = list(experimental_df.columns[2:])
        W(f"- Experimental GRN detected")
        W(f"- Extra columns: `{', '.join(exp_cols)}`")
        n_md_edges = int((experimental_df.get('md_confidence', pd.Series([0])) > 0).sum()) \
                     if 'md_confidence' in experimental_df.columns else 0
        W(f"- Edges with MD evidence: {n_md_edges:,}")
    else:
        W(f"- Standard GRN (no experimental columns)")

    if sources_arr is not None and W_arr is not None:
        W(f"- Candidate pool after prefilter: {len(W_arr):,} edges "
          f"({cfg['prefilter']['target_density']*100:.0f}% density target)")

    # ── 3. Phase 1 Diagnostic Calibration ────────────────────────────────────
    W(_section("3 · Phase 1 — Diagnostic Calibration"))

    dr = diagnostic_report
    lam_eff  = dr.get("lam_eff", 0)
    n_active = dr.get("n_active", 0)
    n_tested = dr.get("n_tested", 0)
    W(f"**λ_center (estimated edges/gene):** {lam_eff:.2f}  ")
    W(f"**Active perturbations:** {n_active} / {n_tested} tested  ")
    if dr.get("impact_range"):
        lo_i, hi_i = dr["impact_range"]
        W(f"**DEG count range (active perts):** {lo_i:.0f} – {hi_i:.0f}  ")
    W(f"**DEG matrix edges:** {dr.get('deg_matrix_nnz', 0):,}  ")
    lfc_shape = dr.get("lfc_matrix_shape", [0, 0])
    W(f"**LFC matrix:** {lfc_shape[0]} perturbations × {lfc_shape[1]} genes  \n")

    W(f"**Utopian Bounds and Probe Confidence:**\n")
    W(f"| Parameter | Target Interval | Probe Confidence | Loss Weight | Probe Method |")
    W(f"|---|---|---|---|---|")

    probes = dr.get("probes_used", {})
    confs  = dr.get("raw_confidences", {})
    param_labels = {
        "alpha": "α (power-law exponent)",
        "gini":  "Gini (degree inequality)",
        "S_max": "S_max (hub saturation)",
        "Q":     "Q (modularity)",
        "C":     "C (clustering coefficient)",
        "rho":   "ρ (assortativity)",
    }
    for param in ["alpha", "gini", "S_max", "Q", "C", "rho"]:
        lo, hi = utopian_bounds[param]
        conf   = confs.get(param, 0.0)
        wt     = loss_weights.get(param, 0.0)
        probe  = probes.get(param, "—")
        label  = param_labels.get(param, param)
        W(f"| {label} | [{lo:.4f}, {hi:.4f}] | {conf:.2f} | {wt:.1f} | `{probe}` |")

    proceed = dr.get("proceed", True)
    W(f"\n**Diagnostic decision:** {'PROCEED ✓' if proceed else 'CAUTION — review diagnostics ⚠'}")

    # ── 4. Prefiltering Summary ───────────────────────────────────────────────
    W(_section("4 · Prefiltering Summary"))

    if sources_arr is not None and W_arr is not None:
        n_pool = len(W_arr)
        W(f"| | Value |")
        W(f"|---|---|")
        W(f"| Target density | {cfg['prefilter']['target_density']*100:.0f}% |")
        W(f"| Candidate pool size | {n_pool:,} edges |")
        if n_genes > 0:
            W(f"| Edges per gene (pool) | {n_pool / n_genes:.1f} |")
        W(f"| Weight range (pool) | [{W_arr.min():.4f}, {W_arr.max():.4f}] |")
        W(f"| Weight median | {np.median(W_arr):.4f} |")

    # ── 5. Biological Priors ──────────────────────────────────────────────────
    W(_section("5 · Biological Priors"))

    W(f"**Pleiotropy prior (χ):**\n")
    if chi_prior is not None:
        n_boosted_chi = int((chi_prior > 1.05).sum())
        W(f"| | Value |")
        W(f"|---|---|")
        W(f"| Genes boosted (χ > 1.05) | {n_boosted_chi:,} / {len(chi_prior):,} "
          f"({_pct(n_boosted_chi, len(chi_prior))}) |")
        W(f"| χ range | [{chi_prior.min():.3f}, {chi_prior.max():.3f}] |")
        W(f"| χ mean (boosted only) | {chi_prior[chi_prior > 1.05].mean():.3f} |")
    else:
        W(f"Disabled.\n")

    W(f"\n**Causal output prior (ρ):**\n")
    if rho_prior is not None:
        n_boosted_rho = int((rho_prior > 1.05).sum())
        W(f"| | Value |")
        W(f"|---|---|")
        W(f"| Genes boosted (ρ > 1.05) | {n_boosted_rho:,} / {len(rho_prior):,} "
          f"({_pct(n_boosted_rho, len(rho_prior))}) |")
        W(f"| ρ range | [{rho_prior.min():.3f}, {rho_prior.max():.3f}] |")
    else:
        W(f"Disabled.\n")

    W(f"\n**Perturbation impact prior (π):**\n")
    if source_pert_impact is not None:
        active_pi = source_pert_impact[source_pert_impact > 1.0]
        floor_pi  = source_pert_impact[source_pert_impact == 1.0]
        W(f"| | Value |")
        W(f"|---|---|")
        W(f"| π range | [{source_pert_impact.min():.4f}, {source_pert_impact.max():.4f}] |")
        W(f"| Genes at floor (π = 1.0) | {len(floor_pi):,} |")
        W(f"| Genes above floor (π > 1.0) | {len(active_pi):,} |")
        W(f"| Mean π (above floor) | {active_pi.mean():.4f} |" if len(active_pi) > 0 else "")

    W(f"\n**Experimental gate:**\n")
    if total_gate is not None and np.any(total_gate != 1.0):
        n_above = int((total_gate > 1.0).sum())
        n_below = int((total_gate < 1.0).sum())
        n_neutral = int((total_gate == 1.0).sum())
        W(f"| | Value |")
        W(f"|---|---|")
        W(f"| Edges boosted (gate > 1.0) | {n_above:,} ({_pct(n_above, len(total_gate))}) |")
        W(f"| Edges penalised (gate < 1.0) | {n_below:,} ({_pct(n_below, len(total_gate))}) |")
        W(f"| Edges neutral (gate = 1.0) | {n_neutral:,} ({_pct(n_neutral, len(total_gate))}) |")
        W(f"| Gate range | [{total_gate.min():.3f}, {total_gate.max():.3f}] |")
        W(f"| alpha_md | {alpha_md} |")
        W(f"| alpha_stab | {alpha_stab} |")
        if pert_efficiency_map:
            W(f"| Genes with efficiency data | {len(pert_efficiency_map):,} |")
    else:
        W(f"Identity gate — no experimental columns or gate inactive.\n")

    # ── 6. SCBER Structural Scoring ───────────────────────────────────────────
    W(_section("6 · SCBER — Structural Scoring"))

    if er_diagnostics and er_diagnostics.get("mode") == "scber":
        W(f"| | Value |")
        W(f"|---|---|")
        W(f"| Mode | SCBER (Source-Conditioned Bridge ER) |")
        W(f"| Communities detected | {er_diagnostics['n_communities']} |")
        W(f"| Modularity achieved (Q) | {er_diagnostics['Q_achieved']:.3f} |")
        W(f"| Inter-module edges | {er_diagnostics['n_inter']:,} "
          f"({er_diagnostics['frac_inter']*100:.1f}%) |")
        W(f"| Intra-module edges | {er_diagnostics['n_intra']:,} |")
        W(f"| η_inter | {er_diagnostics['eta_inter']:.2f} |")
        W(f"| Mean inter-module ER score | {er_diagnostics['inter_er_mean']:.3f} |")
        W(f"| High-ER bridges (R > 0.9) | {er_diagnostics['n_high_er_bridges']:,} |")
        W(f"| Inter-module factor range | [{er_diagnostics['inter_factor_min']:.3f}, 1.000] |")
    elif er_diagnostics and er_diagnostics.get("mode") == "flat_fallback":
        W(f"SCBER ran in flat fallback mode (igraph unavailable).")
    else:
        W(f"SCBER disabled — structural boost set to identity for all edges.")

    # ── 7. Search Summary ─────────────────────────────────────────────────────
    W(_section("7 · Search Summary"))

    if df_expansive is not None:
        n_exp_total    = len(df_expansive)
        n_exp_viable   = int((df_expansive["is_shattered"] == 0).sum())
        n_exp_zero     = int((df_expansive["utopia_loss"] <= 1e-6).sum())
        best_exp_loss  = (df_expansive.loc[df_expansive["is_shattered"] == 0,
                          "utopia_loss"].min()
                          if n_exp_viable > 0 else float("inf"))
        shatter_reasons = {}
        for r in df_expansive.loc[df_expansive["is_shattered"] == 1,
                                  "shatter_reason"].dropna():
            shatter_reasons[str(r)] = shatter_reasons.get(str(r), 0) + 1

        W(f"**Phase 5 — Expansive Search:**\n")
        W(f"| | Value |")
        W(f"|---|---|")
        W(f"| Total configurations evaluated | {n_exp_total:,} |")
        W(f"| Viable (non-shattered) | {n_exp_viable:,} ({_pct(n_exp_viable, n_exp_total)}) |")
        W(f"| Shattered | {n_exp_total - n_exp_viable:,} "
          f"({_pct(n_exp_total - n_exp_viable, n_exp_total)}) |")
        W(f"| Best loss (expansive) | {best_exp_loss:.6f} |")
        W(f"| Zero-loss solutions | {n_exp_zero:,} |")
        W(f"| Sobol sample count | {cfg['expansive_search']['n_samples']:,} |")

        if shatter_reasons:
            W(f"\n**Shatter breakdown:**\n")
            W(f"| Reason | Count |")
            W(f"|---|---|")
            for reason, count in sorted(shatter_reasons.items(),
                                        key=lambda x: -x[1]):
                W(f"| `{reason}` | {count:,} |")

    if df_refinement is not None:
        n_ref_total  = len(df_refinement)
        n_ref_viable = int((df_refinement["is_shattered"] == 0).sum())
        n_ref_zero   = int((df_refinement["utopia_loss"] <= 1e-6).sum())
        best_ref_loss = (df_refinement.loc[df_refinement["is_shattered"] == 0,
                         "utopia_loss"].min()
                         if n_ref_viable > 0 else float("inf"))
        W(f"\n**Phase 7 — Refinement Search:**\n")
        W(f"| | Value |")
        W(f"|---|---|")
        W(f"| Total refinement evaluations | {n_ref_total:,} |")
        W(f"| Viable | {n_ref_viable:,} ({_pct(n_ref_viable, n_ref_total)}) |")
        W(f"| Zero-loss solutions | {n_ref_zero:,} |")
        W(f"| Best refinement loss | {best_ref_loss:.6f} |")

        if "basin_idx" in df_refinement.columns:
            n_basins = df_refinement["basin_idx"].dropna().nunique()
            W(f"| Basins explored | {int(n_basins)} |")

    # ── 8. Cohort Overview ────────────────────────────────────────────────────
    W(_section("8 · Cohort Overview"))

    W(f"| Rank | Loss | α | Gini | S_max | Q | C | ρ | β | δ | κ | k_core | λ | ψ | Edges |")
    W(f"|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, row in cohort.iterrows():
        champ_star = " ★" if row["is_champion"] else ""
        W(f"| {int(row['cohort_rank'])}{champ_star} "
          f"| {row['utopia_loss']:.4f} "
          f"| {row['alpha']:.3f} | {row['Gini']:.3f} | {row['S_max']:.3f} "
          f"| {row['Q']:.3f} | {row['C']:.3f} | {row['rho']:.3f} "
          f"| {row['beta']:.2f} | {row['delta']:.2f} | {row['kappa']:.3f} "
          f"| {row['k_core']:.1f} | {row['lambda']:.2f} | {row['psi']:.2f} "
          f"| {int(row['n_edges']):,} |")

    W(f"\n★ = Champion")

    # ── Per-graph deep dives ───────────────────────────────────────────────────
    for rank_num in graphs_to_report:
        rows = cohort[cohort["cohort_rank"] == rank_num]
        if len(rows) == 0:
            continue
        champ_row = rows.iloc[0]
        tag = "Champion" if champ_row["is_champion"] else f"Alternate {rank_num}"

        # ── 9. Champion Topology Deep-Dive ────────────────────────────────────
        W(_section(f"9 · {tag} — Topology Deep-Dive"))

        W(f"**Hyperparameters:**\n")
        W(f"| Parameter | Value |")
        W(f"|---|---|")
        for hp in ["beta", "delta", "kappa", "k_core", "lambda", "psi"]:
            W(f"| {hp} | {champ_row[hp]:.4f} |")

        W(f"\n**Topology targets:**\n")
        W(f"| Parameter | Observed | Target Interval | Status | Loss Weight |")
        W(f"|---|---|---|---|---|")
        param_map = [("alpha","alpha"), ("gini","Gini"), ("S_max","S_max"),
                     ("Q","Q"), ("C","C"), ("rho","rho")]
        n_pass = 0
        for param, col in param_map:
            lo, hi  = utopian_bounds[param]
            val     = champ_row[col]
            inside  = lo <= val <= hi
            n_pass += int(inside)
            wt      = loss_weights.get(param, 0.0)
            W(f"| {param_labels.get(param, param)} "
              f"| {val:.4f} "
              f"| {_fmt_bounds(lo, hi)} "
              f"| {_tick(inside)} "
              f"| {wt:.1f} |")
        W(f"\n**{n_pass}/6 topology targets satisfied**  ")
        W(f"**Utopia loss: {champ_row['utopia_loss']:.6f}**")

        # ── 10. Hub / Out-degree Analysis ─────────────────────────────────────
        W(_section(f"10 · {tag} — Hub & Out-degree Analysis"))

        out_deg = graph_df["Regulator"].value_counts()
        in_deg  = graph_df["Target"].value_counts()
        n_edges = len(graph_df)
        n_nodes = len(set(graph_df["Regulator"]) | set(graph_df["Target"]))

        W(f"| | Value |")
        W(f"|---|---|")
        W(f"| Total edges | {n_edges:,} |")
        W(f"| Nodes present | {n_nodes:,} |")
        W(f"| Mean out-degree | {out_deg.mean():.2f} |")
        W(f"| Median out-degree | {out_deg.median():.1f} |")
        W(f"| Max out-degree | {out_deg.max()} |")
        W(f"| Genes with ≥1 outgoing edge | {len(out_deg):,} |")
        W(f"| Genes with 0 outgoing edges | {n_genes - len(out_deg):,} |")

        # Out-degree distribution
        od_vals = out_deg.values
        pcts    = [50, 75, 90, 95, 99]
        W(f"\n**Out-degree percentiles:**\n")
        W(f"| Percentile | Out-degree |")
        W(f"|---|---|")
        for p in pcts:
            W(f"| p{p} | {np.percentile(od_vals, p):.0f} |")

        # Top 20 regulators
        W(f"\n**Top 20 regulators by out-degree:**\n")
        W(f"| Rank | Gene | Out-degree | % of all edges |")
        W(f"|---|---|---|---|")
        for rank_i, (gene, cnt) in enumerate(out_deg.head(20).items(), 1):
            W(f"| {rank_i} | `{gene}` | {cnt} | {_pct(cnt, n_edges)} |")

        # ── 11. Target / In-degree Analysis ───────────────────────────────────
        W(_section(f"11 · {tag} — Target & In-degree Analysis"))

        W(f"| | Value |")
        W(f"|---|---|")
        W(f"| Mean in-degree | {in_deg.mean():.2f} |")
        W(f"| Median in-degree | {in_deg.median():.1f} |")
        W(f"| Max in-degree | {in_deg.max()} |")
        W(f"| Genes with ≥1 incoming edge | {len(in_deg):,} |")

        W(f"\n**Top 20 targets by in-degree:**\n")
        W(f"| Rank | Gene | In-degree | % of all edges |")
        W(f"|---|---|---|---|")
        for rank_i, (gene, cnt) in enumerate(in_deg.head(20).items(), 1):
            W(f"| {rank_i} | `{gene}` | {cnt} | {_pct(cnt, n_edges)} |")

        # ── 12. Perturbation Coverage Audit ───────────────────────────────────
        W(_section(f"12 · {tag} — Perturbation Coverage Audit"))

        if adata is not None:
            pert_col   = cfg["input"]["perturbation_column"]
            ctrl_label = cfg["input"]["control_label"]
            pert_genes = [g for g in adata.obs[pert_col].unique()
                          if g != ctrl_label]
            graph_sources = set(graph_df["Regulator"])
            in_src    = [g for g in pert_genes if g in graph_sources]
            not_in    = [g for g in pert_genes if g not in graph_sources]

            W(f"| | Count |")
            W(f"|---|---|")
            W(f"| Total perturbation targets | {len(pert_genes)} |")
            W(f"| Present as source in graph | {len(in_src)} "
              f"({_pct(len(in_src), len(pert_genes))}) |")
            W(f"| Missing from graph | {len(not_in)} |")

            if not_in:
                W(f"\n**Perturbation genes absent from graph:**\n")
                W(", ".join(f"`{g}`" for g in sorted(not_in)))

        # ── 13. DEG Reachability Audit ────────────────────────────────────────
        W(_section(f"13 · {tag} — DEG Reachability Audit"))

        W(f"*Computing DEG reachability — this may take a few minutes...*\n")

        try:
            df_reach, reach_summary = _compute_deg_reachability(adata, graph_df, cfg)

            W(f"| | Value |")
            W(f"|---|---|")
            W(f"| Perturbations in graph | {reach_summary['n_in_graph']} / "
              f"{reach_summary['n_in_graph'] + reach_summary['n_not_in_graph']} |")
            W(f"| Perturbations not in graph | {reach_summary['n_not_in_graph']} |")
            W(f"| Total DEGs across all perturbations | {reach_summary['total_degs']:,} |")
            W(f"| DEGs from in-graph perturbations | "
              f"{reach_summary['total_degs_in_graph']:,} |")
            W(f"| Total DEGs reachable | {reach_summary['total_reachable']:,} |")
            W(f"| **Mean reachability** | **{reach_summary['mean_reachability']:.4f} "
              f"({reach_summary['mean_reachability']*100:.1f}%)** |")
            W(f"| **Median reachability** | **{reach_summary['median_reachability']:.4f} "
              f"({reach_summary['median_reachability']*100:.1f}%)** |")
            W(f"| **Global reachability** | **{reach_summary['global_reachability']:.4f} "
              f"({reach_summary['global_reachability']*100:.1f}%)** |")
            W(f"| LFC threshold used | {reach_summary['lfc_threshold_used']} |")

            # Reachability histogram
            in_graph_reach = df_reach[df_reach["in_graph"] & (df_reach["n_degs"] > 0)]
            W(f"\n**Reachability distribution ({len(in_graph_reach)} "
              f"in-graph perturbations):**\n")
            W("```")
            W(_reachability_histogram(in_graph_reach))
            W("```")

            # Top 15 most reachable
            W(f"\n**Top 15 most reachable perturbations:**\n")
            W(f"| Gene | In Graph | DEGs | Reachable | Fraction |")
            W(f"|---|---|---|---|---|")
            for _, r in df_reach[df_reach["in_graph"]].head(15).iterrows():
                W(f"| `{r['perturbation']}` | ✓ | {r['n_degs']} "
                  f"| {r['n_reachable']} | {r['fraction_reachable']:.3f} |")

            # Bottom 15 in-graph with zero reachability
            zero_reach = df_reach[df_reach["in_graph"] &
                                  (df_reach["fraction_reachable"] == 0.0) &
                                  (df_reach["n_degs"] > 0)]
            if len(zero_reach) > 0:
                W(f"\n**Zero-reachability in-graph perturbations ({len(zero_reach)}):**\n")
                W(f"| Gene | DEGs | Note |")
                W(f"|---|---|---|")
                for _, r in zero_reach.iterrows():
                    W(f"| `{r['perturbation']}` | {r['n_degs']} "
                      f"| DEG edges exist in parent but not selected by DASH |")

            # Full per-perturbation table
            W(f"\n<details>\n<summary>Full per-perturbation reachability table "
              f"(click to expand)</summary>\n")
            W(f"\n| Gene | In Graph | DEGs | Reachable | Fraction |")
            W(f"|---|---|---|---|---|")
            for _, r in df_reach.iterrows():
                in_g = "✓" if r["in_graph"] else "✗"
                W(f"| `{r['perturbation']}` | {in_g} | {r['n_degs']} "
                  f"| {r['n_reachable']} | {r['fraction_reachable']:.3f} |")
            W(f"\n</details>\n")

            # Save reachability CSV alongside report
            reach_path = output_dir / f"{run_name}_deg_reachability.csv"
            df_reach.to_csv(reach_path, index=False)
            W(f"\n*Full reachability table saved → `{reach_path.name}`*")

        except Exception as e:
            W(f"\n⚠ DEG reachability computation failed: `{e}`")
            W(f"  Ensure `adata` is passed and contains perturbation labels.")

        # ── 14. Edge Sign Composition ─────────────────────────────────────────
        if "Sign" in graph_df.columns:
            W(_section(f"14 · {tag} — Edge Sign Composition"))

            n_pos   = int((graph_df["Sign"] == 1).sum())
            n_neg   = int((graph_df["Sign"] == -1).sum())
            n_unsig = int((graph_df["Sign"] == 0).sum())
            n_tot   = len(graph_df)
            W(f"| Sign | Count | Fraction |")
            W(f"|---|---|---|")
            W(f"| Activating (+1) | {n_pos:,} | {_pct(n_pos, n_tot)} |")
            W(f"| Repressing (−1) | {n_neg:,} | {_pct(n_neg, n_tot)} |")
            W(f"| Unsigned (0) | {n_unsig:,} | {_pct(n_unsig, n_tot)} |")
            W(f"| **Total signed** | **{n_pos + n_neg:,}** | "
              f"**{_pct(n_pos + n_neg, n_tot)}** |")
            if n_pos + n_neg > 0:
                W(f"\n**Sign ratio (activating : repressing):** "
                  f"{n_pos / max(n_neg, 1):.2f} : 1")

        # ── 15. Edge Weight Distribution ──────────────────────────────────────
        W(_section(f"15 · {tag} — Edge Weight Distribution"))

        wts = graph_df["Weight"].values
        W(f"| Statistic | Value |")
        W(f"|---|---|")
        W(f"| Min | {wts.min():.4f} |")
        W(f"| p25 | {np.percentile(wts, 25):.4f} |")
        W(f"| Median | {np.median(wts):.4f} |")
        W(f"| p75 | {np.percentile(wts, 75):.4f} |")
        W(f"| p90 | {np.percentile(wts, 90):.4f} |")
        W(f"| p99 | {np.percentile(wts, 99):.4f} |")
        W(f"| Max | {wts.max():.4f} |")
        W(f"| Mean | {wts.mean():.4f} |")
        W(f"| Std | {wts.std():.4f} |")

    # ── 16. Alternate Graph Comparison ────────────────────────────────────────
    if len(cohort) > 1:
        W(_section("16 · Alternate Graph Comparison"))

        W(f"| Rank | Tag | Loss | Edges | α | Gini | S_max | Q | C | ρ | β | λ | ψ |")
        W(f"|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, row in cohort.iterrows():
            tag_str = "Champion ★" if row["is_champion"] else f"Alternate {int(row['cohort_rank'])}"
            n_pass_alt = sum(
                1 for param, col in [("alpha","alpha"),("gini","Gini"),("S_max","S_max"),
                                      ("Q","Q"),("C","C"),("rho","rho")]
                if utopian_bounds[param][0] <= row[col] <= utopian_bounds[param][1]
            )
            W(f"| {int(row['cohort_rank'])} | {tag_str} "
              f"| {row['utopia_loss']:.4f} | {int(row['n_edges']):,} "
              f"| {row['alpha']:.3f} | {row['Gini']:.3f} | {row['S_max']:.3f} "
              f"| {row['Q']:.3f} | {row['C']:.3f} | {row['rho']:.3f} "
              f"| {row['beta']:.2f} | {row['lambda']:.2f} | {row['psi']:.2f} |")

        W(f"\nAll five cohort graphs are selected by farthest-point sampling in "
          f"normalized topology space, ensuring diverse regulatory architectures.")

    # ── 17. Configuration Snapshot ────────────────────────────────────────────
    W(_section("17 · Configuration Snapshot"))

    W(f"```yaml")
    import yaml as _yaml
    W(_yaml.dump(
        {k: v for k, v in cfg.items() if k not in ("_config_path",)},
        default_flow_style=False, allow_unicode=True).rstrip())
    W(f"```")

    # ── 18. Warnings and Flags ────────────────────────────────────────────────
    W(_section("18 · Warnings and Flags"))

    flags = []

    if champion_row["utopia_loss"] > 1e-6:
        flags.append(f"⚠ Champion has non-zero utopia loss "
                     f"({champion_row['utopia_loss']:.4f}). "
                     f"No perfectly topology-satisfying graph was found.")

    n_pass_champ = sum(
        1 for param, col in [("alpha","alpha"),("gini","Gini"),("S_max","S_max"),
                              ("Q","Q"),("C","C"),("rho","rho")]
        if utopian_bounds[param][0] <= champion_row[col] <= utopian_bounds[param][1]
    )
    if n_pass_champ < 6:
        failing = [
            param for param, col in [("alpha","alpha"),("gini","Gini"),("S_max","S_max"),
                                      ("Q","Q"),("C","C"),("rho","rho")]
            if not (utopian_bounds[param][0] <= champion_row[col] <= utopian_bounds[param][1])
        ]
        flags.append(f"⚠ Champion fails {6 - n_pass_champ}/6 topology targets: "
                     f"{', '.join(failing)}.")

    if not dr.get("proceed", True):
        flags.append(f"⚠ Phase 1 diagnostics returned CAUTION — "
                     f"low statistical confidence in probe estimates.")

    if n_active < 30:
        flags.append(f"⚠ Only {n_active} active perturbations detected in Phase 1. "
                     f"Probe estimates may be unreliable.")

    if df_expansive is not None:
        shatter_rate = (df_expansive["is_shattered"].sum() / max(len(df_expansive), 1))
        if shatter_rate > 0.90:
            flags.append(f"⚠ Expansive search shatter rate was "
                         f"{shatter_rate*100:.1f}%. "
                         f"Consider widening hyperparameter bounds or "
                         f"relaxing shatter constraints.")

    if champion_row.get("S_max", 0) < utopian_bounds["S_max"][0] * 1.05:
        flags.append(f"⚠ Champion S_max ({champion_row['S_max']:.4f}) is near the "
                     f"lower bound ({utopian_bounds['S_max'][0]:.4f}). "
                     f"Hub structure may be under-developed.")

    if len(flags) == 0:
        W(f"✓ No warnings raised. Run completed cleanly.")
    else:
        for flag in flags:
            W(f"\n{flag}")

    W(f"\n\n---\n*FUNGI Diagnostic Report — {run_name} — {timestamp}*")

    # ── Write to disk ──────────────────────────────────────────────────────────
    report_text = "\n".join(lines)
    out_path    = output_dir / f"{run_name}_report.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)

    print(f"FUNGI report saved → {out_path}")
    return report_text
