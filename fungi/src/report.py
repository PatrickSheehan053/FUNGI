"""
FUNGI — src/report.py
─────────────────────
Post-run diagnostic report generator.

Produces a self-contained Markdown report covering every stage of a FUNGI run:
input data, diagnostic calibration, prefiltering, biological priors, structural
scoring, search trajectory, cohort overview, champion topology deep-dive,
HYPHAE graph fingerprint, hub/target analysis, perturbation coverage, DEG
reachability, edge composition, and a full configuration snapshot.

New in this version:
  - Section 9F: HYPHAE Graph Fingerprint with six thematic clusters
    (degree architecture, hub topology, community structure, edge weight
    distribution, spectral properties, perturbation routing).
  - compute_fingerprint_inline(): pure scipy/pandas structural fingerprint,
    no networkx required.  Community detection uses igraph/leidenalg if
    available (same environment as engine.py SCBER).
  - Mode-aware sections (organic vs synthetic): bounds table, cohort overview,
    topology targets table, alternate comparison, warnings.

Usage (notebook):
    from report import generate_report
    generate_report(
        run_name                 = run_tag,
        cfg                      = cfg,
        adata                    = adata,
        graph_df                 = graph_df,
        cohort                   = cohort,
        utopian_bounds           = utopian_bounds,
        loss_weights             = loss_weights,
        diagnostic_report        = diagnostic_report,
        experimental_df          = None,
        df_expansive             = df_expansive,
        df_refinement            = df_refinement,
        sources_arr              = evaluator.srcs,
        targets_arr              = evaluator.tgts,
        W_arr                    = W_arr,
        source_pert_impact       = evaluator.source_pert_impact,
        chi_prior                = evaluator.chi_prior,
        rho_prior                = evaluator.rho_prior,
        total_gate               = None,
        er_diagnostics           = scber_diag,
        pert_efficiency_map      = None,
        alpha_md                 = 0.5,
        alpha_stab               = 0.3,
        output_dir               = Path("../DATA/FUNGI_outputs/reports"),
        graphs_to_report         = [1],
        fungi_mode               = FUNGI_MODE,
        fingerprint              = None,           # pass pre-computed HYPHAE JSON here
        champ_secondary_metrics  = _cohort_sec.get(1, {}),
        cohort_secondary_metrics = _cohort_sec,
    )
"""

import gc
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as _stats
import scanpy as sc

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# DEG reachability (self-contained, no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_deg_reachability(adata, graph_df, cfg):
    """
    Per-perturbation DEG reachability against the champion graph.

    Runs Wilcoxon DE per perturbation, then BFS-checks what fraction of each
    perturbation's significant DEGs are reachable from that gene in the graph.
    Returns (df_reach, summary).
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

    adata_sub = adata.copy()
    sc.tl.rank_genes_groups(
        adata_sub, groupby=pert_col, reference=ctrl_label,
        method="wilcoxon", use_raw=False,
        n_jobs=cfg["diagnostics"].get("n_jobs", 6))

    pert_genes = [g for g in adata.obs[pert_col].unique() if g != ctrl_label]
    gene_set   = set(adata.var_names)

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
# Fingerprint helpers — adapted from HYPHAE_fingerprint.ipynb
# Pure scipy/pandas; community detection uses igraph/leidenalg if available
# ─────────────────────────────────────────────────────────────────────────────

def _fp_gini(arr):
    a = np.sort(np.abs(np.asarray(arr, dtype=float)))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum() - (n + 1) * a.sum()) / (n * a.sum()))


def _fp_entropy(values, n_bins=50):
    counts, _ = np.histogram(values, bins=max(n_bins, 2))
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _fp_topology_metrics(graph_df):
    """Degree distribution, power-law fit, and degree assortativity."""
    out_deg = graph_df["Regulator"].value_counts()
    in_deg  = graph_df["Target"].value_counts()
    od = out_deg.values.astype(float)

    n_edges   = len(graph_df)
    n_nodes   = len(set(graph_df["Regulator"]) | set(graph_df["Target"]))
    n_sources = len(out_deg)

    # Hill MLE above median
    k_min = max(1, int(np.percentile(od, 50)))
    tail  = od[od >= k_min]
    alpha_mle = None
    w1 = None
    if len(tail) > 2:
        alpha_mle = float(1 + len(tail) / np.sum(np.log(tail / (k_min - 0.5))))
        if alpha_mle > 1:
            rng = np.random.default_rng(42)
            u   = rng.uniform(0, 1, size=len(od))
            ref = k_min * (1 - u) ** (-1 / (alpha_mle - 1))
            w1  = float(_stats.wasserstein_distance(od, ref))

    # Assortativity: Pearson(out-deg of source, in-deg of target) across all edges
    src_d = graph_df["Regulator"].map(out_deg).fillna(0).values.astype(float)
    tgt_d = graph_df["Target"].map(in_deg).fillna(0).values.astype(float)
    rho = None
    try:
        r, _ = _stats.pearsonr(src_d, tgt_d)
        rho  = float(r) if not np.isnan(r) else None
    except Exception:
        pass

    return dict(
        n_edges               = n_edges,
        n_nodes               = n_nodes,
        n_sources             = n_sources,
        mean_out_degree       = round(float(od.mean()), 3),
        median_out_degree     = round(float(np.median(od)), 3),
        max_out_degree        = int(od.max()),
        std_out_degree        = round(float(od.std()), 3),
        degree_gini           = round(_fp_gini(od), 4),
        degree_entropy        = round(
            _fp_entropy(od, n_bins=min(50, max(5, len(np.unique(od))))), 4),
        alpha_powerlaw        = round(alpha_mle, 4) if alpha_mle is not None else None,
        w1_degree_vs_powerlaw = round(w1, 3) if w1 is not None else None,
        rho_assortativity     = round(rho, 4) if rho is not None else None,
    )


def _fp_weight_metrics(graph_df):
    """Edge weight distribution and FAGCN attention signal quality."""
    if "Weight" not in graph_df.columns:
        return {}
    w = graph_df["Weight"].values.astype(float)
    w = w[np.isfinite(w)]
    if len(w) == 0:
        return {}
    w_min, w_max = float(w.min()), float(w.max())
    uniform      = np.random.default_rng(42).uniform(w_min, w_max, size=len(w))
    top10_thr    = np.percentile(w, 90)
    w_conc       = float(w[w >= top10_thr].sum() / w.sum()) if w.sum() > 0 else None
    return dict(
        weight_mean                = round(float(w.mean()), 4),
        weight_median              = round(float(np.median(w)), 4),
        weight_std                 = round(float(w.std()), 4),
        weight_gini                = round(_fp_gini(w), 4),
        weight_entropy             = round(_fp_entropy(w, n_bins=50), 4),
        w1_weights_vs_uniform      = round(float(_stats.wasserstein_distance(w, uniform)), 4),
        weight_concentration_top10 = round(w_conc, 4) if w_conc is not None else None,
    )


def _fp_community_metrics(graph_df):
    """
    Leiden modularity, community count, and heterophily.

    Requires igraph + leidenalg (same as engine.py SCBER).  Returns all None
    fields if unavailable — the rest of the fingerprint is unaffected.
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        return {"modularity_Q": None, "n_communities": None,
                "community_entropy": None, "heterophily": None, "homophily": None}

    nodes = sorted(set(graph_df["Regulator"]) | set(graph_df["Target"]))
    n2i   = {n: i for i, n in enumerate(nodes)}
    edges = [(n2i[r], n2i[t])
             for r, t in zip(graph_df["Regulator"], graph_df["Target"])]

    g    = ig.Graph(n=len(nodes), edges=edges, directed=False)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=42)
    mem  = np.array(part.membership)
    q    = float(part.modularity)
    n_comm = int(len(np.unique(mem)))

    sizes = np.bincount(mem).astype(float)
    ce    = _fp_entropy(sizes, n_bins=max(5, n_comm))

    mem_map = {nodes[i]: int(mem[i]) for i in range(len(nodes))}
    cross   = sum(
        1 for r, t in zip(graph_df["Regulator"], graph_df["Target"])
        if mem_map.get(r, -1) != mem_map.get(t, -2)
    )
    hetero = cross / max(len(graph_df), 1)

    return dict(
        modularity_Q      = round(q, 4),
        n_communities     = n_comm,
        community_entropy = round(ce, 4),
        heterophily       = round(hetero, 4),
        homophily         = round(1 - hetero, 4),
    )


def _fp_hub_metrics(graph_df, pert_targets=None):
    """Hub saturation and perturbation coverage."""
    out_deg = graph_df["Regulator"].value_counts()
    od      = out_deg.values.astype(float)
    n_edges = len(graph_df)

    if n_edges == 0 or len(od) == 0:
        return {"S_max": None, "hub_concentration_p99": None,
                "n_perturbed_in_graph": None,
                "pct_perturbed_in_graph": None,
                "perturbation_hub_frac": None}

    s_max    = float(od.max()) / n_edges
    thr99    = np.percentile(od, 99)
    hub_conc = float(od[od >= thr99].sum()) / n_edges

    result = dict(
        S_max                 = round(s_max, 5),
        hub_concentration_p99 = round(hub_conc, 4),
    )

    if pert_targets is not None:
        in_graph = [p for p in pert_targets
                    if p in out_deg.index and out_deg[p] > 0]
        n_in     = len(in_graph)
        pct      = n_in / len(pert_targets) if pert_targets else 0.0
        pert_hub = sum(1 for p in in_graph if out_deg.get(p, 0) >= thr99) / max(1, n_in)
        result.update(
            n_perturbed_in_graph   = n_in,
            pct_perturbed_in_graph = round(pct, 4),
            perturbation_hub_frac  = round(pert_hub, 4),
        )
    else:
        result.update(
            n_perturbed_in_graph   = None,
            pct_perturbed_in_graph = None,
            perturbation_hub_frac  = None,
        )

    return result


def _fp_spectral_metrics(graph_df, k=6):
    """
    Approximate spectral gap and entropy via ARPACK on the symmetrised graph.

    spectral_gap (lambda_2 of normalised Laplacian): controls signal mixing speed.
    spectral_entropy: Shannon entropy of top-k eigenvalues (frequency richness).
    """
    from scipy.sparse.linalg import eigsh

    nodes = sorted(set(graph_df["Regulator"]) | set(graph_df["Target"]))
    n     = len(nodes)
    if n < 10:
        return {"spectral_gap": None, "spectral_entropy": None}

    n2i   = {nd: i for i, nd in enumerate(nodes)}
    rows_, cols_, wts_ = [], [], []
    for r, t, w in zip(graph_df["Regulator"], graph_df["Target"],
                       graph_df["Weight"].values.astype(float)):
        i, j = n2i[r], n2i[t]
        rows_ += [i, j]; cols_ += [j, i]; wts_ += [float(w), float(w)]

    A  = sp.csr_matrix((wts_, (rows_, cols_)), shape=(n, n))
    dv = np.array(A.sum(axis=1)).ravel()
    dv[dv == 0] = 1.0
    L = (sp.eye(n, format="csr")
         - sp.diags(1.0 / np.sqrt(dv)) @ A @ sp.diags(1.0 / np.sqrt(dv)))

    try:
        k_act = min(k + 1, n - 2)
        eigs  = np.sort(np.abs(eigsh(L, k=k_act, which="SM",
                                      return_eigenvectors=False,
                                      tol=1e-4, maxiter=1000)))
        gap   = float(eigs[1]) if len(eigs) > 1 else None
        ep    = eigs[eigs > 1e-10]
        s_ent = _fp_entropy(ep, n_bins=max(3, len(ep)))
    except Exception:
        gap, s_ent = None, None

    return dict(
        spectral_gap     = round(gap, 5) if gap is not None else None,
        spectral_entropy = round(s_ent, 4) if s_ent is not None else None,
    )


def compute_fingerprint_inline(graph_df, pert_targets=None):
    """
    Compute a structural graph fingerprint from a FUNGI output DataFrame.

    Covers topology, weight distribution, community structure, hub metrics,
    and spectral properties.  Routing metrics (EPR@k, reachability, path
    length) are NOT computed here — they require expression data.  Pass a
    pre-computed HYPHAE fingerprint dict via generate_report(fingerprint=...)
    to include those in the report's Section 9F.

    Args:
        graph_df: DataFrame with columns Regulator, Target, Weight.
        pert_targets: list of CRISPRi target gene names (enables hub coverage).

    Returns:
        Flat dict matching the HYPHAE registry schema.
    """
    fp = {}
    fp.update(_fp_topology_metrics(graph_df))
    fp.update(_fp_weight_metrics(graph_df))
    fp.update(_fp_community_metrics(graph_df))
    fp.update(_fp_hub_metrics(graph_df, pert_targets=pert_targets))
    fp.update(_fp_spectral_metrics(graph_df))
    for key in ("epr_k", "reachability_mean", "reachability_median",
                "mean_path_to_deg", "median_path_to_deg", "n_in_graph_perts"):
        fp.setdefault(key, None)
    fp["computed_at"] = datetime.now().isoformat()
    return fp


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
# Fingerprint section renderer
# ─────────────────────────────────────────────────────────────────────────────

def _fv(fp, key, fmt=".4f"):
    """Format a fingerprint value; return '—' if None or missing."""
    v = fp.get(key)
    if v is None:
        return "—"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)

def _fi(fp, key):
    """Format a fingerprint integer count with comma separators."""
    v = fp.get(key)
    return f"{int(v):,}" if v is not None else "—"


def _render_fingerprint_section(lines, fp, tag):
    """
    Render Section 9F: HYPHAE Graph Fingerprint.

    Six thematic clusters, each with a brief interpretive note:
    A) Scale & Degree Architecture
    B) Hub Topology
    C) Community & Modularity
    D) Edge Weight Distribution (FAGCN attention signal)
    E) Spectral Properties
    F) Perturbation Routing (routing metrics only if pre-computed)
    """
    W = lines.append

    W(_section(f"9F · {tag} — HYPHAE Graph Fingerprint"))
    W(
        "This fingerprint characterises the champion graph across six biological and "
        "computational dimensions, clustered to answer the question: *is this graph "
        "shaped correctly to support FAGCN-based perturbation prediction?*  "
        "Routing metrics in Cluster F require a pre-computed HYPHAE fingerprint "
        "(pass `fingerprint=json.load(open(cache))` to `generate_report`).\n"
    )

    # ── A: Scale & Degree Architecture ────────────────────────────────────────
    W(_subsection("A · Scale & Degree Architecture"))

    alpha_v = fp.get("alpha_powerlaw")
    if alpha_v is None:
        alpha_note = "—"
    elif alpha_v < 2.0:
        alpha_note = "⚠ super-hub dominated (α < 2)"
    elif alpha_v <= 3.0:
        alpha_note = "✓ Barabási criterion satisfied [2, 3]"
    else:
        alpha_note = "⚠ near-uniform topology (α > 3)"

    rho_v = fp.get("rho_assortativity")
    if rho_v is None:
        rho_note = "—"
    elif rho_v < -0.05:
        rho_note = "✓ disassortative — hub-to-target pattern expected"
    elif rho_v <= 0.05:
        rho_note = "neutral"
    else:
        rho_note = "⚠ assortative — hub-to-hub edges (unusual for GRN)"

    gini_v = fp.get("degree_gini")
    if gini_v is None:
        gini_note = "—"
    elif gini_v < 0.35:
        gini_note = "low inequality"
    elif gini_v < 0.65:
        gini_note = "moderate inequality"
    else:
        gini_note = "high inequality"

    W("*A scale-free GRN satisfies the Barabási criterion (α ∈ [2, 3]): a heavy-tailed "
      "out-degree distribution with a small number of transcription factor hubs. "
      "Negative assortativity is expected because hub TFs connect to low-degree target "
      "genes rather than to other hubs.*\n")
    W("| Metric | Value | Interpretation |")
    W("|---|---|---|")
    W(f"| Total edges | {_fi(fp, 'n_edges')} | |")
    W(f"| Nodes present | {_fi(fp, 'n_nodes')} | |")
    W(f"| Source genes | {_fi(fp, 'n_sources')} | |")
    W(f"| Mean out-degree | {_fv(fp, 'mean_out_degree', '.2f')} | |")
    W(f"| Median out-degree | {_fv(fp, 'median_out_degree', '.1f')} | |")
    W(f"| Max out-degree | {fp.get('max_out_degree', '—')} | |")
    W(f"| Std out-degree | {_fv(fp, 'std_out_degree', '.3f')} | |")
    W(f"| α power-law exponent | {_fv(fp, 'alpha_powerlaw')} | {alpha_note} |")
    W(f"| Degree Gini | {_fv(fp, 'degree_gini')} | {gini_note} |")
    W(f"| Degree entropy | {_fv(fp, 'degree_entropy')} bits | |")
    W(f"| ρ assortativity | {_fv(fp, 'rho_assortativity')} | {rho_note} |")
    W(f"| W₁ degree vs power-law | {_fv(fp, 'w1_degree_vs_powerlaw', '.3f')} "
      f"| smaller = closer to ideal scale-free |")

    # ── B: Hub Topology ────────────────────────────────────────────────────────
    W(_subsection("B · Hub Topology"))

    smax_v = fp.get("S_max")
    if smax_v is None:
        smax_note = "—"
    elif smax_v < 0.005:
        smax_note = "✓ distributed — no single gene monopolises"
    elif smax_v < 0.02:
        smax_note = "moderate hub concentration"
    else:
        smax_note = "⚠ single gene accounts for >2% of all edges"

    pct_v = fp.get("pct_perturbed_in_graph")
    if pct_v is None:
        pct_note = "—"
    elif pct_v < 0.70:
        pct_note = "⚠ sparse — many perturbed TFs excluded from graph"
    elif pct_v < 0.85:
        pct_note = "adequate coverage"
    else:
        pct_note = "✓ excellent — most perturbed TFs retained as sources"

    W("*S_max measures whether one gene dominates all outgoing regulation. "
      "Hub concentration tracks the top-1% of sources collectively. "
      "Perturbation coverage confirms that the CRISPRi targets are present as "
      "source nodes — if they are absent, SPECTRA cannot route their signals.*\n")
    W("| Metric | Value | Interpretation |")
    W("|---|---|---|")
    W(f"| S_max (top-hub edge fraction) | {_fv(fp, 'S_max', '.5f')} | {smax_note} |")
    W(f"| Hub concentration (top 1%) | {_fv(fp, 'hub_concentration_p99', '.4f')} "
      f"| fraction of edges from top-1% of sources |")

    n_pert = fp.get("n_perturbed_in_graph")
    pct_pert = fp.get("pct_perturbed_in_graph")
    ph = fp.get("perturbation_hub_frac")
    if n_pert is not None:
        pct_str = f"{pct_pert*100:.1f}%" if pct_pert is not None else "—"
        ph_str  = f"{ph*100:.1f}% are top-1% hubs" if ph is not None else "—"
        W(f"| CRISPRi targets in graph | {n_pert} ({pct_str}) | {pct_note} |")
        W(f"| Perturbation hub fraction | {_fv(fp, 'perturbation_hub_frac', '.4f')} "
          f"| {ph_str} |")
    else:
        W(f"| CRISPRi targets in graph | — | adata not passed to fingerprint |")
        W(f"| Perturbation hub fraction | — | |")

    # ── C: Community & Modularity ──────────────────────────────────────────────
    W(_subsection("C · Community Structure & Modularity"))

    q_v = fp.get("modularity_Q")
    if q_v is None:
        q_note = "— (requires igraph/leidenalg)"
    elif q_v < 0.15:
        q_note = "⚠ weak — no meaningful community partition found"
    elif q_v < 0.30:
        q_note = "moderate community structure"
    else:
        q_note = "✓ well-modular — clear regulatory module separation"

    hetero_v = fp.get("heterophily")
    if hetero_v is None:
        hetero_note = "— (requires igraph/leidenalg)"
    elif hetero_v < 0.40:
        hetero_note = "⚠ homophilic — FAGCN attention benefit reduced"
    elif hetero_v < 0.65:
        hetero_note = "✓ mixed heterophily — FAGCN-compatible"
    else:
        hetero_note = "✓ strongly heterophilic — FAGCN designed for this"

    W("*Heterophily measures how often edges cross community boundaries. "
      "FAGCN was explicitly designed for heterophilic graphs (different-type node pairs). "
      "A GRN where TF nodes connect to their non-TF targets across module boundaries is "
      "naturally heterophilic. Q > 0.15 confirms meaningful community structure exists.*\n")
    W("| Metric | Value | Interpretation |")
    W("|---|---|---|")
    W(f"| Modularity Q | {_fv(fp, 'modularity_Q')} | {q_note} |")
    W(f"| Communities (Leiden) | {fp.get('n_communities', '—')} | |")
    W(f"| Community entropy | {_fv(fp, 'community_entropy')} bits "
      f"| higher = more balanced community sizes |")
    W(f"| Heterophily | {_fv(fp, 'heterophily')} | {hetero_note} |")
    W(f"| Homophily | {_fv(fp, 'homophily')} | |")

    # ── D: Edge Weight Distribution ────────────────────────────────────────────
    W(_subsection("D · Edge Weight Distribution — FAGCN Attention Signal"))

    we_v = fp.get("weight_entropy")
    if we_v is None:
        we_note = "—"
    elif we_v < 2.5:
        we_note = "⚠ near-binary — FAGCN attention impaired (all edges similar weight)"
    elif we_v < 4.0:
        we_note = "adequate signal for attention learning"
    else:
        we_note = "✓ rich distribution — strong FAGCN attention gradients"

    wg_v = fp.get("weight_gini")
    if wg_v is None:
        wg_note = "—"
    elif wg_v < 0.30:
        wg_note = "✓ well-spread — strong attention differentiation"
    elif wg_v < 0.60:
        wg_note = "moderate concentration"
    else:
        wg_note = "⚠ highly concentrated — few edges carry most weight mass"

    W("*FAGCN uses edge weights as adaptive attention coefficients. When weights cluster "
      "near 1.0 (near-binary), every edge counts equally and FAGCN cannot learn which "
      "regulatory connections matter most. Higher weight entropy = richer attention "
      "landscape = better perturbation prediction potential.*\n")
    W("| Metric | Value | Interpretation |")
    W("|---|---|---|")
    W(f"| Weight mean | {_fv(fp, 'weight_mean')} | |")
    W(f"| Weight median | {_fv(fp, 'weight_median')} | |")
    W(f"| Weight std | {_fv(fp, 'weight_std')} | |")
    W(f"| Weight entropy | {_fv(fp, 'weight_entropy')} bits | {we_note} |")
    W(f"| Weight Gini | {_fv(fp, 'weight_gini')} | {wg_note} |")
    W(f"| W₁ vs uniform | {_fv(fp, 'w1_weights_vs_uniform')} "
      f"| 0 = perfectly uniform distribution |")
    W(f"| Top-10% weight fraction | {_fv(fp, 'weight_concentration_top10')} "
      f"| fraction of total weight in heaviest 10% of edges |")

    # ── E: Spectral Properties ─────────────────────────────────────────────────
    W(_subsection("E · Spectral Properties — Signal Propagation Speed"))

    sg_v = fp.get("spectral_gap")
    if sg_v is None:
        sg_note = "—"
    elif sg_v < 0.02:
        sg_note = "⚠ slow mixing — graph may have near-disconnected components"
    elif sg_v < 0.10:
        sg_note = "moderate propagation speed"
    else:
        sg_note = "✓ fast signal propagation across the graph"

    W("*The spectral gap (λ₂ of the normalised Laplacian) controls how quickly "
      "perturbation signals spread through the graph during GNN message-passing. "
      "A large gap means fewer hops are needed for a signal to reach distant targets, "
      "which matters because FAGCN in SPECTRA has depth L=3 (hard propagation ceiling).*\n")
    W("| Metric | Value | Interpretation |")
    W("|---|---|---|")
    W(f"| Spectral gap (λ₂) | {_fv(fp, 'spectral_gap', '.5f')} | {sg_note} |")
    W(f"| Spectral entropy | {_fv(fp, 'spectral_entropy')} bits "
      f"| higher = richer frequency content for FAGCN |")

    # ── F: Perturbation Routing ────────────────────────────────────────────────
    W(_subsection("F · Perturbation Routing"))

    epr_v = fp.get("epr_k")
    if epr_v is None:
        W("*Routing metrics require a pre-computed HYPHAE fingerprint. "
          "Run `HYPHAE_fingerprint.ipynb` on this graph and pass the cached JSON "
          "via `generate_report(fingerprint=json.load(open(cache_path)))`. "
          "See Section 13 (DEG Reachability Audit) for BFS-based reachability "
          "computed directly here.*\n")
    else:
        epr_note = (
            "⚠ low precision"   if epr_v < 0.20 else
            "adequate"          if epr_v < 0.35 else
            "✓ strong edge-to-DEG precision"
        )
        path_v = fp.get("mean_path_to_deg")
        if path_v is None:
            path_note = "—"
        elif path_v < 2.0:
            path_note = "✓ very direct routing"
        elif path_v <= 3.0:
            path_note = "✓ within FAGCN propagation depth (L=3)"
        else:
            path_note = "⚠ exceeds FAGCN depth limit (L=3) — signal may be lost"

        W("*EPR@k (Expected Precision at k): for each perturbed TF, what fraction of "
          "its top-k outgoing edges lead to a known DEG for that perturbation? "
          "Mean path length to DEG is compared against the FAGCN depth L=3 — paths "
          "longer than 3 hops cannot be learned by SPECTRA.*\n")
        W("| Metric | Value | Interpretation |")
        W("|---|---|---|")
        W(f"| EPR@k | {epr_v:.4f} | {epr_note} |")
        W(f"| Mean DEG reachability | {_fv(fp, 'reachability_mean')} "
          f"| fraction of DEGs reachable from each perturbed gene |")
        W(f"| Median DEG reachability | {_fv(fp, 'reachability_median')} | |")
        W(f"| Mean path to DEG | {_fv(fp, 'mean_path_to_deg', '.3f')} hops | {path_note} |")
        W(f"| Median path to DEG | {_fv(fp, 'median_path_to_deg', '.3f')} hops | |")
        W(f"| In-graph perturbations (routing) | {fp.get('n_in_graph_perts', '—')} | |")


# ─────────────────────────────────────────────────────────────────────────────
# Module-level parameter labels (shared by all mode-aware sections)
# ─────────────────────────────────────────────────────────────────────────────

_PARAM_LABELS = {
    "alpha":          "α (power-law exponent)",
    "gini":           "Gini (out-degree Gini)",
    "gini_in":        "Gini_in (in-degree Gini)",
    "Q":              "Q (community modularity)",
    "S_max":          "S_max (hub saturation)",
    "C":              "C (clustering coefficient)",
    "rho":            "ρ (assortativity)",
    "epr_k":          "EPR@k (precision at k)",
    "weight_entropy": "Weight entropy (bits)",
    "spectral_gap":   "Spectral gap (λ₂)",
    "source_conc":    "Source concentration",
    "heterophily":    "Heterophily",
}

# Q (modularity) was removed entirely in session 4 (22 June 2026): SCBER's
# cross-community bridge promotion structurally fought it with no
# counterbalancing force, so no probe substrate could ever satisfy it.
# gini_in (in-degree Gini) was its replacement in the organic mode's
# "module" target slot. Session 5 restores Q as a genuine 7th organic
# target now that m_intra (engine.py, the 8th searched hyperparameter)
# gives the optimizer a positive pull toward intra-community structure --
# gini_in stays live too, both are real targets now, not Q-replaces-gini_in.
# See markdowns/claude_code/claude_code_session_4.md Section 8 and
# claude_code_session_5.md.
_ORGANIC_PARAMS   = ["alpha", "gini", "gini_in", "Q", "S_max", "C", "rho"]
_SYNTHETIC_PARAMS = ["epr_k", "weight_entropy", "spectral_gap", "source_conc", "heterophily"]

# Column name in cohort DataFrame for each organic param
_ORGANIC_COL = {"alpha": "alpha", "gini": "Gini", "gini_in": "gini_in", "Q": "Q",
                "S_max": "S_max", "C": "C", "rho": "rho"}


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
    experimental_df          = None,
    df_expansive             = None,
    df_refinement            = None,
    sources_arr              = None,
    targets_arr              = None,
    W_arr                    = None,
    source_pert_impact       = None,
    chi_prior                = None,
    rho_prior                = None,
    total_gate               = None,
    er_diagnostics           = None,
    pert_efficiency_map      = None,
    alpha_md                 = 0.5,
    alpha_stab               = 0.3,
    output_dir               = None,
    graphs_to_report         = None,
    fungi_mode               = "organic",
    fingerprint              = None,
    champ_secondary_metrics  = None,
    cohort_secondary_metrics = None,
):
    """
    Generate a full diagnostic report for a completed FUNGI run.

    Args:
        run_name: identifier string (used in filename and report title).
        cfg: FUNGI config dict.
        adata: AnnData object (for DEG reachability and perturbation info).
        graph_df: champion graph as DataFrame (Regulator, Target, Weight).
        cohort: DataFrame from select_diverse_cohort (cohort_rank, is_champion, …).
        utopian_bounds: dict of param → (lo, hi) from run_diagnostics.
        loss_weights: dict of param → weight.
        diagnostic_report: dict from Phase 0 diagnostics.
        fungi_mode: "organic" or "synthetic" — controls which sections are shown.
        fingerprint: pre-computed HYPHAE fingerprint dict (optional).  If None,
            the structural fingerprint is computed inline from graph_df.
        champ_secondary_metrics: dict from _secondary_metrics for the champion
            (required for synthetic mode topology table and warnings).
        cohort_secondary_metrics: dict of cohort_rank → _secondary_metrics dict
            (enables synthetic-mode cohort table).
        output_dir: Path to write report file.  Default: data/reports/.
        graphs_to_report: list of cohort_rank ints to deep-dive.  Default: [1].

    Writes {run_name}_report.md to output_dir and returns the report as a string.
    """
    if output_dir is None:
        output_dir = Path("data/reports")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if graphs_to_report is None:
        graphs_to_report = [1]

    if champ_secondary_metrics is None:
        champ_secondary_metrics = {}
    if cohort_secondary_metrics is None:
        cohort_secondary_metrics = {}

    # exp_011 Item 6: canonical mode name is "biologic"; "organic" accepted as a
    # legacy alias. Discriminate on "synthetic" so both biologic/organic route to
    # the biologic report sections (_ORGANIC_PARAMS kept as the internal var name).
    fungi_mode = (fungi_mode or "biologic").lower()
    if fungi_mode != "synthetic":
        fungi_mode = "biologic"
    mode_params = _SYNTHETIC_PARAMS if fungi_mode == "synthetic" else _ORGANIC_PARAMS
    n_mode_targets = len(mode_params)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    W = lines.append

    # ── Cover ─────────────────────────────────────────────────────────────────
    W(f"# FUNGI Diagnostic Report")
    W(f"\n**Run name:** `{run_name}`  ")
    W(f"**Generated:** {timestamp}  ")
    W(f"**Mode:** `{fungi_mode.upper()}`  ")
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
    W(f"| FUNGI mode | `{fungi_mode.upper()}` |")
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

    W(f"**Utopian Bounds and Probe Confidence ({fungi_mode.upper()} mode):**\n")
    W(f"| Parameter | Target Interval | Probe Confidence | Loss Weight | Probe Method |")
    W(f"|---|---|---|---|---|")

    probes = dr.get("probes_used", {})
    confs  = dr.get("raw_confidences", {})
    disabled_targets = set(dr.get("disabled_organic_targets", []))
    for param in mode_params:
        lo, hi = utopian_bounds.get(param, (float("nan"), float("nan")))
        conf   = confs.get(param, 0.0)
        wt     = loss_weights.get(param, 0.0)
        probe  = probes.get(param, "—")
        label  = _PARAM_LABELS.get(param, param)
        if param in disabled_targets:
            probe = f"{probe} [DISABLED -- excluded from loss]"
        if not (lo != lo or hi != hi):  # not nan
            W(f"| {label} | [{lo:.4f}, {hi:.4f}] | {conf:.2f} | {wt:.1f} | `{probe}` |")
        else:
            W(f"| {label} | — | — | {wt:.1f} | `{probe}` |")

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
        if len(active_pi) > 0:
            W(f"| Mean π (above floor) | {active_pi.mean():.4f} |")

    W(f"\n**Experimental gate:**\n")
    if total_gate is not None and np.any(total_gate != 1.0):
        n_above   = int((total_gate > 1.0).sum())
        n_below   = int((total_gate < 1.0).sum())
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

    if fungi_mode != "synthetic":
        W(f"| Rank | Loss | α | Gini | Gini_in | Q | S_max | C | ρ | β | δ | κ | k_core | λ | ψ | ν | m_intra | Edges |")
        W(f"|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, row in cohort.iterrows():
            champ_star = " ★" if row["is_champion"] else ""
            W(f"| {int(row['cohort_rank'])}{champ_star} "
              f"| {row['utopia_loss']:.4f} "
              f"| {row['alpha']:.3f} | {row['Gini']:.3f} | {row['gini_in']:.3f} | {row['Q']:.4f} "
              f"| {row['S_max']:.3f} "
              f"| {row['C']:.3f} | {row['rho']:.3f} "
              f"| {row['beta']:.2f} | {row['delta']:.2f} | {row['kappa']:.3f} "
              f"| {row['k_core']:.1f} | {row['lambda']:.2f} | {row['psi']:.2f} "
              f"| {row['nu']:.2f} | {row['m_intra']:.2f} "
              f"| {int(row['n_edges']):,} |")
    else:
        W(f"| Rank | Loss | EPR@k | W_ent | SpGap | Hetero | "
          f"β | δ | κ | k_core | λ | ψ | ν | m_intra | Edges |")
        W(f"|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, row in cohort.iterrows():
            champ_star = " ★" if row["is_champion"] else ""
            rank       = int(row["cohort_rank"])
            csec       = cohort_secondary_metrics.get(rank, {})
            n_ed       = csec.get("n_edges", int(row.get("n_edges", 0)))

            def _sf(d, k):
                v = d.get(k)
                return f"{v:.4f}" if v is not None else "—"

            W(f"| {rank}{champ_star} | {row['utopia_loss']:.4f} "
              f"| {_sf(csec,'epr_k')} | {_sf(csec,'weight_entropy')} "
              f"| {_sf(csec,'spectral_gap')} | {_sf(csec,'heterophily')} "
              f"| {row['beta']:.2f} | {row['delta']:.2f} | {row['kappa']:.3f} "
              f"| {row['k_core']:.1f} | {row['lambda']:.2f} | {row['psi']:.2f} "
              f"| {row['nu']:.2f} | {row['m_intra']:.2f} "
              f"| {int(n_ed):,} |")

    W(f"\n★ = Champion  │  Mode: {fungi_mode.upper()}")

    # ── Per-graph deep dives ───────────────────────────────────────────────────
    for rank_num in graphs_to_report:
        rows = cohort[cohort["cohort_rank"] == rank_num]
        if len(rows) == 0:
            continue
        champ_row = rows.iloc[0]
        tag = "Champion" if champ_row["is_champion"] else f"Alternate {rank_num}"

        # ── 9. Topology Deep-Dive ─────────────────────────────────────────────
        W(_section(f"9 · {tag} — Topology Deep-Dive"))

        W(f"**Hyperparameters:**\n")
        W(f"| Parameter | Value |")
        W(f"|---|---|")
        for hp in ["beta", "delta", "kappa", "k_core", "lambda", "psi", "nu", "m_intra"]:
            W(f"| {hp} | {champ_row[hp]:.4f} |")

        W(f"\n**{fungi_mode.upper()} topology targets:**\n")
        W(f"| Parameter | Observed | Target Interval | Status | Loss Weight |")
        W(f"|---|---|---|---|---|")

        n_pass = 0
        if fungi_mode != "synthetic":
            param_map = [(p, _ORGANIC_COL[p]) for p in _ORGANIC_PARAMS]
            for param, col in param_map:
                lo, hi = utopian_bounds.get(param, (0.0, 1.0))
                val    = float(champ_row.get(col, float("nan")))
                inside = lo <= val <= hi if val == val else False
                n_pass += int(inside)
                wt     = loss_weights.get(param, 0.0)
                W(f"| {_PARAM_LABELS.get(param, param)} "
                  f"| {val:.4f} | {_fmt_bounds(lo, hi)} "
                  f"| {_tick(inside)} | {wt:.1f} |")
        else:
            param_map = [(p, p) for p in _SYNTHETIC_PARAMS]
            for param, _ in param_map:
                lo, hi = utopian_bounds.get(param, (0.0, 1.0))
                val    = float(champ_secondary_metrics.get(param, float("nan")))
                inside = lo <= val <= hi if val == val else False
                n_pass += int(inside)
                wt     = loss_weights.get(param, 0.0)
                W(f"| {_PARAM_LABELS.get(param, param)} "
                  f"| {'—' if val != val else f'{val:.4f}'} "
                  f"| {_fmt_bounds(lo, hi)} "
                  f"| {_tick(inside)} | {wt:.1f} |")

        W(f"\n**{n_pass}/{n_mode_targets} topology targets satisfied**  ")
        W(f"**Utopia loss: {champ_row['utopia_loss']:.6f}**")

        # ── 9F. HYPHAE Graph Fingerprint ──────────────────────────────────────

        # Determine perturbation targets for hub coverage
        _pert_targets = None
        if adata is not None:
            try:
                pc  = cfg["input"]["perturbation_column"]
                cl  = cfg["input"]["control_label"]
                _pert_targets = [g for g in adata.obs[pc].unique() if g != cl]
            except Exception:
                pass

        # Use provided fingerprint or compute structural fingerprint inline
        if fingerprint is not None:
            _fp = fingerprint
        else:
            _fp = compute_fingerprint_inline(graph_df, pert_targets=_pert_targets)

        _render_fingerprint_section(lines, _fp, tag)

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

        od_vals = out_deg.values
        pcts    = [50, 75, 90, 95, 99]
        W(f"\n**Out-degree percentiles:**\n")
        W(f"| Percentile | Out-degree |")
        W(f"|---|---|")
        for p in pcts:
            W(f"| p{p} | {np.percentile(od_vals, p):.0f} |")

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

            in_graph_reach = df_reach[df_reach["in_graph"] & (df_reach["n_degs"] > 0)]
            W(f"\n**Reachability distribution ({len(in_graph_reach)} "
              f"in-graph perturbations):**\n")
            W("```")
            W(_reachability_histogram(in_graph_reach))
            W("```")

            W(f"\n**Top 15 most reachable perturbations:**\n")
            W(f"| Gene | In Graph | DEGs | Reachable | Fraction |")
            W(f"|---|---|---|---|---|")
            for _, r in df_reach[df_reach["in_graph"]].head(15).iterrows():
                W(f"| `{r['perturbation']}` | ✓ | {r['n_degs']} "
                  f"| {r['n_reachable']} | {r['fraction_reachable']:.3f} |")

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

            W(f"\n<details>\n<summary>Full per-perturbation reachability table "
              f"(click to expand)</summary>\n")
            W(f"\n| Gene | In Graph | DEGs | Reachable | Fraction |")
            W(f"|---|---|---|---|---|")
            for _, r in df_reach.iterrows():
                in_g = "✓" if r["in_graph"] else "✗"
                W(f"| `{r['perturbation']}` | {in_g} | {r['n_degs']} "
                  f"| {r['n_reachable']} | {r['fraction_reachable']:.3f} |")
            W(f"\n</details>\n")

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

        if fungi_mode != "synthetic":
            W(f"| Rank | Tag | Loss | Edges | α | Gini | Gini_in | Q | S_max | C | ρ | β | λ | ψ | ν | m_intra |")
            W(f"|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for _, row in cohort.iterrows():
                tag_str = "Champion ★" if row["is_champion"] else f"Alternate {int(row['cohort_rank'])}"
                W(f"| {int(row['cohort_rank'])} | {tag_str} "
                  f"| {row['utopia_loss']:.4f} | {int(row['n_edges']):,} "
                  f"| {row['alpha']:.3f} | {row['Gini']:.3f} | {row['gini_in']:.3f} | {row['Q']:.4f} "
                  f"| {row['S_max']:.3f} "
                  f"| {row['C']:.3f} | {row['rho']:.3f} "
                  f"| {row['beta']:.2f} | {row['lambda']:.2f} | {row['psi']:.2f} "
                  f"| {row['nu']:.2f} | {row['m_intra']:.2f} |")
        else:
            W(f"| Rank | Tag | Loss | Edges | EPR@k | W_ent | SpGap | Hetero | β | λ | ψ | ν | m_intra |")
            W(f"|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for _, row in cohort.iterrows():
                tag_str = "Champion ★" if row["is_champion"] else f"Alternate {int(row['cohort_rank'])}"
                rank    = int(row["cohort_rank"])
                csec    = cohort_secondary_metrics.get(rank, {})
                n_ed    = csec.get("n_edges", int(row.get("n_edges", 0)))

                def _sf2(d, k):
                    v = d.get(k)
                    return f"{v:.4f}" if v is not None else "—"

                W(f"| {rank} | {tag_str} "
                  f"| {row['utopia_loss']:.4f} | {int(n_ed):,} "
                  f"| {_sf2(csec,'epr_k')} | {_sf2(csec,'weight_entropy')} "
                  f"| {_sf2(csec,'spectral_gap')} | {_sf2(csec,'heterophily')} "
                  f"| {row['beta']:.2f} | {row['lambda']:.2f} | {row['psi']:.2f} "
                  f"| {row['nu']:.2f} | {row['m_intra']:.2f} |")

        W(f"\nAll cohort graphs are selected by farthest-point sampling in "
          f"normalised hyperparameter space, ensuring diverse regulatory architectures.")

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

    if fungi_mode != "synthetic":
        organic_map = [(p, _ORGANIC_COL[p]) for p in _ORGANIC_PARAMS]
        n_pass_champ = sum(
            1 for param, col in organic_map
            if param in utopian_bounds
            and utopian_bounds[param][0] <= float(champion_row.get(col, float("nan"))) <= utopian_bounds[param][1]
        )
        if n_pass_champ < n_mode_targets:
            failing = [param for param, col in organic_map
                       if param not in utopian_bounds
                       or not (utopian_bounds[param][0]
                               <= float(champion_row.get(col, float("nan")))
                               <= utopian_bounds[param][1])]
            flags.append(f"⚠ Champion fails {n_mode_targets - n_pass_champ}/"
                         f"{n_mode_targets} organic topology targets: "
                         f"{', '.join(failing)}.")
        s_max_val = float(champion_row.get("S_max", 0))
        if "S_max" in utopian_bounds and s_max_val < utopian_bounds["S_max"][0] * 1.05:
            flags.append(f"⚠ Champion S_max ({s_max_val:.4f}) is near the "
                         f"lower bound ({utopian_bounds['S_max'][0]:.4f}). "
                         f"Hub structure may be under-developed.")
    else:
        if champ_secondary_metrics:
            n_pass_champ = sum(
                1 for param in _SYNTHETIC_PARAMS
                if param in utopian_bounds
                and not (champ_secondary_metrics.get(param) is None)
                and utopian_bounds[param][0]
                    <= float(champ_secondary_metrics[param])
                    <= utopian_bounds[param][1]
            )
            if n_pass_champ < n_mode_targets:
                failing = [param for param in _SYNTHETIC_PARAMS
                           if param not in utopian_bounds
                           or champ_secondary_metrics.get(param) is None
                           or not (utopian_bounds[param][0]
                                   <= float(champ_secondary_metrics[param])
                                   <= utopian_bounds[param][1])]
                flags.append(f"⚠ Champion fails {n_mode_targets - n_pass_champ}/"
                             f"{n_mode_targets} synthetic topology targets: "
                             f"{', '.join(failing)}.")
        else:
            flags.append("⚠ champ_secondary_metrics not provided — "
                         "cannot audit synthetic topology targets.")

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


# ─────────────────────────────────────────────────────────────────────────────
# Raw (no-FUNGI-search) graph report -- session 6, SHROOM_2-family bake-off
# ─────────────────────────────────────────────────────────────────────────────

def generate_raw_graph_report(run_name, graph_df, output_dir,
                               adata=None, pert_col=None, control_label=None,
                               pert_targets=None, note=None):
    """
    Lightweight structural report for a graph that has NOT been through a
    FUNGI search -- a raw SHROOM dense output, an individual FUNGI cohort
    member re-examined standalone, or a graph naively pruned to a target
    edge count for a same-size comparison. Reuses the same structural-
    fingerprint machinery generate_report() uses for its Section 9F (topology,
    hub, community, weight, spectral -- all graph-only, no hyperparameters
    needed) plus a hub/in-degree table and an optional perturbation-coverage
    check, but skips every FUNGI-search-specific section (hyperparameters,
    utopian bounds, loss weights, cohort, search/refinement trajectory) since
    none of that exists for an untreated graph.

    Returns (report_text, fingerprint_dict) -- the dict is for programmatic
    aggregation across many graphs, not just the human-readable report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    W = lines.append

    W(f"# FUNGI Structural Report (RAW -- no FUNGI search applied)")
    W(f"\n**Run name:** `{run_name}`  ")
    W(f"**Generated:** {timestamp}  ")
    if note:
        W(f"**Note:** {note}  ")
    W(f"\n---\n")

    n_edges = len(graph_df)
    n_nodes = len(set(graph_df["Regulator"]) | set(graph_df["Target"]))
    W(_section("1 · Run Identity"))
    W(f"| Field | Value |")
    W(f"|---|---|")
    W(f"| Run name | `{run_name}` |")
    W(f"| Timestamp | {timestamp} |")
    W(f"| Total edges | {n_edges:,} |")
    W(f"| Nodes present | {n_nodes:,} |")

    pert_targets_resolved = pert_targets
    if pert_targets_resolved is None and adata is not None and pert_col is not None:
        pert_targets_resolved = [g for g in adata.obs[pert_col].unique() if g != control_label]

    fp = compute_fingerprint_inline(graph_df, pert_targets=pert_targets_resolved)
    _render_fingerprint_section(lines, fp, tag=run_name)

    # Hub/out-degree + target/in-degree -- same content as generate_report's
    # sections 10-11, graph-only, no hyperparameters needed.
    out_deg = graph_df["Regulator"].value_counts()
    in_deg = graph_df["Target"].value_counts()

    W(_section(f"10 · {run_name} — Hub & Out-degree Analysis"))
    W(f"| | Value |")
    W(f"|---|---|")
    W(f"| Mean out-degree | {out_deg.mean():.2f} |")
    W(f"| Median out-degree | {out_deg.median():.1f} |")
    W(f"| Max out-degree | {out_deg.max()} |")
    W(f"| Genes with ≥1 outgoing edge | {len(out_deg):,} |")
    W(f"\n**Top 20 regulators by out-degree:**\n")
    W(f"| Rank | Gene | Out-degree | % of all edges |")
    W(f"|---|---|---|---|")
    for rank_i, (gene, cnt) in enumerate(out_deg.head(20).items(), 1):
        W(f"| {rank_i} | `{gene}` | {cnt} | {_pct(cnt, n_edges)} |")

    W(_section(f"11 · {run_name} — Target & In-degree Analysis"))
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

    if pert_targets_resolved is not None:
        graph_sources = set(graph_df["Regulator"])
        in_src = [g for g in pert_targets_resolved if g in graph_sources]
        not_in = [g for g in pert_targets_resolved if g not in graph_sources]
        W(_section(f"12 · {run_name} — Perturbation Coverage Audit"))
        W(f"| | Count |")
        W(f"|---|---|")
        W(f"| Total perturbation targets | {len(pert_targets_resolved)} |")
        W(f"| Present as source in graph | {len(in_src)} ({_pct(len(in_src), len(pert_targets_resolved))}) |")
        W(f"| Missing from graph | {len(not_in)} |")

    W(f"\n\n---\n*FUNGI Structural Report (raw graph) — {run_name} — {timestamp}*")

    report_text = "\n".join(lines)
    out_path = output_dir / f"{run_name}_raw_report.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"Raw graph report saved -> {out_path}")
    return report_text, fp


if __name__ == "__main__":
    import argparse as _argparse
    import json as _json

    import anndata as _ad

    _ap = _argparse.ArgumentParser(description="Standalone CLI for generate_raw_graph_report")
    _ap.add_argument("--graph", required=True, help="Graph parquet (Regulator/Target/Weight or "
                                                      "Target/Regulator/Importance columns)")
    _ap.add_argument("--run-name", required=True)
    _ap.add_argument("--output-dir", required=True)
    _ap.add_argument("--sc-input", default=None, help="h5ad for perturbation-coverage audit (optional)")
    _ap.add_argument("--pert-col", default="gene")
    _ap.add_argument("--control-label", default="non-targeting")
    _ap.add_argument("--fingerprint-json", default=None,
                      help="Also dump the computed fingerprint dict to this JSON path")
    _args = _ap.parse_args()

    _df = pd.read_parquet(_args.graph)
    _cols = {c.lower(): c for c in _df.columns}
    _reg = _cols.get("regulator", _df.columns[1])
    _tgt = _cols.get("target", _df.columns[0])
    _w = _cols.get("weight", _cols.get("importance", _df.columns[2]))
    _graph_df = _df[[_reg, _tgt, _w]].copy()
    _graph_df.columns = ["Regulator", "Target", "Weight"]

    _adata = None
    if _args.sc_input:
        _adata = _ad.read_h5ad(_args.sc_input, backed="r")

    _text, _fp = generate_raw_graph_report(
        run_name=_args.run_name, graph_df=_graph_df, output_dir=_args.output_dir,
        adata=_adata, pert_col=_args.pert_col, control_label=_args.control_label)

    if _args.fingerprint_json:
        with open(_args.fingerprint_json, "w", encoding="utf-8") as _fh:
            _json.dump(_fp, _fh, indent=2, default=str)
        print(f"Fingerprint JSON saved -> {_args.fingerprint_json}")