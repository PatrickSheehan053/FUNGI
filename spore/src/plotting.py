"""
SPORE+ · src/plotting.py
────────────────────────────────
Publication-quality plots for the SPORE+ pipeline.

Design philosophy
─────────────────
• White / seaborn "paper" background — no dark navy blocks
• Information density: every figure encodes ≥2 variables simultaneously
• 'mako' colormap exclusively for continuous data
• Layered geometry: KDE fill + contour lines + marginal density where applicable
• Compact grid layout — figures sized to fit a laptop screen without scrolling
• Bold panel-letter labels (A, B, C …) for multi-panel figures
• Text outputs replace trivial single-number plots entirely

Colour conventions (Mako + Accent)
──────────────────────────────────
  Marginals / Deepest:  #251729
  Low / Base:           #403B78
  Mid / Muted:          #3497A9
  High / Highlight:     #AEE3C0
  Accent / Alert:       #CB3D22
  Text / Dark UI:       #1a1a1a  (dark slate gray)
  Background:           #ffffff  (white)
  Grid / Spines:        #cccccc  (light gray)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_THEME = {
    "bg":        "#ffffff",
    "grid":      "#cccccc",
    "spine":     "#cccccc",
    "text":      "#1a1a1a",
    "dark":      "#251729",
    "low":       "#403B78",
    "mid":       "#3480a9",
    "high":      "#AEE3C0",
    "accent":    "#CB3D22"
}

def _apply_theme():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    mpl.rcParams.update({
        "figure.facecolor":  _THEME["bg"],
        "axes.facecolor":    _THEME["bg"],
        "axes.edgecolor":    _THEME["spine"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        _THEME["grid"],
        "grid.linewidth":    0.6,
        "axes.labelcolor":   _THEME["text"],
        "xtick.color":       _THEME["text"],
        "ytick.color":       _THEME["text"],
        "text.color":        _THEME["text"],
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "legend.frameon":    False,
        "legend.fontsize":   10,
        "savefig.dpi":       200,
        "savefig.bbox":      "tight",
        "figure.dpi":        120,
    })

def _label(ax, letter):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="left",
            color=_THEME["neutral"])


def _save(fig, cfg, stem):
    figs = cfg.get("paths", {}).get("_figures")
    if figs is None:
        return None
    path = Path(figs) / f"{stem}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=_THEME["bg"])
    return path


def _show(fig):
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 0 · Sparsity overview
# ─────────────────────────────────────────────────────────────────────────────

def plot_sparsity_overview(adata, cfg):
    _apply_theme()
    import scipy.sparse as sp
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    n_cells, n_genes = adata.shape
    
    # ── THE FIX: Safe Matrix Iteration for Large Datasets ──
    # If the dataset is backed (large_dataset_mode) or massive, we process
    # it in memory-safe chunks to prevent catastrophic RAM spikes.
    is_large = getattr(adata, 'isbacked', False) or n_cells > 100000

    if is_large:
        print(f"  Large/Backed dataset detected. Computing sparsity metrics in chunks to prevent OOM...")
        cpg = np.zeros(n_genes, dtype=int)
        gpc = np.zeros(n_cells, dtype=int)
        total_counts = np.zeros(n_cells, dtype=np.float32)

        chunk_size = 50000
        for i in range(0, n_cells, chunk_size):
            end = min(i + chunk_size, n_cells)
            X_chunk = adata.X[i:end]

            if sp.issparse(X_chunk):
                cpg += np.asarray((X_chunk > 0).sum(axis=0)).ravel()
                gpc[i:end] = np.asarray((X_chunk > 0).sum(axis=1)).ravel()
                total_counts[i:end] = np.asarray(X_chunk.sum(axis=1)).ravel()
            else:
                cpg += np.asarray((X_chunk > 0).sum(axis=0)).ravel()
                gpc[i:end] = np.asarray((X_chunk > 0).sum(axis=1)).ravel()
                total_counts[i:end] = np.asarray(X_chunk.sum(axis=1)).ravel()
    else:
        # Fast path for small, fully-in-memory datasets
        X = adata.X
        if sp.issparse(X):
            cpg = np.asarray((X > 0).sum(axis=0)).ravel()
            gpc = np.asarray((X > 0).sum(axis=1)).ravel()
            total_counts = np.asarray(X.sum(axis=1)).ravel()
        else:
            cpg = (X > 0).sum(axis=0)
            gpc = (X > 0).sum(axis=1)
            total_counts = X.sum(axis=1)

    # 2. Setup the Canvas
    fig = plt.figure(figsize=(15, 6.5))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.25)
    
    # ── PANEL A: Sequencing Saturation Scatter Plot (Left) ─────────────
    inner_left = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[0], 
        width_ratios=[4, 1], height_ratios=[1, 4], 
        wspace=0.05, hspace=0.05
    )
    ax_main = fig.add_subplot(inner_left[1, 0])
    ax_top = fig.add_subplot(inner_left[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(inner_left[1, 1], sharey=ax_main)
    
    # Avoid log(0) and subsample to 50k for smooth, fast rendering
    log_counts = np.log10(total_counts + 1)
    if n_cells > 50000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_cells, 50000, replace=False)
        plot_x, plot_y = log_counts[idx], gpc[idx]
    else:
        plot_x, plot_y = log_counts, gpc

    # The Scatter Plot
    ax_main.scatter(plot_x, plot_y, color=_THEME["low"], s=5, alpha=0.6, edgecolor="None", rasterized=True)
    
    # The Marginals 
    sns.kdeplot(x=plot_x, ax=ax_top, color=_THEME["high"], fill=True, alpha=0.8, lw=1.5)
    sns.kdeplot(y=plot_y, ax=ax_right, color=_THEME["high"], fill=True, alpha=0.8, lw=1.5)
    
    # Explicitly labeling marginals
    ax_top.set_ylabel("Cell\nDensity", fontsize=10, fontweight="bold", rotation=0, labelpad=25, va="center")
    ax_right.set_xlabel("Cell\nDensity", fontsize=10, fontweight="bold")
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    
    ax_right.xaxis.set_major_locator(ticker.MaxNLocator(nbins=1))
    ax_right.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '0' if x == 0 else f"{x:g}"))
    
    ax_main.set_xlabel("Total UMI Counts (log10)", fontweight="bold")
    ax_main.set_ylabel("Genes Detected per Cell", fontweight="bold")
    ax_top.set_title("A  Sequencing Saturation", loc="left", fontweight="bold", pad=15)
    
    # ── PANELS B & C: Hard Quantification (Right) ─────────────────────
    inner_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.45)
    ax_r1 = fig.add_subplot(inner_right[0])
    ax_r2 = fig.add_subplot(inner_right[1])
    
    # Panel B: CPG Histogram
    p_low = max(cpg.min(), 1) 
    p_high = max(cpg.max(), 2)
    log_bins = np.logspace(np.log10(p_low * 0.8), np.log10(p_high * 1.2), 50)
    
    # Using 'low' to anchor the histogram bars solidly
    ax_r1.hist(cpg, bins=log_bins, color=_THEME["mid"], edgecolor="white", lw=0.5)
    ax_r1.set_xscale("log")
    ax_r1.set_xlim(p_low * 0.5, p_high * 1.5) 
    
    ax_r1.axvline(np.median(cpg), color=_THEME["accent"], ls="--", lw=1.5, label=f"Median: {int(np.median(cpg)):,}")
    ax_r1.set_xlabel("Cells expressing gene (log scale)")
    ax_r1.set_ylabel("Number of Genes")
    ax_r1.set_title("B  Gene Detection Frequency", loc="left", fontweight="bold")
    ax_r1.legend(loc="upper left")
    
    # Panel C: GPC ECDF
    sorted_g = np.sort(gpc)
    ecdf = np.arange(1, len(sorted_g) + 1) / len(sorted_g)
    
    # Matching the 'low' tone for structural consistency
    ax_r2.plot(sorted_g, ecdf, color=_THEME["low"], lw=2.5)
    ax_r2.axvline(np.median(gpc), color=_THEME["accent"], ls="--", lw=1.5, label=f"Median: {int(np.median(gpc)):,}")
    ax_r2.set_xlabel("Genes Detected per Cell")
    ax_r2.set_ylabel("Cumulative Fraction")
    ax_r2.set_title("C  Genes per Cell (ECDF)", loc="left", fontweight="bold")
    ax_r2.legend(loc="lower right")
    
    fig.suptitle(f"Data Sparsity & Saturation  |  {adata.n_obs:,} cells × {adata.n_vars:,} genes", 
                 fontsize=15, fontweight="bold", y=1.04)
    
    sns.despine()
    # Safely handle the save/show logic
    try:
        path = _save(fig, cfg, "sparsity_saturation")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 · QC
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_qc_for_plots(adata, cfg):
    """Silent firewall for plotting raw data."""
    import scanpy as sc
    if "total_counts" not in adata.obs.columns or "pct_counts_mt" not in adata.obs.columns:
        organism = cfg.get("dataset", {}).get("organism", "human")
        mt_prefix = "MT-" if organism == "human" else "mt-"
        adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    return adata

def plot_qc_violin(adata, cfg):
    _apply_theme()
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Trigger firewall to ensure math exists
    adata = _ensure_qc_for_plots(adata, cfg)
    
    # 1. Setup Canvas
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    
    # 2. Extract Phase 2 Thresholds for visual cut-offs
    p2 = cfg.get("phase2_cell_triage", {})
    min_genes = p2.get("min_genes_per_cell", 200)
    max_genes = p2.get("max_genes_per_cell", 10000)
    min_counts = p2.get("min_counts_per_cell", 500)
    max_counts = p2.get("max_counts_per_cell", 80000)
    mt_thresh = p2.get("mt_threshold", 0.20) * 100
    
    # Theme Mapping
    base_color = _THEME["accent"]       # Fill color
    cut_color = _THEME["dark"]     # Threshold lines
    border_color = _THEME["dark"]    # Light gray for border/inner box
    
    # ── Plot A: Genes Detected ──
    sns.violinplot(y=adata.obs["n_genes_by_counts"], ax=axes[0], color=base_color, 
                   inner="box", linewidth=1.5, linecolor=border_color, alpha=0.95, cut=0)
    axes[0].axhline(min_genes, color=cut_color, ls="--", lw=1.5, label=f"Min: {min_genes:,}")
    axes[0].axhline(max_genes, color=cut_color, ls="--", lw=1.5, label=f"Max: {max_genes:,}")
    
    # Clamp Axis A
    g_min, g_max = adata.obs["n_genes_by_counts"].min(), adata.obs["n_genes_by_counts"].max()
    g_pad = (g_max - g_min) * 0.05
    axes[0].set_ylim(max(0, g_min - g_pad), g_max + g_pad)
    
    axes[0].set_title("A  Genes Detected", loc="left", fontweight="bold")
    axes[0].set_ylabel("Number of Genes")
    axes[0].legend(loc="upper right")
    
    # ── Plot B: Total UMI Counts (Log Scaled) ──
    sns.violinplot(y=adata.obs["total_counts"], ax=axes[1], color=base_color, 
                   inner="box", linewidth=1.5, linecolor=border_color, alpha=0.95, cut=0)
    axes[1].axhline(min_counts, color=cut_color, ls="--", lw=1.5, label=f"Min: {min_counts:,}")
    axes[1].axhline(max_counts, color=cut_color, ls="--", lw=1.5, label=f"Max: {max_counts:,}")
    axes[1].set_yscale("log")
    
    # Clamp Axis B (Log space padding)
    c_min, c_max = adata.obs["total_counts"].min(), adata.obs["total_counts"].max()
    axes[1].set_ylim(c_min * 0.8, c_max * 1.25)
    
    axes[1].set_title("B  Total UMI Counts", loc="left", fontweight="bold")
    axes[1].set_ylabel("Total Counts (log scale)")
    axes[1].legend(loc="upper right")
    
    # ── Plot C: Mitochondrial Fraction ──
    sns.violinplot(y=adata.obs["pct_counts_mt"], ax=axes[2], color=base_color, 
                   inner="box", linewidth=1.5, linecolor=border_color, alpha=0.95, cut=0)
    axes[2].axhline(mt_thresh, color=cut_color, ls="--", lw=1.5, label=f"Max: {mt_thresh}%")
    
    # Clamp Axis C
    mt_min, mt_max = adata.obs["pct_counts_mt"].min(), adata.obs["pct_counts_mt"].max()
    mt_pad = (mt_max - mt_min) * 0.1
    axes[2].set_ylim(max(0, mt_min - mt_pad), mt_max + mt_pad)
    
    axes[2].set_title("C  Mitochondrial Fraction", loc="left", fontweight="bold")
    axes[2].set_ylabel("% Mitochondrial")
    axes[2].legend(loc="upper right")
    
    # ── Clean up aesthetics ──
    for i, ax in enumerate(axes):
        ax.set_xticks([]) # Remove useless x-ticks
        ax.set_xlabel("")
        ax.grid(False, axis='x') # Explicitly turn off vertical gridlines
        
        if i == 1:
            # Force BOTH major and minor gridlines for the log-scaled axis
            ax.grid(True, axis='y', which='both', color=_THEME["grid"], linewidth=0.6, alpha=0.7)
        else:
            ax.grid(True, axis='y', which='major', color=_THEME["grid"], linewidth=0.6)
            
    fig.suptitle(f"Phase 2 · Pre-Filter QC Metrics | {adata.n_obs:,} cells", 
                 fontweight="bold", fontsize=16, y=1.05)
    plt.tight_layout()
    
    # Safely handle the save/show logic
    try:
        path = _save(fig, cfg, "p02_qc_violin")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

def plot_mt_scatter(adata, cfg):
    _apply_theme()
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    
    # Trigger firewall to ensure math exists
    adata = _ensure_qc_for_plots(adata, cfg)
    
    # 1. Extract and prep data
    total_counts = adata.obs["total_counts"].values
    pct_mt = adata.obs["pct_counts_mt"].values
    log_counts = np.log10(total_counts + 1)
    
    # Downsample for rendering speed and to prevent visual over-saturation
    n_cells = len(log_counts)
    if n_cells > 30000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_cells, 30000, replace=False)
        x, y = log_counts[idx], pct_mt[idx]
    else:
        x, y = log_counts, pct_mt
        
    # 2. Setup the Canvas
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    # ── 3. The Layered Bivariate Main Plot ──
    ax_main.scatter(x, y, s=10, color=_THEME["dark"], alpha=0.9, edgecolor="none", zorder=1, rasterized=True)
    sns.histplot(x=x, y=y, bins=100, pthresh=0.05, cmap="mako", ax=ax_main, zorder=2, alpha=0.9)
    
    # Contour: Increased levels to 10, dropped bw_adjust to 0.5 for craggy topology
    sns.kdeplot(x=x, y=y, levels=10, bw_adjust=0.5, color="white", linewidths=0.8, ax=ax_main, zorder=3, alpha=0.8)
    
    # Layer 4: Threshold lines 
    mt_thresh = cfg.get("phase2_cell_triage", {}).get("mt_threshold", 0.20) * 100
    min_counts_val = cfg.get("phase2_cell_triage", {}).get("min_counts_per_cell", 500)
    min_counts = np.log10(min_counts_val) if min_counts_val > 0 else 0
    
    # Highlight lines using 'accent'
    ax_main.axhline(mt_thresh, color=_THEME["accent"], ls="--", lw=1.5, zorder=4, label=f"MT Limit: {mt_thresh}%")
    ax_main.axvline(min_counts, color=_THEME["accent"], ls="--", lw=1.5, zorder=4, label=f"Min Counts: {min_counts_val}")
    
    # ── 4. The Marginals ──
    # Using 'mid' for visibility of density curves against white
    marginal_color = _THEME["high"]
    sns.kdeplot(x=x, ax=ax_top, color=marginal_color, fill=True, alpha=0.7, lw=2)
    sns.kdeplot(y=y, ax=ax_right, color=marginal_color, fill=True, alpha=0.7, lw=2)
    
    ax_top.axis("off")
    ax_right.axis("off")
    
    # ── FIX: Clamp the axes AT THE VERY END so Matplotlib can't override it ──
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    ax_main.set_xlim(x_min - (x_range * 0.05), x_max + (x_range * 0.05))
    ax_main.set_ylim(max(0, y_min - (y_range * 0.05)), y_max + (y_range * 0.15))
    
    ax_main.set_xlabel("Total UMI Counts (log10)", fontweight="bold")
    ax_main.set_ylabel("% Mitochondrial Counts", fontweight="bold")
    ax_main.legend(loc="upper right")
    
    fig.suptitle(f"Phase 2 · Mitochondrial RNA Analysis | {adata.n_obs:,} cells", fontweight="bold", fontsize=15, y=0.92)
    
    # Safely handle the save/show logic
    try:
        path = _save(fig, cfg, "p02_mt_scatter")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3 · Ambient RNA Decontamination
# ─────────────────────────────────────────────────────────────────────────────

def plot_ambient_diagnostics(adata, cfg, kde_levels=8, kde_smooth=0.5, scatter_alpha=0.9):
    _apply_theme()
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    import numpy as np
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    
    if "ambient_score" not in adata.obs.columns:
        print("No ambient scoring found. Run Phase 3 first.")
        return None, None

    # 1. Extract Data
    scores = adata.obs["ambient_score"].values
    total_counts = adata.obs["total_counts"].values
    log_counts = np.log10(total_counts + 1)
    
    # Thresholds
    threshold = float(cfg.get("phase3_ambient", {}).get("flag_threshold", 0.80))
    min_umi_pct = float(cfg.get("phase3_ambient", {}).get("min_umi_pct", 0.10))
    umi_thresh_val = np.percentile(total_counts, min_umi_pct * 100)
    log_umi_thresh = np.log10(umi_thresh_val + 1)
    
    # Downsample if massive
    n_cells = len(scores)
    if n_cells > 100000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_cells, 100000, replace=False)
        x, y = scores[idx], log_counts[idx]
    else:
        x, y = scores, log_counts

    # 2. Setup Canvas
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # ── Anchor elements to the data bounds ──
    y_min_visible = y.min() * 0.98  
    rect_height = log_umi_thresh - y_min_visible

    # ── 3. Danger Zone Shading ──
    danger_rect = patches.Rectangle(
        (threshold, y_min_visible), 
        1.0 - threshold,            
        rect_height,                
        linewidth=0, facecolor=_THEME["accent"], alpha=0.3, zorder=1
    )
    ax_main.add_patch(danger_rect)

    # ── 4. The Layered Bivariate Main Plot ──
    ax_main.scatter(x, y, s=5, color=_THEME["dark"], alpha=scatter_alpha, edgecolor="none", zorder=2, rasterized=True)
    
    # Mako Heatmap core
    sns.histplot(x=x, y=y, bins=100, pthresh=0.05, cmap="mako", ax=ax_main, zorder=3, alpha=0.9)
    
    # White Topographical Contours (Tuned by your function arguments)
    sns.kdeplot(x=x, y=y, levels=kde_levels, bw_adjust=kde_smooth, color="white", linewidths=0.8, ax=ax_main, zorder=4, alpha=0.8)

    # ── 5. Threshold Lines ──
    line_color = _THEME["accent"]
    ax_main.axvline(threshold, color=line_color, ls="--", lw=1.5, zorder=5, label=f"Score Limit: {threshold}")
    ax_main.axhline(log_umi_thresh, color=line_color, ls="--", lw=1.5, zorder=5, label=f"UMI Bottom {int(min_umi_pct*100)}%")

    # ── 6. Marginals ──
    sns.kdeplot(x=x, ax=ax_top, color=_THEME["high"], fill=True, alpha=0.7, lw=1.5)
    sns.kdeplot(y=y, ax=ax_right, color=_THEME["high"], fill=True, alpha=0.7, lw=1.5)
    
    ax_top.axis("off")
    ax_right.axis("off")

    # ── 7. Clamping and Labels ──
    x_min, x_max = x.min(), x.max()
    y_max = y.max()
    
    ax_main.set_xlim(max(0, x_min - 0.02), min(1.02, x_max + 0.02))
    ax_main.set_ylim(y_min_visible, y_max * 1.02)
    
    ax_main.set_xlabel("Ambient RNA Score", fontweight="bold")
    ax_main.set_ylabel("Total UMI Counts (log10)", fontweight="bold")
    ax_main.legend(loc="upper left")

    fig.suptitle(f"Phase 3 · Ambient RNA Detection | {adata.n_obs:,} cells", 
                 fontweight="bold", fontsize=15, y=0.94)
    
    try:
        path = _save(fig, cfg, "p03_ambient_scatter")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 4 · Doublet Detection
# ─────────────────────────────────────────────────────────────────────────────

def plot_doublets(adata, cfg, kde_levels=6, kde_smooth=0.5, scatter_alpha=0.8):
    _apply_theme()
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    import numpy as np
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    
    if "doublet_score" not in adata.obs.columns:
        print("No doublet scoring found. Run Phase 4 first.")
        return None, None

    # 1. Extract Data
    scores = adata.obs["doublet_score"].values
    pred_mask = adata.obs["predicted_doublet"].values.astype(bool)
    total_counts = adata.obs["total_counts"].values
    log_counts = np.log10(total_counts + 1)
    
    # 2. Reconstruct the dynamic Scrublet threshold
    if pred_mask.sum() > 0:
        threshold = scores[pred_mask].min()
    else:
        threshold = 0.5  # Safe fallback if 0 doublets were found
        
    # Downsample for rendering speed
    n_cells = len(scores)
    if n_cells > 100000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_cells, 100000, replace=False)
        x_samp = scores[idx]
        y_samp = log_counts[idx]
        pred_samp = pred_mask[idx]
    else:
        x_samp, y_samp, pred_samp = scores, log_counts, pred_mask

    # ── FIX: Dynamic X-Axis Limit ──
    x_min_plot = max(-0.01, scores.min() - 0.01)
    x_max_plot = scores.max() + 0.05
    
    # 3. Setup the Master Canvas (Single Joint Plot)
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    y_min_visible = y_samp.min() * 0.98  
    rect_height = y_samp.max() * 1.02 - y_min_visible

    # ── 4. The Layered Bivariate Main Plot ──
    
    # Layer 1: Danger Zone Shading
    if threshold <= x_max_plot:
        danger_rect = patches.Rectangle(
            (threshold, y_min_visible), 
            x_max_plot - threshold, rect_height,                
            linewidth=0, facecolor=_THEME["accent"], alpha=0.15, zorder=1
        )
        ax_main.add_patch(danger_rect)

    # Layer 2: Singlet Scatter base
    ax_main.scatter(x_samp[~pred_samp], y_samp[~pred_samp], 
                    s=10, color=_THEME["dark"], alpha=scatter_alpha, edgecolor="none", zorder=2, rasterized=True, label="Singlet")
    
    # Layer 3: Mako Heatmap core
    sns.histplot(x=x_samp, y=y_samp, bins=100, pthresh=0.05, cmap="mako", ax=ax_main, zorder=3, alpha=0.9)
    
    # Layer 4: White Topographical Contours
    sns.kdeplot(x=x_samp, y=y_samp, levels=kde_levels, bw_adjust=kde_smooth, color="white", linewidths=0.8, ax=ax_main, zorder=4, alpha=0.8)

    # Layer 5: Doublet Scatter (Must sit on top of heatmap to be visible)
    if pred_samp.sum() > 0:
        ax_main.scatter(x_samp[pred_samp], y_samp[pred_samp], 
                        s=8, color=_THEME["high"], alpha=0.9, edgecolor="none", zorder=5, rasterized=True, label="Doublet")

    # Layer 6: Threshold Lines & Annotation Logic
    if threshold <= x_max_plot:
        ax_main.axvline(threshold, color=_THEME["accent"], ls="--", lw=1.5, zorder=6, label=f"Limit: {threshold:.3f}")
    else:
        # Threshold is off-screen
        ax_main.annotate(
            f"Limit: {threshold:.2f} ➔", 
            xy=(0.98, 0.5), xycoords='axes fraction', 
            ha='right', va='center', 
            color=_THEME["accent"], fontweight='bold', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", fc=_THEME["bg"], ec=_THEME["accent"], lw=1.5, alpha=0.95),
            zorder=6
        )

    # Marginals
    sns.kdeplot(x=x_samp, ax=ax_top, color=_THEME["high"], fill=True, alpha=0.7, lw=1.5)
    sns.kdeplot(y=y_samp, ax=ax_right, color=_THEME["high"], fill=True, alpha=0.7, lw=1.5)
    
    ax_top.axis("off")
    ax_right.axis("off")

    # Clamping and Labels
    ax_main.set_xlim(x_min_plot, x_max_plot)
    ax_main.set_ylim(y_min_visible, y_samp.max() * 1.02)
    
    ax_main.set_xlabel("Scrublet Score", fontweight="bold")
    ax_main.set_ylabel("Total UMI Counts (log10)", fontweight="bold")
    
    # Only show legend if there are items to show
    handles, labels = ax_main.get_legend_handles_labels()
    if handles:
        ax_main.legend(handles, labels, loc="upper left")

    fig.suptitle(f"Phase 4 · Doublet Detection | {adata.n_obs:,} cells | {pred_mask.sum():,} predicted doublets", 
                 fontweight="bold", fontsize=15, y=0.94)
    
    # Safely handle the save/show logic
    try:
        path = _save(fig, cfg, "p04_doublets")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 5 · Escaper summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_perturbation_efficiency(adata, cfg, kde_levels=9, kde_smooth=0.8, scatter_alpha=1.0):
    _apply_theme()
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    import numpy as np
    import pandas as pd
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    
    if "knockdown_efficiency" not in adata.uns:
        print("No efficiency metrics found. Run Phase 5 first.")
        return None, None

    # 1. Extract Data
    eff_dict = adata.uns["knockdown_efficiency"]
    df = pd.DataFrame.from_dict(eff_dict, orient="index")
    df.index.name = "perturbation"
    df = df.reset_index()
    df = df.dropna(subset=["knockdown_depth"])
    
    if len(df) == 0:
        print("No valid perturbation data to plot.")
        return None, None

    # 2. Setup Thresholds
    p5 = cfg.get("phase5_escaper_filtering", {})
    eff_thresh = p5.get("efficiency_threshold", 0.50)
    
    x = df["knockdown_depth"].values
    y = df["efficiency_score"].values
    
    x_min = max(0, x.min() - 0.05)
    x_max = max(1.1, x.max() + 0.05)
    y_min, y_max = max(0, min(y.min() - 0.1, eff_thresh - 0.05)), 1.02

    # 3. Setup Canvas
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # 4. Danger Zone Shading
    danger_rect = patches.Rectangle(
        (0, y_min),                 
        x_max,                      
        eff_thresh - y_min,         
        linewidth=0, facecolor=_THEME["accent"], alpha=0.15, zorder=1
    )
    ax_main.add_patch(danger_rect)

    # 5. Reference Lines
    ax_main.axhline(eff_thresh, color=_THEME["accent"], ls="--", lw=1.5, zorder=2, label=f"Min Efficiency: {eff_thresh*100:.0f}%")
    ax_main.axvline(1.0, color=_THEME["grid"], ls=":", lw=1.5, zorder=2, label="No Effect (1.0)")

    # 6. Main Scatter (Below Contours)
    pass_mask = y >= eff_thresh
    if pass_mask.any():
        ax_main.scatter(x[pass_mask], y[pass_mask], s=60, color=_THEME["low"], alpha=scatter_alpha, 
                        edgecolor=_THEME["bg"], linewidth=0.5, zorder=3, label="Pass")
    if (~pass_mask).any():
        ax_main.scatter(x[~pass_mask], y[~pass_mask], s=80, color=_THEME["accent"], alpha=0.9, 
                        edgecolor=_THEME["bg"], linewidth=1.0, zorder=4, label="Low Efficiency")

    # 7. Topological Contours (ON TOP of Scatter)
    # Using 'mid' so it contrasts well over the scatter points without overpowering the plot
    sns.kdeplot(x=x, y=y, levels=kde_levels, bw_adjust=kde_smooth, color=_THEME["dark"], 
                linewidths=1.2, ax=ax_main, zorder=5, alpha=0.75)

    # 8. Marginals
    sns.kdeplot(x=x, ax=ax_top, color=_THEME["high"], fill=True, alpha=0.8, lw=1.5)
    sns.kdeplot(y=y, ax=ax_right, color=_THEME["high"], fill=True, alpha=0.8, lw=1.5)
    
    ax_top.axis("off")
    ax_right.axis("off")

    # 9. Clamping and Labels
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)
    
    direction = p5.get("direction", "auto")
    if direction == "auto":
        pert_type = cfg.get("dataset", {}).get("perturbation_type", "CRISPRi")
        direction = "activation" if "CRISPRa" in str(pert_type) else "knockdown"
        
    x_label = "Knockdown Depth (0 = Perfect KO, 1 = No Effect)" if direction == "knockdown" else "Activation Fold Change (>1 = Increased Expression)"

    ax_main.set_xlabel(x_label, fontweight="bold")
    ax_main.set_ylabel("Efficiency Score (Fraction of Cells Retained)", fontweight="bold")
    
    ax_main.legend(loc="upper right", frameon=True, facecolor=_THEME["bg"], edgecolor="none", framealpha=0.8)

    # 10. Annotate worst guide if struggling (Top Z-Order)
    if len(df) > 0:
        worst_idx = df["efficiency_score"].idxmin()
        worst_row = df.loc[worst_idx]
        if worst_row["efficiency_score"] < 0.80:
            ax_main.annotate(
                f"{worst_row['perturbation']}",
                xy=(worst_row["knockdown_depth"], worst_row["efficiency_score"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=9, fontweight="bold", color=_THEME["accent"], zorder=6
            )

    fig.suptitle(f"Phase 5 · Guide Performance Landscape | {len(df)} targets", 
                 fontweight="bold", fontsize=15, y=0.94)
    
    # Safely handle the save/show logic
    try:
        path = _save(fig, cfg, "p05_efficiency_scatter")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 6 · Gene expression histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase6_diagnostics(adata, cfg):
    _apply_theme()
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    
    # ── Fallback/Safety Checks ──
    if "phase6_waterfall" not in adata.uns:
        print("No Phase 6 waterfall metadata found. Run Phase 6 first.")
        return None, None

    waterfall_dict = adata.uns["phase6_waterfall"]
    
    # Handle the .uns vs .var data location
    if "phase6_gene_penetrance" in adata.uns:
        orig_pct_cells = adata.uns["phase6_gene_penetrance"]
    elif "pct_cells_expressed" in adata.var:
        orig_pct_cells = adata.var["pct_cells_expressed"].values
        print("Note: Penetrance data loaded from .var. Filtered genes will not be shown.")
    else:
        print("No penetrance data found.")
        return None, None

    # ═════════════════════════════════════════════════════════════════
    #  MASTER CANVAS SETUP
    # ═════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)
    
    # ── PANEL A: Triage Waterfall ──
    ax_wf = fig.add_subplot(gs[0])

    stages = list(waterfall_dict.keys())
    counts = list(waterfall_dict.values())

    # Map colors: Dark (Start), Mid (Intermediate), Accent (Final)
    colors = [_THEME["low"]] + [_THEME["mid"]] * (len(stages) - 2) + [_THEME["high"]]

    y_pos = np.arange(len(stages))
    bars = ax_wf.barh(y_pos, counts, color=colors, edgecolor="none", height=0.6)
    ax_wf.invert_yaxis()

    ax_wf.set_yticks(y_pos)
    ax_wf.set_yticklabels(stages, fontweight="bold", color=_THEME["text"])
    ax_wf.set_xlabel("Number of Genes", fontweight="bold")
    ax_wf.set_title("A  Gene Survival Waterfall", loc="left", fontweight="bold", fontsize=14)

    for bar in bars:
        width = bar.get_width()
        if width < (max(counts) * 0.15):
            ax_wf.text(width + (max(counts) * 0.02), bar.get_y() + bar.get_height()/2,
                       f"{int(width):,}", va='center', ha='left', color=_THEME["text"], fontweight='bold')
        else:
            ax_wf.text(width - (max(counts) * 0.02), bar.get_y() + bar.get_height()/2,
                       f"{int(width):,}", va='center', ha='right', color=_THEME["bg"], fontweight='bold')

    ax_wf.spines["top"].set_visible(False)
    ax_wf.spines["right"].set_visible(False)
    ax_wf.spines["bottom"].set_color(_THEME["grid"])
    ax_wf.spines["left"].set_color(_THEME["grid"])

    # ── PANEL B: Pre-Filter Transcriptome Penetrance ──
    ax_hist = fig.add_subplot(gs[1])
    
    p6 = cfg.get("phase6_gene_triage", {})
    pct_thresh = p6.get("pct_filter", 0.01) * 100
    
    clip_val = 25.0
    df = pd.DataFrame({
        "penetrance": np.clip(orig_pct_cells, 0, clip_val),
        "status": ["Kept" if x >= pct_thresh else "Filtered" for x in orig_pct_cells]
    })
    
    # Theme palette mapping
    palette = {"Kept": _THEME["dark"], "Filtered": _THEME["mid"]}
    
    sns.histplot(data=df, x="penetrance", hue="status", palette=palette, 
                 bins=60, multiple="stack", edgecolor=_THEME["bg"], alpha=0.9, ax=ax_hist)
    
    ax_hist.axvline(pct_thresh, color=_THEME["accent"], ls="--", lw=2, label=f"Filter Limit ({pct_thresh}%)")
    
    ax_hist.set_xlim(0, clip_val)
    ax_hist.set_title("B  Transcriptome Sparsity Distribution", loc="left", fontweight="bold", fontsize=14)
    ax_hist.set_xlabel("Percentage of Cells Expressing Gene", fontweight="bold")
    ax_hist.set_ylabel("Number of Genes", fontweight="bold")
    
    if (orig_pct_cells > clip_val).any():
        n_high = (orig_pct_cells > clip_val).sum()
        ax_hist.annotate(f"+{n_high:,} genes\nabove {clip_val}%", 
                         xy=(0.95, 0.85), xycoords='axes fraction', 
                         ha='right', va='top', color=_THEME["text"], fontweight='bold', fontsize=10)
                         
    ax_hist.legend(loc="upper right", frameon=True, facecolor=_THEME["bg"], edgecolor="none")
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    fig.suptitle(f"Phase 6 · Feature Selection | {len(orig_pct_cells):,} Starting Genes", 
                 fontweight="bold", fontsize=16, y=1.02)
    
    # Safely handle the save/show logic
    try:
        path = _save(fig, cfg, "p06_triage_dashboard")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 7 · Split summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase7_splits(split_result, cfg):
    _apply_theme()
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker
    import seaborn as sns
    import pandas as pd
    import numpy as np

    if not split_result or "split_info" not in split_result[0]:
        print("No valid split data provided. Run Phase 7 first.")
        return None, None

    res = split_result[0]
    
    train_cells = res["split_info"]["train"]
    val_cells   = res["split_info"]["val"]
    test_cells  = res["split_info"]["test"]
    total_cells = train_cells + val_cells + test_cells
    
    deg_counts  = res["deg_counts"]
    train_lbls  = set(res["train_labels"])
    val_lbls    = set(res["val_labels"])
    test_lbls   = set(res["test_labels"])
    pert_col    = cfg.get("dataset", {}).get("perturbation_col", "perturbation")

    fig = plt.figure(figsize=(18, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 1.2], wspace=0.25)
    
    # ── STRICT COLOR MAPPING ──
    palette = {"Train": _THEME["low"], "Val": _THEME["mid"], "Test": _THEME["accent"]}
    colors = [_THEME["low"], _THEME["mid"], _THEME["accent"]]
    split_order = ["Train", "Val", "Test"]
    split_map = {"Train": train_lbls, "Val": val_lbls, "Test": test_lbls}

    # ═════════════════════════════════════════════════════════════════
    # PANEL A: Cell Allocation
    # ═════════════════════════════════════════════════════════════════
    ax_bar = fig.add_subplot(gs[0])
    
    counts = [train_cells, val_cells, test_cells]
    
    y_pos = np.arange(len(split_order))
    ax_bar.barh(y_pos, counts, color=colors, edgecolor="none", height=0.6)
    ax_bar.invert_yaxis() 
    
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(split_order, fontweight="bold", color=_THEME["text"])
    ax_bar.set_xlabel("Number of Cells", fontweight="bold")
    ax_bar.set_title("A  Cell count by split", loc="left", fontweight="bold", fontsize=14)

    # Convert large X-axis numbers to "K" format to prevent clipping
    ax_bar.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}K" if x >= 1000 else str(int(x))))

    for i, count in enumerate(counts):
        pct = (count / total_cells) * 100
        ax_bar.text(count + (max(counts) * 0.02), i,
                    f"{int(count):,} ({pct:.1f}%)",
                    va='center', ha='left', color=_THEME["text"], fontweight='bold')
        
    ax_bar.set_xlim(0, max(counts) * 1.35) 
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["bottom"].set_color(_THEME["grid"])
    ax_bar.spines["left"].set_color(_THEME["grid"])

    # ═════════════════════════════════════════════════════════════════
    # PANEL B: Perturbation Size Distributions
    # ═════════════════════════════════════════════════════════════════
    ax_kde = fig.add_subplot(gs[1])

    sizes_data = []
    for split_name, key in [("Train", "train"), ("Val", "val"), ("Test", "test")]:
        if key in res and res[key] is not None:
            vc = res[key].obs[pert_col].value_counts()
            for target in split_map[split_name]:
                if target in vc:
                    sizes_data.append({"Split": split_name, "Size": vc[target]})

    df_sizes = pd.DataFrame(sizes_data)
    if not df_sizes.empty:
        # THE FIX: Added hue_order to force strict color mapping
        sns.kdeplot(data=df_sizes, x="Size", hue="Split", hue_order=split_order, palette=palette,
                    fill=True, common_norm=False, alpha=0.5, linewidth=2, 
                    clip=(0, None), ax=ax_kde)

    ax_kde.set_title("B  Perturbation size distributions", loc="left", fontweight="bold", fontsize=14)
    ax_kde.set_xlabel("Cells per perturbation", fontweight="bold")
    ax_kde.set_ylabel("Density", fontweight="bold")
    ax_kde.spines["top"].set_visible(False)
    ax_kde.spines["right"].set_visible(False)

    # ═════════════════════════════════════════════════════════════════
    # PANEL C: Zero-Shot Stratification 
    # ═════════════════════════════════════════════════════════════════
    ax_vio = fig.add_subplot(gs[2])

    df_strat = pd.DataFrame({"target": deg_counts.index, "shift": deg_counts.values})
    
    def get_split(t):
        if t in train_lbls: return "Train"
        if t in val_lbls: return "Val"
        if t in test_lbls: return "Test"
        return "Unknown"
        
    df_strat["Split"] = df_strat["target"].apply(get_split)
    
    # THE FIX: Added order parameter to force strict x-axis and color mapping
    sns.violinplot(data=df_strat, x="Split", y="shift", order=split_order, palette=palette, 
                   inner="quartile", alpha=0.9, linecolor= _THEME["spine"], linewidth=1.5, ax=ax_vio)
                  
    ax_vio.set_title("C  Zero-Shot Stratification", loc="left", fontweight="bold", fontsize=14)
    ax_vio.set_xlabel("")
    ax_vio.set_ylabel("Perturbation Severity (Mean Expression Shift)", fontweight="bold")
    ax_vio.spines["top"].set_visible(False)
    ax_vio.spines["right"].set_visible(False)
    
    fig.suptitle(f"Phase 7 · Data Splits Summary | {total_cells:,} total cells", 
                 fontweight="bold", fontsize=16, y=1.05)
    
    fig.tight_layout()
    
    # Safely handle the save/show logic
    try:
        path = _save(fig, cfg, "p07_splits_dashboard")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 8 · HVG mean-variance
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase8_hvg(adata, cfg):
    _apply_theme()
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import pandas as pd

    if "hvg_stats" not in adata.uns:
        print("No HVG stats found. Run Phase 8 first.")
        return None, None

    hvg_df = pd.DataFrame.from_dict(adata.uns["hvg_stats"], orient="index")
    
    if "variances_norm" in hvg_df.columns:
        y_col = "variances_norm"
        y_label = "Normalized Variance (seurat_v3)"
    elif "dispersions_norm" in hvg_df.columns:
        y_col = "dispersions_norm"
        y_label = "Normalized Dispersion"
    else:
        return None, None

    hvg_df = hvg_df.dropna(subset=["means", y_col])

    # ── Assign Status Tags matching our robust diagnostics ──
    hvg_df["Status"] = "Non-Variable"
    hvg_df.loc[hvg_df["highly_variable"], "Status"] = "Highly Variable (Core)"
    
    if "rescued_target" in hvg_df.columns:
        hvg_df.loc[hvg_df["rescued_target"], "Status"] = "Target Rescue (Swapped)"
    if "rescued_ghost" in hvg_df.columns:
        hvg_df.loc[hvg_df["rescued_ghost"], "Status"] = "Cell Cycle Rescue (Stacked)"

    palette = {
        "Non-Variable": _THEME["dark"], 
        "Highly Variable (Core)": _THEME["low"], 
        "Target Rescue (Swapped)": _THEME["high"],
        "Cell Cycle Rescue (Stacked)": _THEME["high"]
    }
    order = ["Non-Variable", "Highly Variable (Core)", "Target Rescue (Swapped)", "Cell Cycle Rescue (Stacked)"]
    present_order = [s for s in order if s in hvg_df["Status"].unique()]

    # Dynamic Y-Axis Scaling
    y_min = hvg_df[y_col].min()
    y_max = hvg_df[y_col].max()
    y_pad = (y_max - y_min) * 0.05
    y_lower = y_min - y_pad
    y_upper = y_max + y_pad

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.2, 1], width_ratios=[1, 0.2], wspace=0.03, hspace=0.03)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_marg_x = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # ═════════════════════════════════════════════════════════════════
    # MAIN SCATTER
    # ═════════════════════════════════════════════════════════════════
    non_hvg = hvg_df[hvg_df["Status"] == "Non-Variable"]
    ax_main.scatter(non_hvg["means"], non_hvg[y_col], color=_THEME["dark"], alpha=0.3, s=8, label="Non-Variable")

    hvg = hvg_df[hvg_df["Status"] == "Highly Variable (Core)"]
    ax_main.scatter(hvg["means"], hvg[y_col], color=_THEME["low"], alpha=0.6, s=15, label="Highly Variable (Core)")

    # Target Rescues (Stars)
    t_res = hvg_df[hvg_df["Status"] == "Target Rescue (Swapped)"]
    if not t_res.empty:
        ax_main.scatter(t_res["means"], t_res[y_col], color=_THEME["high"], alpha=1.0, s=80, 
                     marker='*', edgecolor=_THEME["bg"], linewidth=0.5, label=f"Target Rescue ({len(t_res)})")

    # Ghost/Cell Cycle Rescues (Diamonds)
    g_res = hvg_df[hvg_df["Status"] == "Cell Cycle Rescue (Stacked)"]
    if not g_res.empty:
        ax_main.scatter(g_res["means"], g_res[y_col], color=_THEME["high"], alpha=0.9, s=30, 
                     marker='D', edgecolor=_THEME["bg"], linewidth=0.5, label=f"Cell Cycle Rescue ({len(g_res)})")

    ax_main.set_xscale("log")
    ax_main.set_ylim(bottom=y_lower, top=y_upper)
    ax_main.axhline(0, color=_THEME["grid"], linestyle="--", linewidth=1.5, alpha=0.8)
    
    ax_main.text(0.01, 0.02, f"Axis Floor: {y_lower:.2f}", transform=ax_main.transAxes, 
               fontsize=10, color=_THEME["text"], va="bottom", ha="left", fontweight="bold")

    # ═════════════════════════════════════════════════════════════════
    # MARGINALS (Colored by Status)
    # ═════════════════════════════════════════════════════════════════
    sns.kdeplot(data=hvg_df, x="means", hue="Status", palette=palette, hue_order=present_order,
                fill=True, alpha=0.5, common_norm=False, legend=False, ax=ax_marg_x, warn_singular=False, cut=0)
    
    sns.kdeplot(data=hvg_df, y=y_col, hue="Status", palette=palette, hue_order=present_order,
                fill=True, alpha=0.5, common_norm=False, legend=False, ax=ax_marg_y, warn_singular=False, cut=0)

    # Styling & Labels
    ax_marg_x.axis("off")
    ax_marg_y.axis("off")
    
    ax_main.set_xlabel("Mean Expression (log scale)", fontweight="bold")
    ax_main.set_ylabel(y_label, fontweight="bold")
    
    ax_marg_x.set_title("Phase 8 · Mean-Variance Distribution", loc="left", fontweight="bold", fontsize=16, pad=15)
    
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    
    legend = ax_main.legend(loc="upper left", frameon=True, facecolor=_THEME["bg"], edgecolor=_THEME["grid"])
    for handle in legend.legend_handles: handle.set_alpha(1.0)

    # Safely handle the save/show logic
    try:
        path = _save(fig, cfg, "p08_hvg_dashboard")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 9 · Normalization
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase9_normalization(p9_cache, cfg):
    _apply_theme()
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if not p9_cache or "raw" not in p9_cache:
        print("No cache data provided. Run Phase 9 first.")
        return None, None

    # Setup canvas (letting _apply_theme handle the background colors globally)
    fig, ax = plt.subplots(figsize=(10, 6))

    raw = np.array(p9_cache["raw"])
    norm = np.array(p9_cache["norm"])

    # Apply log1p visually to the raw data so it shares the same scale as the post-data
    raw_log = np.log1p(raw)

    # Apply global theme colors
    sns.kdeplot(raw_log, color=_THEME["dark"], fill=True, alpha=0.85, label="Pre-Normalization (Raw Log1p)", ax=ax, clip=(0, None))
    sns.kdeplot(norm, color=_THEME["high"], fill=True, alpha=0.6, label="Post-Normalization (CP10k Log1p)", ax=ax, clip=(0, None))

    ax.set_title("Phase 9 · Global Expression Harmonization", loc="left", fontweight="bold", fontsize=14)
    ax.set_xlabel("Log1p(Expression)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    legend = ax.legend(frameon=True, facecolor=_THEME["bg"], edgecolor=_THEME["grid"])
    for handle in legend.legend_handles: 
        handle.set_alpha(1.0)
    
    fig.tight_layout()
    
    # Safely handle the save/show logic
    try:
        path = _save(fig, cfg, "p09_normalization")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 10 · UMAP (layered bivariate)
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase10_integration(adata_p10, cfg):
    _apply_theme()
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scanpy as sc
    import pandas as pd

    batch_col = cfg.get("dataset", {}).get("batch_col", "gem_group")
    
    if batch_col not in adata_p10.obs.columns:
        print(f"CRITICAL: Column '{batch_col}' not found.")
        return None, None

    # Set your dynamic limit here
    umap_cell_count = 50000
    
    print(f"Subsampling to {umap_cell_count:,} cells for rapid UMAP visualization...")
    viz_adata = adata_p10.copy()
    if viz_adata.n_obs > umap_cell_count:
        sc.pp.subsample(viz_adata, n_obs=umap_cell_count, random_state=42)

    # Let the global theme handle the background
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ─── THE MAKO GRADIENT FIX ───────────────────────────────────────────
    # Sort the batches so the gradient applies linearly across sample 1-16
    batches = sorted(viz_adata.obs[batch_col].unique())
    
    # Extract equidistant points directly from the Mako colormap
    colors = sns.color_palette("mako", n_colors=len(batches))
    batch_palette = dict(zip(batches, colors))

    # Panel A: Uncorrected
    print("Computing Uncorrected UMAP...")
    sc.pp.neighbors(viz_adata, use_rep="X_pca", n_neighbors=15, n_pcs=50)
    sc.tl.umap(viz_adata, min_dist=0.3)
    
    df_uncorrected = pd.DataFrame(viz_adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
    df_uncorrected["Batch"] = viz_adata.obs[batch_col].values
    
    # Shuffle the dataframe so no single sample dominates the top layer (Z-order fix)
    df_uncorrected = df_uncorrected.sample(frac=1, random_state=42)

    sns.scatterplot(data=df_uncorrected, x="UMAP1", y="UMAP2", hue="Batch", 
                    palette=batch_palette, s=5, alpha=0.6, edgecolor="none", ax=axes[0], legend=False)
    
    axes[0].set_title("A  Uncorrected (Raw PCA)", loc="left", fontweight="bold", fontsize=14)
    axes[0].set_xlabel("UMAP 1", fontweight="bold")
    axes[0].set_ylabel("UMAP 2", fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].spines["bottom"].set_color(_THEME["grid"])
    axes[0].spines["left"].set_color(_THEME["grid"])

    # Panel B: Harmony Corrected
    if "X_pca_harmony" in viz_adata.obsm:
        print("Computing Harmony Corrected UMAP...")
        sc.pp.neighbors(viz_adata, use_rep="X_pca_harmony", n_neighbors=15, n_pcs=50)
        sc.tl.umap(viz_adata, min_dist=0.3)
        
        df_harmony = pd.DataFrame(viz_adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
        df_harmony["Batch"] = viz_adata.obs[batch_col].values
        
        # Shuffle the dataframe for Z-order
        df_harmony = df_harmony.sample(frac=1, random_state=42)

        sns.scatterplot(data=df_harmony, x="UMAP1", y="UMAP2", hue="Batch", 
                        palette=batch_palette, s=5, alpha=0.6, edgecolor="none", ax=axes[1])
        
        axes[1].set_title("B  Harmony Integrated", loc="left", fontweight="bold", fontsize=14)
        axes[1].set_xlabel("UMAP 1", fontweight="bold")
        axes[1].set_ylabel("UMAP 2", fontweight="bold")
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].spines["bottom"].set_color(_THEME["grid"])
        axes[1].spines["left"].set_color(_THEME["grid"])
        
        axes[1].legend(title="Batch / GEM Group", loc="center left", bbox_to_anchor=(1, 0.5),
                       frameon=True, facecolor=_THEME["bg"], edgecolor=_THEME["grid"], markerscale=3)
    else:
        axes[1].text(0.5, 0.5, "Harmony Embeddings Not Found", ha="center", va="center", color=_THEME["text"])

    fig.tight_layout()
    
    # Safely handle the save/show logic
    try:
        path = _save(fig, cfg, "p10_integration_dashboard")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 11 · Cell line PCA (layered bivariate with marginals)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cell_line_pca(adata, cfg):
    _apply_theme()
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    emb_key = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    if emb_key not in adata.obsm:
        print(f"Embedding {emb_key} not found.")
        return None, None

    pc1 = adata.obsm[emb_key][:, 0]
    pc2 = adata.obsm[emb_key][:, 1]

    df = pd.DataFrame({"PC1": pc1, "PC2": pc2})

    # Subsample purely for rendering speed
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    # ─── THE JOINT GRID ──────────────────────────────────────────────────
    g = sns.JointGrid(data=df, x="PC1", y="PC2", height=8, ratio=5, space=0.1)

    # 1. Scatter tails (Using our theme's mid-tone so it fades smoothly)
    sns.scatterplot(x=df["PC1"], y=df["PC2"], s=5, color=_THEME["dark"], alpha=0.9, ax=g.ax_joint)
    
    # 2. High-Resolution Pixelated mako histogram
    sns.histplot(x=df["PC1"], y=df["PC2"], bins=120, pthresh=.05, cmap="mako", alpha=0.9, ax=g.ax_joint)
    
    # 3. High-Density, Unsmoothed Contours (Using theme background color)
    sns.kdeplot(x=df["PC1"], y=df["PC2"], levels=8, bw_adjust=0.5, color=_THEME["bg"], linewidths=0.8, ax=g.ax_joint)

    # 4. The Marginals (Density curves)
    sns.kdeplot(x=df["PC1"], fill=True, color=_THEME["high"], alpha=0.8, bw_adjust=0.5, ax=g.ax_marg_x)
    sns.kdeplot(y=df["PC2"], fill=True, color=_THEME["high"], alpha=0.8, bw_adjust=0.5, ax=g.ax_marg_y)

    # ─── FORMATTING ──────────────────────────────────────────────────────
    title_emb = "Harmony-Corrected PCA" if "harmony" in emb_key else "Standard PCA"
    g.figure.suptitle(f"Phase 11: Global Bivariate Topology\n{title_emb}", fontsize=15, fontweight="bold", y=1.03, color=_THEME["text"])
    g.ax_joint.set_xlabel("Principal Component 1", fontweight="bold", color=_THEME["text"])
    g.ax_joint.set_ylabel("Principal Component 2", fontweight="bold", color=_THEME["text"])

    # Safely handle the save/show logic
    try:
        path = _save(g.figure, cfg, "p11_pca_2d")
        _show(g.figure)
    except NameError:
        path = None
        
    return g.figure, path

def plot_cell_line_pca_3d(adata, cfg):
    import pandas as pd
    import seaborn as sns
    import plotly.express as px

    emb_key = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
    if emb_key not in adata.obsm or "sporeplus_cell_line" not in adata.obs.columns:
        print("Required embeddings or labels not found.")
        return None, None

    if adata.obsm[emb_key].shape[1] < 3:
        print("Less than 3 Principal Components available.")
        return None, None

    pc1 = adata.obsm[emb_key][:, 0]
    pc2 = adata.obsm[emb_key][:, 1]
    pc3 = adata.obsm[emb_key][:, 2]
    labels = adata.obs["sporeplus_cell_line"].astype(str).values

    df = pd.DataFrame({"PC1": pc1, "PC2": pc2, "PC3": pc3, "Cell Line": labels})

    # Subsample to 50k to ensure butter-smooth 60fps rotation in the browser
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    # Extract exactly the right number of Mako colors to keep them equidistant
    unique_lines = sorted(df["Cell Line"].unique())
    colors = sns.color_palette("mako", n_colors=len(unique_lines)).as_hex()
    color_map = dict(zip(unique_lines, colors))

    title_emb = "Harmony-Corrected PCA" if "harmony" in emb_key else "Standard PCA"
    
    # Try to resolve _THEME colors safely for text and grid lines
    try:
        text_color = _THEME["text"]
        grid_color = _THEME["grid"]
    except NameError:
        text_color = "#2a2f4d"
        grid_color = "#e0e0e0"

    # Build the WebGL Plotly Figure
    fig = px.scatter_3d(
        df, x="PC1", y="PC2", z="PC3",
        color="Cell Line",
        color_discrete_map=color_map,
        title=f"Phase 11: Interactive 3D Latent Space Topology<br><sup>{title_emb}</sup>",
        opacity=0.75
    )

    # Format the markers to be tight without borders
    fig.update_traces(marker=dict(size=3, line=dict(width=0)))
    
    # ── THE FIX ──────────────────────────────────────────────────────────────
    fig.update_layout(
        width=900,   # Forces the wider 3:2 Aspect Ratio
        height=600,
        paper_bgcolor='rgba(0,0,0,0)', # Completely transparent outer background
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_color),
        legend=dict(
            title=dict(text="Detected Cell Lines", font=dict(size=14, color=text_color)),
            itemsizing='constant',  # Forces huge legend dots
            font=dict(size=13, color=text_color),
            bgcolor='rgba(0,0,0,0)', # Transparent legend prevents white-box collisions
            bordercolor=grid_color,
            borderwidth=0,
            yanchor="top",
            y=0.85,
            xanchor="left",
            x=1.05
        ),
        margin=dict(l=0, r=0, b=10, t=60),
        scene=dict(
            # showbackground=False removes the harsh solid walls behind the grid
            xaxis=dict(title=dict(text="PC 1", font=dict(size=14)), gridcolor=grid_color, showbackground=False),
            yaxis=dict(title=dict(text="PC 2", font=dict(size=14)), gridcolor=grid_color, showbackground=False),
            zaxis=dict(title=dict(text="PC 3", font=dict(size=14)), gridcolor=grid_color, showbackground=False)
        )
    )

    fig.show()
    return fig, None

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 12 · Metacell QC
# ─────────────────────────────────────────────────────────────────────────────

def plot_phase12_metacell_dashboard(all_meta_splits, cfg):
    _apply_theme()
    import pandas as pd
    import numpy as np
    import anndata as ad
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import TruncatedSVD
    import math

    cell_lines = [cl for cl in all_meta_splits.keys() if cl != "single"]
    if not cell_lines:
        cell_lines = ["single"]
    
    n_lines = len(cell_lines)
    
    # THE FIX: Fully dynamic grid sizing
    cols = min(3, n_lines)
    rows = math.ceil(n_lines / cols)
    fig = plt.figure(figsize=(7 * cols, 7 + 5 * rows))

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 1: The Yield Bar Chart (Top Row)
    # ════════════════════════════════════════════════════════════════════════
    # Dynamically span exactly the number of columns we actually have
    ax_bar = plt.subplot2grid((rows + 1, cols), (0, 0), colspan=cols)

    records = []
    for cl, splits in all_meta_splits.items():
        for split_name, adata in splits.items():
            records.append({"Cell Line": cl, "Split": split_name.capitalize(), "Metacells": adata.n_obs})
    
    df_counts = pd.DataFrame(records)
    split_palette = {"Train": _THEME["dark"], "Val": _THEME["mid"], "Test": _THEME["accent"]}

    sns.barplot(data=df_counts, x="Cell Line", y="Metacells", hue="Split",
                palette=split_palette, ax=ax_bar, edgecolor=_THEME["bg"], linewidth=1.5)

    ax_bar.set_title("Phase 12: Metacell Yield per Cell Line & Split", fontweight="bold", fontsize=16, pad=15)
    ax_bar.set_ylabel("Total Metacells", fontweight="bold")
    ax_bar.set_xlabel("")
    
    for p in ax_bar.patches:
        height = p.get_height()
        if pd.notna(height) and height > 0:
            ax_bar.annotate(f'{int(height):,}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=10, color=_THEME["text"],
                            xytext=(0, 3), textcoords='offset points', rotation=45)
            
    ax_bar.legend(title="Split", frameon=True, facecolor=_THEME["bg"], edgecolor=_THEME["grid"])
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ════════════════════════════════════════════════════════════════════════
    # PANEL 2: The Manifold Condensation Grid (Bottom Rows)
    # ════════════════════════════════════════════════════════════════════════
    dataset_name = cfg.get("dataset", {}).get("name", "dataset")
    splits_dir   = cfg.get("paths", {}).get("_splits", cfg.get("paths", {}).get("splits_dir"))

    try:
        train_p9_backed = ad.read_h5ad(splits_dir / f"{dataset_name}_train_p9.h5ad", backed='r')
        sc_labels = train_p9_backed.obs.get("sporeplus_cell_line", train_p9_backed.obs.get("sample"))

        for i, cl in enumerate(cell_lines):
            row_idx = (i // cols) + 1
            col_idx = i % cols
            ax = plt.subplot2grid((rows + 1, cols), (row_idx, col_idx))

            # ── OOM FIREWALL: Subsample indices BEFORE loading into memory ──
            cl_indices = np.where(sc_labels == cl)[0]
            
            # Cap the background cloud to 30k cells max
            if len(cl_indices) > 30000:
                rng = np.random.default_rng(42)
                safe_indices = rng.choice(cl_indices, size=30000, replace=False)
                safe_indices = np.sort(safe_indices) # H5py requires strictly sorted indices
            else:
                safe_indices = cl_indices
                
            # Safely load ONLY the allowed cells into active memory
            sc_adata = train_p9_backed[safe_indices, :].to_memory()
            
            mc_adata = all_meta_splits[cl].get("train")
            if mc_adata is None:
                continue

            # Align feature space before projection
            sc_adata = sc_adata[:, mc_adata.var_names]

            svd = TruncatedSVD(n_components=2, random_state=42)
            sc_pc = svd.fit_transform(sc_adata.X)
            mc_pc = svd.transform(mc_adata.X)

            ax.scatter(sc_pc[:, 0], sc_pc[:, 1], color=_THEME["grid"], s=5, alpha=0.3, edgecolors='none', label="Single Cells")
            ax.scatter(mc_pc[:, 0], mc_pc[:, 1], color=_THEME["accent"], s=35, alpha=0.9, edgecolors=_THEME["bg"], linewidth=0.8, label="Metacells")

            ax.set_title(f"{cl} Manifold", fontweight="bold", fontsize=13)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.legend(frameon=True, facecolor=_THEME["bg"], edgecolor=_THEME["grid"], loc="best")

    except Exception as e:
        print(f"Failed to render manifolds: {e}")

    fig.tight_layout()
    
    try:
        path = _save(fig, cfg, "p12_metacell_dashboard")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

def plot_phase12_size_distribution(all_meta_splits, cfg):
    _apply_theme()
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as mpatches

    records = []
    for cl, splits in all_meta_splits.items():
        if cl == "single": 
            continue
        for split_name, adata in splits.items():
            if "n_cells_in_metacell" in adata.obs:
                sizes = adata.obs["n_cells_in_metacell"].values
                for s in sizes:
                    records.append({
                        "Cell Line": cl, 
                        "Split": split_name.capitalize(), 
                        "Cells per Metacell": s
                    })

    df = pd.DataFrame(records)
    if df.empty:
        print("No 'n_cells_in_metacell' data found in the objects.")
        return None, None

    split_palette = {"Train": _THEME["low"], "Val": _THEME["mid"], "Test": _THEME["accent"]}

    # ─── THE FIX: Reverse the hue_order ───────────────────────────────────
    # Seaborn draws layers in reverse order. By putting "Val" first, 
    # Seaborn draws it last (putting it on top). "Train" is drawn first (in the back).
    draw_order = ["Val", "Test", "Train"]

    g = sns.FacetGrid(df, col="Cell Line", col_wrap=3, height=4, aspect=1.2, sharex=False, sharey=False)
    
    g.map_dataframe(
        sns.histplot, x="Cells per Metacell", hue="Split", 
        hue_order=draw_order, 
        palette=split_palette, element="step", stat="count", 
        common_norm=False, alpha=0.8, linewidth=1.5, bins=15
    )

    g.set_titles(col_template="{col_name}", fontweight="bold", size=13)
    g.set_axis_labels("Single Cells per Metacell", "Count of Metacells", fontweight="bold")
    
    if g.legend:
        g.legend.remove()

    # The legend visually remains in chronological order
    legend_handles = [
        mpatches.Patch(color=split_palette["Train"], label="Train", alpha=0.9),
        mpatches.Patch(color=split_palette["Val"], label="Val", alpha=0.7),
        mpatches.Patch(color=split_palette["Test"], label="Test", alpha=0.7)
    ]
    
    g.figure.legend(handles=legend_handles, title="Split", frameon=True, 
                    facecolor=_THEME["bg"], edgecolor=_THEME["grid"],
                    bbox_to_anchor=(1.02, 0.5), loc='center left', 
                    title_fontproperties={'weight': 'bold'})
    
    g.figure.subplots_adjust(top=0.88, hspace=0.3)
    g.figure.suptitle("Phase 12: Dynamic Graining Size Distribution", fontsize=16, fontweight="bold")

    try:
        path = _save(g.figure, cfg, "p12_size_distribution")
        _show(g.figure)
    except NameError:
        path = None

    return g.figure, path

# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 13 · CHITIN diagnostics (replaces useless rectangle plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_chitin_pareto_grid(all_chitin, cfg):
    """
    Renders a 2D Pareto Optimization grid for all cell lines in the dataset.
    - X: Rank Disruption
    - Y: Discrimination Ratio
    - Size: Proximity to the Pareto Front
    - Shape: Distance Metric
    - Color Saturation: Signal Stability (mako_r: light/transparent = low, dark/navy = high)
    - Highlight: Theme 'accent' for the True Winner across all modes
    """
    _apply_theme()
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from scipy.stats import rankdata

    cell_lines = list(all_chitin.keys())
    n_lines = len(cell_lines)
    if n_lines == 0: return None, None

    cols = min(3, n_lines)
    rows = math.ceil(n_lines / cols)
    
    # ── THE ULTIMATE GRIDSPEC FIX ──
    fig = plt.figure(figsize=(6 * cols + 1.5, 5 * rows + 1.5))
    
    gs = fig.add_gridspec(rows + 1, cols + 1, 
                          width_ratios=[1]*cols + [0.05], 
                          height_ratios=[1]*rows + [0.15],
                          wspace=0.25, hspace=0.35)
    
    # ── THE FIX: Reversed Mako Colormap ──
    # mako_r pushes 0.0 to light mint/white, and 1.0 to deep navy blue.
    cmap = sns.color_palette("mako_r", as_cmap=True)
        
    axes = []
    for r in range(rows):
        for c in range(cols):
            axes.append(fig.add_subplot(gs[r, c]))
            
    for idx, cl in enumerate(cell_lines):
        ax = axes[idx]
        model = all_chitin[cl].get("model")
        
        if model is None or model.sweep_results is None or model.sweep_results.empty:
            ax.text(0.5, 0.5, "Auto-Calibration Disabled/Failed", ha='center', va='center', fontsize=12, color=_THEME["text"])
            ax.set_title(cl, fontweight='bold', color=_THEME["text"])
            continue
            
        df_sweep = model.sweep_results.copy()
        
        pareto_keys = []
        if model.pareto_front is not None:
            pareto_keys = list(zip(model.pareto_front['mode'], model.pareto_front['k'], model.pareto_front['n_pcs'], model.pareto_front['metric']))
        
        df_sweep['is_pareto'] = df_sweep.apply(
            lambda r: (r.get('mode','knn'), r['k'], r['n_pcs'], r['metric']) in pareto_keys, axis=1
        )
        
        x_min, x_max = df_sweep['rank_disruption'].min(), df_sweep['rank_disruption'].max()
        y_min, y_max = df_sweep['disc_ratio'].min(), df_sweep['disc_ratio'].max()
        norm_x = (df_sweep['rank_disruption'] - x_min) / (x_max - x_min + 1e-9)
        norm_y = (df_sweep['disc_ratio'] - y_min) / (y_max - y_min + 1e-9)
        
        pareto_mask = df_sweep['is_pareto']
        if pareto_mask.sum() > 0:
            px, py = norm_x[pareto_mask].values, norm_y[pareto_mask].values
            distances = [np.min(np.sqrt((px - ix)**2 + (py - iy)**2)) for ix, iy in zip(norm_x, norm_y)]
            df_sweep['dist_to_pareto'] = distances
        else:
            df_sweep['dist_to_pareto'] = 1.0
        
        df_sweep['marker_size'] = 20 + 160 * np.exp(-4 * df_sweep['dist_to_pareto'])
        df_sweep['norm_stability'] = (rankdata(df_sweep['signal_stability']) - 1) / (len(df_sweep) - 1)
        
        sel = model.selected_params if model.selected_params else {}
        sel_mode, sel_k, sel_pcs, sel_metric = sel.get('mode', ''), sel.get('k', -1), sel.get('n_pcs', -1), sel.get('metric', '')
        
        df_sweep['is_winner'] = (df_sweep['mode'] == sel_mode) & (df_sweep['k'] == sel_k) & (df_sweep['n_pcs'] == sel_pcs) & (df_sweep['metric'] == sel_metric)
        
        for metric, marker in [('cosine', 'o'), ('euclidean', 's')]:
            sub_df = df_sweep[df_sweep['metric'] == metric]
            if sub_df.empty: continue
            
            nw_df = sub_df[~sub_df['is_winner']]
            if not nw_df.empty:
                colors = cmap(nw_df['norm_stability']) 
                alphas = nw_df['is_pareto'].map({True: 0.95, False: 0.6})
                ax.scatter(nw_df['rank_disruption'], nw_df['disc_ratio'], 
                           c=colors, s=nw_df['marker_size'], marker=marker, 
                           alpha=alphas, edgecolor=_THEME["bg"], linewidth=0.5)
            
            w_df = sub_df[sub_df['is_winner']]
            if not w_df.empty:
                ax.scatter(w_df['rank_disruption'], w_df['disc_ratio'], 
                           color=_THEME["accent"], s=300, marker=marker, 
                           edgecolor=_THEME["text"], linewidth=1.5, zorder=5)

        ax.set_title(cl, fontweight='bold', fontsize=14, color=_THEME["text"])
        if idx % cols == 0: ax.set_ylabel("Discrimination Ratio\n(Post/Pre)", fontweight="bold", color=_THEME["text"])
        if idx >= n_lines - cols: ax.set_xlabel("Rank Disruption\n(1 - Spearman Rho)", fontweight="bold", color=_THEME["text"])
            
    for idx in range(n_lines, len(axes)): axes[idx].set_visible(False)
    
    # ── COLORBAR ──
    cbar_ax = fig.add_subplot(gs[:rows, -1])
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Relative Signal Stability (Percentile Rank)', fontweight='bold', rotation=270, labelpad=20, color=_THEME["text"])
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'], fontweight='bold', color=_THEME["text"])
    
    # ── LEGEND ──
    leg_ax = fig.add_subplot(gs[-1, :cols])
    leg_ax.axis('off')
    
    winner_circ = Line2D([0], [0], marker='o', color='w', markerfacecolor=_THEME["accent"], markeredgecolor=_THEME["text"], markersize=12, markeredgewidth=1.5)
    winner_sq   = Line2D([0], [0], marker='s', color='w', markerfacecolor=_THEME["accent"], markeredgecolor=_THEME["text"], markersize=12, markeredgewidth=1.5)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_THEME["mid"], markersize=10),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=_THEME["mid"], markersize=10),
        (winner_circ, winner_sq),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_THEME["dark"], markersize=12),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_THEME["grid"], markersize=5)
    ]
    labels = ['Cosine Metric', 'Euclidean Metric', 'Selected Optimum', 'Pareto Optimal (Large)', 'Sub-Optimal (Small)']
    
    leg_ax.legend(handles=legend_elements, labels=labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)}, 
                  loc='center', ncol=5, frameon=True, facecolor=_THEME["bg"], edgecolor=_THEME["grid"], 
                  fontsize=11, labelcolor=_THEME["text"])
               
    fig.suptitle("Phase 13: CHITIN Hyperparameter Calibration", fontsize=18, fontweight="bold", y=0.98, color=_THEME["text"])
    
    try:
        fig.tight_layout()
    except Exception:
        pass
        
    try:
        path = _save(fig, cfg, "p13_chitin_pareto")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

def plot_chitin_bivariate_grid(all_chitin_dict, cfg):
    """
    Generates a dynamic composite visualization with Marginals.
    Layers:
      1. Pre-CHITIN Shape: Filled KDE, Theme 'high', 60% opacity, matched smoothness.
      2. Post-CHITIN Heatmap: 'mako' colormap.
      3. Post-CHITIN Contours: Sharp theme background lines to pop against the density.
      4. Marginals: X/Y distributions for Pre (High) and Post (Dark/Low).
    """
    _apply_theme()
    import math
    import scanpy as sc
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import TruncatedSVD
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cell_lines = list(all_chitin_dict.keys())
    n_lines = len(cell_lines)
    if n_lines == 0: return None, None

    cols = min(3, n_lines)
    rows = math.ceil(n_lines / cols)
    
    # ── THE ULTIMATE GRIDSPEC FIX (With room for marginals) ──
    fig = plt.figure(figsize=(8 * cols, 7 * rows + 1.5))
    
    # Outer GridSpec: Main plots + 1 dedicated row at the bottom for the Legend
    gs_outer = fig.add_gridspec(rows + 1, cols, 
                                height_ratios=[1]*rows + [0.1], 
                                hspace=0.35, wspace=0.25)

    for idx, cl in enumerate(cell_lines):
        r, c = divmod(idx, cols)

        # ── INNER GRIDSPEC: Carving out the Marginals ──
        # Creates a mini 2x2 grid inside this specific cell line's box
        gs_inner = gs_outer[r, c].subgridspec(2, 2, 
                                              width_ratios=[5, 1], 
                                              height_ratios=[1, 5], 
                                              wspace=0.05, hspace=0.05)
        
        ax_main = fig.add_subplot(gs_inner[1, 0])
        ax_marg_x = fig.add_subplot(gs_inner[0, 0], sharex=ax_main)
        ax_marg_y = fig.add_subplot(gs_inner[1, 1], sharey=ax_main)

        # 1. Robust memory extraction
        res = all_chitin_dict[cl]
        if isinstance(res, dict):
            if 'train' in res: adata = res['train']
            elif 'Standalone' in res: adata = res['Standalone']
            else: adata = next((v for v in res.values() if hasattr(v, 'X')), None)
        else:
            adata = res

        if adata is None: continue

        import scipy.sparse as sp

        # 2. Extract Matrices (KEEP SPARSE!)
        if 'pre_chitin' not in adata.layers:
            X_pre = adata.X
        else:
            X_pre = adata.layers['pre_chitin']

        X_post = adata.X

        # 3. Fit SVD on POST-CHITIN (New Biological Space)
        # TruncatedSVD natively eats sparse matrices, keeping RAM usage microscopic
        svd = TruncatedSVD(n_components=2, random_state=42)
        pca_post = svd.fit_transform(X_post)

        # 4. Reverse Projection (Math trick to avoid Dense Broadcasting)
        if sp.issparse(X_pre):
            # (X - mean) * V  is mathematically equal to  (X * V) - (mean * V)
            pca_pre_raw = svd.transform(X_pre)
            mean_vec = np.asarray(X_pre.mean(axis=0))
            mean_proj = mean_vec @ svd.components_.T
            pca_pre = pca_pre_raw - mean_proj
        else:
            X_pre_centered = X_pre - X_pre.mean(axis=0)
            pca_pre = svd.transform(X_pre_centered)

        # Microscopic jitter for KDE safety
        visual_radius = np.max(np.abs(pca_pre)) * 0.015
        pca_post_safe = pca_post + np.random.normal(0, visual_radius, pca_post.shape)

        # ════════════════════════════════════════════════════════════════════
        # MAIN PLOT
        # ════════════════════════════════════════════════════════════════════
        
        # LAYER 1: Pre-CHITIN Topographic Shape 
        # (Matched bw_adjust=0.5, Opacity=0.6, Theme='high')
        sns.kdeplot(x=pca_pre[:, 0], y=pca_pre[:, 1], ax=ax_main, 
                    fill=True, color=_THEME["high"], alpha=0.6, 
                    thresh=0.05, bw_adjust=0.5, linewidths=0, zorder=1, warn_singular=False)

        # LAYER 2: Post-CHITIN Pixel Heatmap (Mako Aesthetic)
        sns.histplot(x=pca_post_safe[:, 0], y=pca_post_safe[:, 1], ax=ax_main,
                     bins=25, pthresh=0.05, cmap='mako', alpha=0.9, zorder=2)

        # LAYER 3: Post-CHITIN Contours (Theme background to stay visible)
        sns.kdeplot(x=pca_post_safe[:, 0], y=pca_post_safe[:, 1], ax=ax_main,
                    color=_THEME["bg"], levels=6, linewidths=0.8, bw_adjust=0.5, zorder=3, warn_singular=False)

        # Subtle crosshairs mapping the origin
        ax_main.axhline(0, color=_THEME["grid"], linestyle=':', linewidth=1.5, alpha=0.8, zorder=0)
        ax_main.axvline(0, color=_THEME["grid"], linestyle=':', linewidth=1.5, alpha=0.8, zorder=0)

        # ════════════════════════════════════════════════════════════════════
        # MARGINALS
        # ════════════════════════════════════════════════════════════════════
        
        # Top Marginal (X-axis Density)
        sns.kdeplot(x=pca_post_safe[:, 0], ax=ax_marg_x, fill=True, color=_THEME["low"], alpha=0.8, linewidth=0, warn_singular=False)
        sns.kdeplot(x=pca_pre[:, 0], ax=ax_marg_x, fill=True, color=_THEME["high"], alpha=0.75, linewidth=0, warn_singular=False)
        
        # Right Marginal (Y-axis Density)
        sns.kdeplot(y=pca_post_safe[:, 1], ax=ax_marg_y, fill=True, color=_THEME["low"], alpha=0.8, linewidth=0, warn_singular=False)
        sns.kdeplot(y=pca_pre[:, 1], ax=ax_marg_y, fill=True, color=_THEME["high"], alpha=0.75, linewidth=0, warn_singular=False)

        # ════════════════════════════════════════════════════════════════════
        # FORMATTING
        # ════════════════════════════════════════════════════════════════════
        
        # Hide the axes on the marginals so they look clean
        ax_marg_x.axis('off')
        ax_marg_y.axis('off')

        # Put the title on the Top Marginal so it clears the new graphics
        ax_marg_x.set_title(f"{cl}", fontweight='bold', fontsize=15, color=_THEME["text"], pad=10)
        
        ax_main.set_xlabel("New PC1 (Biological Variance)", fontweight='bold', color=_THEME["text"])
        if c == 0:
            ax_main.set_ylabel("New PC2 (Biological Variance)", fontweight='bold', color=_THEME["text"])
        else:
            ax_main.set_ylabel("")

        ax_main.spines["top"].set_visible(False)
        ax_main.spines["right"].set_visible(False)
        
        # Ensure tick parameters map to the text color
        ax_main.tick_params(colors=_THEME["text"])

    # ── LEGEND: Isolated to the bottom row ──
    leg_ax = fig.add_subplot(gs_outer[-1, :])
    leg_ax.axis('off')

    # THE FIX: Swapped "accent" (which was red) for "dark" (Navy Blue) to match the Mako heatmap core
    custom_lines = [
        Patch(facecolor=_THEME["high"], alpha=0.6, edgecolor='none'),
        Patch(facecolor=_THEME["low"], edgecolor=_THEME["bg"], alpha=0.8, linewidth=1.5)
    ]
    
    # Anchored securely in the center of its dedicated row
    leg_ax.legend(custom_lines, 
               ['Pre-CHITIN Shape & Marginal (Baseline Spread)', 'Post-CHITIN Density & Marginal (Mako Heatmap)'],
               loc='center', ncol=2, frameon=True, facecolor=_THEME["bg"], edgecolor=_THEME["grid"], 
               fontsize=13, labelcolor=_THEME["text"])

    fig.suptitle("Dual Density & Mako Heatmap: Resolving the Biological Core", fontsize=20, fontweight="bold", y=0.98, color=_THEME["text"])

    # Safely execute layout adjustment, ignoring internal GridSpec warnings
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
    except Exception:
        pass

    try:
        path = _save(fig, cfg, "p13_chitin_bivariate")
        _show(fig)
    except NameError:
        path = None
        
    return fig, path

# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE WATERFALL
# ─────────────────────────────────────────────────────────────────────────────

def plot_pipeline_waterfall(pipeline_tracker, cfg):
    """
    Generates a two-panel pipeline attrition summary.
    Top Panel: Line plot with filled area showing Cell Retention.
    Bottom Panel: Bar chart showing Gene (Feature Space) Retention.
    """
    _apply_theme()
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Robust data extraction
    labels = []
    cells = []
    genes = []
    
    for k, v in pipeline_tracker.items():
        labels.append(k)
        if isinstance(v, dict):
            cells.append(v.get("cells", 0))
            genes.append(v.get("genes", 0))
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            cells.append(v[0])
            genes.append(v[1])
        elif isinstance(v, (int, float)):
            # Fallback if only cells were passed
            cells.append(v)
            genes.append(0)
        else:
            cells.append(0)
            genes.append(0)

    if len(labels) < 2:
        print("Not enough data in pipeline_tracker to plot waterfall.")
        return None, None

    # 2. Figure Setup: 2 Rows, 1 Column (Shared X-axis)
    fig, (ax_cells, ax_genes) = plt.subplots(2, 1, figsize=(max(12, len(labels) * 1.5), 9), sharex=True)
    x_positions = np.arange(len(labels))

    # ════════════════════════════════════════════════════════════════════════
    # TOP PANEL: Cell Retention (Line Plot)
    # ════════════════════════════════════════════════════════════════════════
    ax_cells.plot(x_positions, cells, color=_THEME["accent"], marker='o', 
                  markersize=8, linewidth=2.5, linestyle='-', zorder=3)
    ax_cells.fill_between(x_positions, cells, color=_THEME["accent"], alpha=0.15, zorder=2)
    
    ax_cells.set_title("Observation Attrition (Cell Count)", fontweight="bold", fontsize=15, color=_THEME["text"], pad=15)
    ax_cells.set_ylabel("Total Cells", fontweight="bold", color=_THEME["text"])
    ax_cells.spines["top"].set_visible(False)
    ax_cells.spines["right"].set_visible(False)
    
    max_cells = max(cells) if max(cells) > 0 else 1
    ax_cells.set_ylim(0, max_cells * 1.25)
    
    # Annotate Cell Counts & Percentages
    for x, y in zip(x_positions, cells):
        pct = (y / max_cells) * 100
        ax_cells.text(x, y + (max_cells * 0.05), f"{int(y):,}\n({pct:.1f}%)", 
                      ha="center", va="bottom", fontsize=10.5, color=_THEME["text"], fontweight="bold")

    # ════════════════════════════════════════════════════════════════════════
    # BOTTOM PANEL: Feature Space Retention (Bar Plot)
    # ════════════════════════════════════════════════════════════════════════
    ax_genes.bar(x_positions, genes, color=_THEME["mid"], edgecolor=_THEME["bg"], linewidth=1.5, width=0.6, zorder=3)
    
    ax_genes.set_title("Feature Space Attrition (Gene Count)", fontweight="bold", fontsize=15, color=_THEME["text"], pad=15)
    ax_genes.set_ylabel("Total Genes", fontweight="bold", color=_THEME["text"])
    ax_genes.spines["top"].set_visible(False)
    ax_genes.spines["right"].set_visible(False)
    
    max_genes = max(genes) if max(genes) > 0 else 1
    ax_genes.set_ylim(0, max_genes * 1.25)
    
    # Annotate Gene Counts
    for x, y in zip(x_positions, genes):
        if y > 0:
            pct = (y / max_genes) * 100
            ax_genes.text(x, y + (max_genes * 0.05), f"{int(y):,}\n({pct:.1f}%)", 
                          ha="center", va="bottom", fontsize=10.5, color=_THEME["text"])

    # ════════════════════════════════════════════════════════════════════════
    # FORMATTING & CLEANUP
    # ════════════════════════════════════════════════════════════════════════
    ax_genes.set_xticks(x_positions)
    ax_genes.set_xticklabels(labels, rotation=45, ha="right", fontsize=12, color=_THEME["text"])
    
    ax_cells.grid(True, linestyle=':', alpha=0.6, color=_THEME["grid"], zorder=0)
    ax_genes.grid(True, linestyle=':', alpha=0.6, color=_THEME["grid"], zorder=0)
    
    ax_cells.tick_params(colors=_THEME["text"])
    ax_genes.tick_params(colors=_THEME["text"])

    fig.suptitle("SPORE+ Global Pipeline Attrition Summary", fontsize=20, fontweight="bold", y=1.02, color=_THEME["text"])
    
    # Ensures the 45-degree labels don't get cut off at the bottom
    fig.tight_layout()

    try:
        path = _save(fig, cfg, "pipeline_attrition_waterfall")
        _show(fig)
    except NameError:
        path = None

    return fig, path


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION SUMMARY (text only — a plot would waste space)
# ─────────────────────────────────────────────────────────────────────────────

def plot_detection_summary(detection_results, cfg):
    if not detection_results:
        return
    print("\n" + "─" * 55)
    print("  Phase 1 · Detection Results")
    print("─" * 55)
    for label, key in [
        ("Modality",            "modality"),
        ("Gene ID format",      "gene_id_format"),
        ("IDs harmonized",      "gene_ids_harmonized"),
        ("Combinatorial",       "is_combinatorial"),
        ("Cell line col found", "detected_cell_line_col"),
    ]:
        val  = detection_results.get(key, "—")
        flag = "✓" if val not in [None, False, "—", "not found"] else " "
        print(f"  {flag}  {label:<25}  {val}")
    print("─" * 55 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
#  Mark 2 Additions
# ─────────────────────────────────────────────────────────────────────────────

def plot_v_direction_heatmap(v_val: dict, cfg: dict):
    """
    Heatmap of pairwise cosine similarity between cell lines' V vectors.
    Only meaningful when n_cell_lines > 1.
 
    v_val structure: {"pairwise_cos": {(cl1, cl2): cos_value, ...},
                      "merge_flags": [...]}
    """
    _apply_theme()
    pairwise = v_val.get("pairwise_cos", {})
    if not pairwise:
        return None, None
 
    import pandas as pd
    lines = sorted(set(k for pair in pairwise.keys() for k in pair))
    n     = len(lines)
    mat   = np.eye(n)
    for (a, b), val in pairwise.items():
        i = lines.index(a)
        j = lines.index(b)
        mat[i, j] = val
        mat[j, i] = val
 
    df_mat = pd.DataFrame(mat, index=lines, columns=lines)
 
    fig, ax = plt.subplots(figsize=(max(5, n * 0.9), max(4.5, n * 0.85)))
    sns.heatmap(df_mat, ax=ax, cmap="rocket_r", vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.5,
                linecolor=_THEME["spine"],
                cbar_kws={"label": "Cosine similarity of V vectors",
                           "shrink": 0.8})
    ax.set_title("A  Cell line systematic variation alignment\n"
                 "(values near 1 = shared systematic direction)",
                 loc="left", fontweight="bold")
    _label(ax, "A")
 
    merge_flags = v_val.get("merge_flags", [])
    if merge_flags:
        ax.text(0.98, 0.02,
                f"Merge flags: {', '.join(str(f) for f in merge_flags)}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=_THEME["warn"])
 
    fig.suptitle("Phase 11 · V-direction alignment across cell lines",
                 fontsize=13, fontweight="bold", y=1.02)
    path = _save(fig, cfg, "p11_v_direction_heatmap")
    _show(fig)
    return fig, path
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  Paste into plotting.py at the end of PHASE 13 / CHITIN section
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_chitin_summary(chitin_summary: dict, cfg: dict):
    """
    Wrapper: calls the three CHITIN diagnostic plots for every cell line
    in chitin_summary. Replaces the old useless rectangle plot.
 
    chitin_summary structure:
        {cell_line: {"rank_corrs": np.ndarray,
                     "dist_pre": np.ndarray,
                     "dist_post": np.ndarray,
                     "delta_df": pd.DataFrame,
                     "delta_norms": np.ndarray,
                     "chitin_model": ChitinModel}}
    """
    if not chitin_summary:
        print("  plot_chitin_summary: no CHITIN summary data to plot")
        return None, None
 
    paths = []
    for cell_line, cs in chitin_summary.items():
        rank_corrs  = cs.get("rank_corrs")
        dist_pre    = cs.get("dist_pre")
        dist_post   = cs.get("dist_post")
        delta_df    = cs.get("delta_df")
        delta_norms = cs.get("delta_norms")
        model       = cs.get("chitin_model")
 
        if rank_corrs is not None and len(rank_corrs) > 0:
            _, p = plot_chitin_rank_disruption(rank_corrs, cell_line, cfg)
            if p:
                paths.append(p)
 
        if dist_pre is not None and dist_post is not None:
            _, p = plot_chitin_discrimination(dist_pre, dist_post, cell_line, cfg)
            if p:
                paths.append(p)
 
        if delta_df is not None and delta_norms is not None:
            _, p = plot_chitin_delta_magnitudes(delta_df, delta_norms, cell_line, cfg)
            if p:
                paths.append(p)
 
        if model is not None and model.sweep_results is not None:
            _, p = plot_chitin_pareto(model, cell_line, cfg)
            if p:
                paths.append(p)
 
    return None, paths
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  M2 · Doublet detection QC plot
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_doublet_summary(adata, cfg: dict):
    """
    Two-panel:
    A  Histogram of doublet scores (all cells), threshold marked
    B  ECDF of doublet scores — highlights where predicted doublets fall
    """
    _apply_theme()
    if "doublet_score" not in adata.obs.columns:
        print("  No doublet_score in obs — run Phase 4 first")
        return None, None
 
    scores     = adata.obs["doublet_score"].values.astype(float)
    predicted  = adata.obs.get("predicted_doublet",
                                np.zeros(len(scores), dtype=bool)).values.astype(bool)
    n_doublets = int(predicted.sum())
 
    # Infer threshold from the minimum score among predicted doublets
    threshold = float(scores[predicted].min()) if n_doublets > 0 else 0.5
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
 
    # A: score histogram
    ax1.hist(scores[~predicted], bins=60, color=_THEME["ctrl"],
             edgecolor="white", lw=0.2, alpha=0.8,
             label=f"Singlets ({(~predicted).sum():,})")
    ax1.hist(scores[predicted], bins=60, color=_THEME["accent"],
             edgecolor="white", lw=0.2, alpha=0.8,
             label=f"Doublets ({n_doublets:,})")
    ax1.axvline(threshold, color=_THEME["neutral"], ls="--", lw=1.5,
                label=f"Threshold: {threshold:.3f}")
    ax1.set_xlabel("Doublet score")
    ax1.set_ylabel("Cells")
    ax1.legend(fontsize=9)
    ax1.set_title("A  Doublet score distribution", loc="left", fontweight="bold")
    _label(ax1, "A")
 
    # B: ECDF
    sorted_s = np.sort(scores)
    ecdf     = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax2.plot(sorted_s, ecdf, color=_THEME["ctrl"], lw=2)
    ax2.axvline(threshold, color=_THEME["accent"], ls="--", lw=1.5,
                label=f"Threshold ({n_doublets:,} doublets = "
                      f"{n_doublets/max(len(scores),1)*100:.1f}%)")
    ax2.set_xlabel("Doublet score")
    ax2.set_ylabel("Cumulative fraction")
    ax2.legend(fontsize=9)
    ax2.set_title("B  Doublet score ECDF", loc="left", fontweight="bold")
    _label(ax2, "B")
 
    fig.suptitle(f"Phase 4 · Doublet detection\n"
                 f"{n_doublets:,} / {len(scores):,} predicted doublets "
                 f"({n_doublets/max(len(scores),1)*100:.2f}%)",
                 fontsize=13, fontweight="bold", y=1.04)
    path = _save(fig, cfg, "p04_doublet_summary")
    _show(fig)
    return fig, path
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  M2 · Ambient RNA QC plot
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_ambient_summary(adata, cfg: dict):
    """
    Two-panel:
    A  Scatter: ambient score vs total_counts (reveals the ambient cloud)
    B  Histogram of ambient scores with flag threshold marked
    """
    _apply_theme()
    if "ambient_score" not in adata.obs.columns:
        print("  No ambient_score in obs — run Phase 3 first")
        return None, None
 
    scores   = adata.obs["ambient_score"].values.astype(float)
    flagged  = adata.obs.get("ambient_flagged",
                              np.zeros(len(scores), dtype=bool)).values.astype(bool)
    n_flag   = int(flagged.sum())
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
 
    # A: ambient vs UMI scatter
    if "total_counts" in adata.obs.columns:
        umi = adata.obs["total_counts"].values.astype(float)
        # Subsample for speed
        rng    = np.random.default_rng(42)
        n_show = min(50_000, len(scores))
        idx    = rng.choice(len(scores), n_show, replace=False)
        ax1.scatter(umi[idx], scores[idx], c=flagged[idx].astype(int),
                    cmap="RdYlGn_r", s=3, alpha=0.5, rasterized=True)
        ax1.set_xscale("log")
        ax1.set_xlabel("Total UMI counts (log)")
        ax1.set_ylabel("Ambient score")
        ax1.set_title("A  Ambient score vs UMI — flagged cells in red",
                      loc="left", fontweight="bold")
        _label(ax1, "A")
    else:
        ax1.set_visible(False)
 
    # B: score histogram
    threshold = float(scores[flagged].min()) if n_flag > 0 else 0.8
    ax2.hist(scores[~flagged], bins=60, color=_THEME["ctrl"],
             edgecolor="white", lw=0.2, alpha=0.8,
             label=f"Clean ({(~flagged).sum():,})")
    if n_flag > 0:
        ax2.hist(scores[flagged], bins=40, color=_THEME["accent"],
                 edgecolor="white", lw=0.2, alpha=0.8,
                 label=f"Flagged ({n_flag:,})")
    ax2.axvline(threshold, color=_THEME["neutral"], ls="--", lw=1.5,
                label=f"Threshold: {threshold:.2f}")
    ax2.set_xlabel("Ambient score")
    ax2.set_ylabel("Cells")
    ax2.legend(fontsize=9)
    ax2.set_title("B  Ambient score distribution", loc="left", fontweight="bold")
    _label(ax2, "B")
 
    fig.suptitle(f"Phase 3 · Ambient RNA scoring\n"
                 f"{n_flag:,} / {len(scores):,} cells flagged as potentially ambient",
                 fontsize=13, fontweight="bold", y=1.04)
    path = _save(fig, cfg, "p03_ambient_summary")
    _show(fig)
    return fig, path

def get_theme(cfg=None):
    """Exposes the internal theme dictionary for custom notebook plots."""
    theme = _THEME.copy()
    # Map legacy keys to current SPORE+ aesthetic to prevent KeyErrors
    theme["panel"] = theme["bg"]  
    theme["text"]  = theme["neutral"]
    return theme

def plot_knockdown_efficiency(adata, cfg):
    """Plots the distribution of knockdown efficiencies."""
    _apply_theme()
    if "knockdown_efficiency" not in adata.uns:
        return None, None
        
    import pandas as pd
    eff_df = pd.DataFrame.from_dict(adata.uns["knockdown_efficiency"], orient="index")
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(eff_df["efficiency_score"].dropna(), bins=50,
            color=_THEME["accent"], edgecolor="white", lw=0.3, alpha=0.85)
    
    ax.axvline(0.50, color=_THEME["neutral"], linestyle="--", lw=1.5, label="50% threshold")
    
    ax.legend(fontsize=9)
    ax.set_xlabel("Fraction of cells passing escaper filter")
    ax.set_ylabel("Number of perturbations")
    ax.set_title("A  Knockdown Efficiency Distribution", loc="left", fontweight="bold")
    _label(ax, "A")
    
    fig.suptitle("Phase 5 · Knockdown Efficiency", fontsize=13, fontweight="bold", y=1.02)
    
    path = _save(fig, cfg, "p05_knockdown_efficiency")
    _show(fig)
    return fig, path
