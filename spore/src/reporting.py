"""
SPORE+ · src/reporting.py
───────────────────────────
Comprehensive end-of-run summary report for the full SPORE+ pipeline.

Covers:
  - Dataset metadata and detection results (Phase 1)
  - Cell counts through each phase (waterfall)
  - Cell line detection results (Phase 11)
  - Metacell quality metrics (Phase 12)
  - CHITIN calibration summary per cell line (Phase 13)
  - Output file manifest
  - Warnings and flags raised throughout the pipeline
  - Overall assessment

Saved as:  output/reports/{dataset_name}_SPORE_PLUS_report.md
"""

import numpy as np
from datetime import datetime
from pathlib import Path


def generate_report(
    cfg:             dict,
    pipeline_tracker: dict,        # phase → cell count waterfall
    detection_results: dict,       # from Phase 1
    cell_line_meta:   dict,        # from Phase 11
    chitin_summary:   dict,        # from Phase 13 (per cell line)
    all_meta_splits:  dict,        # from Phase 12 {cell_line: {split: adata}}
    warnings_log:     list,        # list of warning strings collected
) -> Path:
    """
    Generate the SPORE+ comprehensive Markdown report.
    """
    dataset   = cfg.get("dataset", {}).get("name", "UNKNOWN")
    pert_col  = cfg.get("dataset", {}).get("perturbation_col", "gene")
    ctrl_lbl  = cfg.get("dataset", {}).get("control_label", "non-targeting")
    organism  = cfg.get("dataset", {}).get("organism", "human")
    pert_type = cfg.get("dataset", {}).get("perturbation_type", "CRISPRi")
    ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    W = 72
    lines = []

    def h1(title=""):
        lines.append("=" * W)
        if title:
            lines.append(f"  {title}")
            lines.append("=" * W)

    def h2(title):
        lines.append(f"\n## {title}")
        lines.append("-" * 40)

    def row(label, value, flag=""):
        lines.append(f"  {label:<35}  {str(value):<25} {flag}")

    def blank():
        lines.append("")

    # ── Header ──────────────────────────────────────────────────────────────
    lines.append("# SPORE+ Pipeline Report")
    lines.append("")
    h1(f"SPORE+ RUN REPORT — {dataset}")
    row("Generated",       ts)
    row("Dataset",         dataset)
    row("Organism",        organism)
    row("Perturbation type", pert_type)
    row("Perturbation col",  pert_col)
    row("Control label",     ctrl_lbl)
    blank()

    # ── Phase 1: Detection ───────────────────────────────────────────────────
    h2("1 · Detection Results (Phase 1)")
    if detection_results:
        det = detection_results
        row("Modality",           det.get("modality", "unknown"))
        row("Gene ID format",     det.get("gene_id_format", "unknown"))
        row("IDs harmonized",     det.get("gene_ids_harmonized", False))
        pert_struct = cfg.get("dataset", {}).get("perturbation_structure", "single")
        row("Perturbation structure", pert_struct)
        if pert_struct == "combinatorial":
            sep = cfg.get("dataset", {}).get("perturbation_separator", "+")
            n_cg = len(det.get("combinatorial_constituents", set()))
            row("Comb. separator",   sep)
            row("Constituent genes", n_cg)
        cl_col = det.get("detected_cell_line_col", None)
        n_cl   = det.get("n_cell_lines_labeled", 0)
        row("Cell line col (Phase 1)", cl_col or "not found",
            f"({n_cl} lines)" if n_cl else "")
        if det.get("adt_parked"):
            row("ADT features", "parked to obsm['X_adt']", "CITE-seq")
        if det.get("atac_parked"):
            row("ATAC peaks", "parked to obsm['X_atac_peaks']", "Multiome")
    else:
        lines.append("  Detection skipped (disabled in config)")
    blank()

    # ── Cell count waterfall ─────────────────────────────────────────────────
    h2("2 · Cell Count Waterfall")
    if pipeline_tracker:
        start = list(pipeline_tracker.values())[0]
        lines.append(f"  {'Phase':<30}  {'Cells':>12}  {'% retained':>10}")
        lines.append(f"  {'─'*30}  {'─'*12}  {'─'*10}")
        for phase_name, count in pipeline_tracker.items():
            if isinstance(count, int) and start > 0:
                pct = count / start * 100
                lines.append(
                    f"  {phase_name:<30}  {count:>12,}  {pct:>9.1f}%")
            else:
                lines.append(f"  {phase_name:<30}  {str(count):>12}")
    blank()

    # ── Cell line detection ──────────────────────────────────────────────────
    h2("3 · Cell Line Detection (Phase 11)")
    if cell_line_meta and not cell_line_meta.get("skipped"):
        tier   = cell_line_meta.get("tier_used", "?")
        n_cl   = cell_line_meta.get("n_cell_lines", 1)
        cls    = cell_line_meta.get("cell_lines", [])
        row("Tier used",        tier)
        row("Cell lines found", n_cl)
        if cls:
            for cl in cls:
                row(f"  {cl}", "detected")
        conf = cell_line_meta.get("confidence")
        if conf is not None:
            flag = "✓ STABLE" if conf >= 0.85 else "⚠ UNSTABLE"
            row("Bootstrap ARI", f"{conf:.3f}", flag)
        v_val = cell_line_meta.get("v_validation", {})
        if v_val:
            n_merge = v_val.get("n_merge_flagged", 0)
            if n_merge > 0:
                row("V-direction flags", n_merge,
                    "⚠ some pairs may be same cell line in different states")
            else:
                row("V-direction check", "PASS",
                    "all pairs show divergent regulatory programs")
        excl = cell_line_meta.get("excluded_small_lines", [])
        if excl:
            row("Excluded (too small)", str(excl), "⚠")
    else:
        lines.append("  Single cell line (no multi-line detection needed)")
    blank()

    # ── Metacell output summary ──────────────────────────────────────────────
    h2("4 · Metacell Output Summary (Phase 12)")
    if all_meta_splits:
        for cell_line, splits in all_meta_splits.items():
            lines.append(f"\n  Cell Line: **{cell_line}**")
            lines.append(
                f"  {'Split':<10}  {'Metacells':>12}  {'Genes':>8}")
            lines.append(f"  {'─'*10}  {'─'*12}  {'─'*8}")
            total = 0
            for sk in ["train", "val", "test"]:
                adata_m = splits.get(sk)
                if adata_m is None:
                    continue
                n_mc = adata_m.n_obs
                n_g  = adata_m.n_vars
                total += n_mc
                lines.append(f"  {sk:<10}  {n_mc:>12,}  {n_g:>8,}")
            lines.append(f"  {'TOTAL':<10}  {total:>12,}")

            # Quality metric summary
            train_adata = splits.get("train")
            if train_adata is not None and "metacell_quality" in train_adata.uns:
                qdf_records = train_adata.uns["metacell_quality"]
                if qdf_records:
                    import pandas as pd
                    qdf     = pd.DataFrame(qdf_records)
                    n_flag  = qdf.get("high_variance_warning", pd.Series()).sum()
                    mean_iv = qdf["inner_variance"].mean()
                    lines.append(
                        f"\n  Inner variance (train): mean={mean_iv:.4f}, "
                        f"{n_flag} groups flagged")
    blank()

    # ── CHITIN summary ───────────────────────────────────────────────────────
    h2("5 · CHITIN Calibration Summary (Phase 13)")
    if chitin_summary:
        for cell_line, entry in chitin_summary.items():
            lines.append(f"\n  Cell Line: **{cell_line}**")
            row("  Correction mode",  entry.get("correction_mode", "?"))
            row("  Auto-calibrated",  entry.get("auto_calibrated", False))
            row("  Selected k",       entry.get("k", "?"))
            row("  Selected n_pcs",   entry.get("n_pcs", "?"))
            row("  Selected metric",  entry.get("metric", "?"))
            if "rank_disruption" in entry:
                rd = entry["rank_disruption"]
                dr = entry.get("disc_ratio", 0)
                ss = entry.get("signal_stability", 0)
                verdict = (
                    "STRONG" if rd > 0.15
                    else "MODERATE" if rd > 0.08
                    else "WEAK")
                row("  Rank disruption",  f"{rd:.4f}", f"→ {verdict}")
                row("  Disc ratio",       f"{dr:.4f}")
                row("  Signal stability", f"{ss:.4f}")
            sys_pcs = entry.get("systematic_pcs", [])
            if sys_pcs:
                row("  Systematic PCs out",
                    str([f"PC{i+1}" for i in sys_pcs]))
    else:
        lines.append("  CHITIN correction was skipped or disabled")
    blank()

    # ── Output file manifest ─────────────────────────────────────────────────
    h2("6 · Output File Manifest")
    out_dir     = cfg["paths"]["_processed"]
    suffix_mc   = cfg.get("phase12_metacell", {}).get("suffix", "_metacell")
    lines.append("")
    lines.append("  **Metacell outputs (Phase 12):**")
    if all_meta_splits:
        for cell_line, splits_d in all_meta_splits.items():
            cl_tag = f"_{cell_line}" if cell_line != "single" else ""
            for sk in ["train", "val", "test"]:
                if sk in splits_d:
                    fname = f"{dataset}{cl_tag}_{sk}{suffix_mc}.h5ad"
                    lines.append(f"    → {out_dir / fname}")
    blank()
    # CHITIN (Phase 13) removed — no CHITIN output manifest.

    # ── Warnings collected throughout the run ───────────────────────────────
    h2("7 · Warnings & Flags")
    if warnings_log:
        for w in warnings_log:
            lines.append(f"  ⚠ {w}")
    else:
        lines.append("  ✓ No warnings raised")
    blank()

    # ── Overall assessment ───────────────────────────────────────────────────
    h2("8 · Overall Assessment")
    n_warnings = len(warnings_log)
    n_cl       = cell_line_meta.get("n_cell_lines", 1) if cell_line_meta else 1

    if n_warnings == 0:
        verdict = "CLEAN RUN — no warnings raised"
        icon    = "✓"
    elif n_warnings <= 3:
        verdict = "MINOR WARNINGS — review flags in Section 7"
        icon    = "~"
    else:
        verdict = f"MULTIPLE WARNINGS ({n_warnings}) — careful review required"
        icon    = "⚠"

    lines.append(f"\n  {icon} {verdict}")
    lines.append(f"\n  Pipeline summary:")
    row("  Cell lines processed", n_cl)
    row("  Data modality",
        detection_results.get("modality", "unknown") if detection_results else "unknown")
    row("  Gene IDs harmonized",
        detection_results.get("gene_ids_harmonized", False) if detection_results else False)
    pert_struct = cfg.get("dataset", {}).get("perturbation_structure", "single")
    row("  Perturbation structure", pert_struct)
    row("  CHITIN correction",
        "applied" if chitin_summary else "skipped")
    blank()

    # ── Downstream readiness ─────────────────────────────────────────────────
    h2("9 · Downstream Readiness")
    grn = cfg.get("downstream", {}).get("grn", "PSGRN")
    lines.append(f"\n  Target downstream: **{grn}**")
    lines.append("")
    lines.append("  Next steps:")
    lines.append("    1. Run GuanLab PSGRN on each CHITIN-corrected .h5ad")
    lines.append("    2. Pass raw GRN parquets through FUNGI for pruning")
    lines.append("    3. Feed pruned GRNs into SPECTRA for prediction")
    if n_cl > 1:
        lines.append(
            f"    4. (Optional) Run cross-cell-line edge consensus merge "
            f"post-FUNGI for a unified {n_cl}-cell-line master graph")
    blank()
    h1()

    # ── Write to disk ────────────────────────────────────────────────────────
    report_text = "\n".join(lines)

    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{dataset}_SPORE_PLUS_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  📄 Report saved → {report_path}")
    return report_path
