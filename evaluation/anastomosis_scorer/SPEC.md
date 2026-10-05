# obj_010 — ANASTOMOSIS — Unified Prediction-Scoring Gauntlet (design-object blueprint, 2026-07-10)

**This document is the build brief for an autonomous Claude Code agent.** Build the object start to
finish from this document without asking clarifying questions. Every path, format, and gate is stated.
Where a value must be read from a file, the file and the line to read are named.

**Standing project rules that bind this build (non-negotiable):**
- **Clone-only.** Never modify production code in place. Anything from `OBJECTS/obj_009*`, `OBJECTS/obj_003*`,
  `OBJECTS/obj_004`, `SHROOM/`, `FUNGI/`, `SPECTRA*/`, or `obj_010_anastomosis/anastomosis_proto/` that needs
  changing is first copied into `OBJECTS/obj_010_anastomosis/clone/`; edits happen only on the copy. If you only
  *call* or *import* a production file unchanged, import it directly (no copy).
- **Disable, don't delete.** No file deletion at all during this build. If something looks removable, leave it and
  note it. (Patrick's hard rule: pause and ask before any deletion.)
- **GPU-safe on the 2070 (8 GB).** Check free VRAM before any CUDA allocation; batch so you never spill to shared
  memory (a prior spill hung the machine). Default `--device cuda` with an exact CPU fallback.
- **All build artifacts stay inside `OBJECTS/obj_010_anastomosis/`.**

---

## Purpose and Scope

obj_010 (ANASTOMOSIS) is the project's single, canonical **prediction-scoring gauntlet**: given a model's
predicted perturbation response and the ground truth, it computes the full multi-metric panel (error,
correlation, differential-expression recovery, distribution, reference-insensitive, and systematic-variation /
specificity axes) and writes one tidy scored record per model×seed. It exists because the pipeline has been
adjudicating downstream ML quality on a **single** metric (RHIZO's plain RSC), which the previous analysis
proved is metric-fragile and co-expression-confounded — the ranking of arms literally flips between RSC and
DNSA, and pure co-expression (HYPHAE) wins plain RSC by construction. ANASTOMOSIS replaces single-metric
adjudication with a defensible orthogonal panel, and does it fast on GPU so the entire historical corpus of
RHIZO runs can be re-scored in one pass ("the fire sale").

**In scope (this version):**
1. A **dual-mode** scorer: Mode A ingests per-cell predicted/true expression (SPECTRA-style); Mode B ingests
   per-perturbation mean-LFC vectors (RHIZO-style). Both reduce to one canonical internal representation.
2. **GPU-accelerated from the start** — reuse the validated obj_003.2 GPU Mann-Whitney/Wasserstein kernel; add a
   GPU energy-distance and torch-vectorized mean-based tiers. Exact CPU fallback.
3. The full metric panel, organized against the **PerturBench** taxonomy for defensibility, plus the project's
   **Systema** specificity axis and a **co-expression-residualized correlation** (ports obj_009.3's `rsc_coexpr_resid`).
4. A **fire-sale ingestion** subsystem: auto-discover every historical RHIZO result across generations, adapt each
   generation's serialization to the canonical representation, score, and emit per-generation CSV/JSON plus a
   **master ledger** across all of them, with provenance and idempotent upsert.

**Out of scope (this version):** scoring the *graph* itself (that is obj_003/obj_004's job — obj_010 may only
attach graph-topology columns as annotation metadata, never re-derive them); training or running any model;
promotion into the live pipeline (a separate, explicit decision after the gates pass).

---

## Interface Design

### Entry points
- CLI: `python src/anastomosis.py score  --config <cfg.yaml> [--tag TAG | --all | --list | --legend] [--device cuda|cpu]`
- CLI: `python src/anastomosis.py firesale --config <firesale.yaml> [--device cuda|cpu] [--resume]`
- Python API: `from anastomosis import Anastomosis; Anastomosis(cfg).score_tag(tag) -> dict`

### Input modes (auto-detected per tag from the config `input_kind` field; never guessed silently)
- **Mode A — `per_cell`:** a directory of `pred_{seed}.h5ad` / `true_{seed}.h5ad` (cells × genes) + a control
  pool (train h5ad + test controls), exactly the proto layout. Runs the FULL gauntlet including the cell-only
  tiers (Wilcoxon-AUPRC, energy-distance/MMD).
- **Mode B — `mean_lfc`:** an `.npz` (or `.json`) carrying per-perturbation response vectors and the shared
  control mean, in the RHIZO contract:
  keys `{panel, N, signal_mask, mu_ctrl, <split>_names, <split>_LFC, <split>_pidx}` (this is exactly what
  `build_rpe1_lfc.py` writes and what the arbiter consumes). Runs the reduced gauntlet (all mean/rank/Systema
  tiers); the cell-only tiers are reported as `NaN` with an explicit `skipped_cell_tiers=true` flag — never
  faked by broadcasting a mean to pseudo-cells.

### Canonical internal representation (both modes collapse to this before any metric runs)
`ScoredUnit = { means_true[n_pert,n_gene], means_pred[n_pert,n_gene], mu_ctrl[n_gene], signal_mask[n_gene],
pert_gene_idx[n_pert], gene_names, split, (optional) X_cells_true, X_cells_pred, X_ctrl_pool }`.
For Mode B, `means_* = mu_ctrl + LFC_*` (verified exact: RHIZO LFC is `mean(pert)−mu_ctrl` in log1p-CP10K space
— `build_rpe1_lfc.py:5-8,42-60` — so subtracting `mu_ctrl` inside the tiers reconstructs the LFC bit-for-bit).

### Output
- Per tag: a row appended (upsert, keyed on `tag+seed+split+object_version`) to `output_dir/<results_csv>` and a
  companion `output_dir/json/<tag>_s{seed}.json` with the full metric dict + provenance.
- Fire sale: `firesale_out/<generation>/<tag>.csv` per generation, plus one **master ledger**
  `firesale_out/anastomosis_master_ledger.csv` (one row per generation×tag×seed×split) and a `.parquet` mirror.
- Provenance columns on every row: `object_version, generation, dataset, substrate, input_kind, device,
  n_perts_scored, n_genes, coverage_frac, seed, split, config_hash, source_path, timestamp`.

### Configuration (YAML; schema below, superset of the proto config)
```yaml
dataset:      {name, organism, cell_line, perturbation_col, control_label}
paths:        {train_h5ad, val_h5ad, test_h5ad, predictions_root, output_dir, results_csv}
scoring:
  device: cuda                 # cuda|cpu ; cpu fallback must be byte-identical on shared tiers
  data_format: lognorm         # lognorm|delta  (RHIZO/SPORE default = lognorm)
  lfc_space: lognorm_meandiff  # declared per dataset; guards the mode-B adapter scale
  deg_k_values: [10,20,50,100,200,500,1000]
  run_cell_tiers: auto         # auto = on for per_cell, off for mean_lfc
  run_pertpy_metrics: false    # pertpy/E-distance; obj_010 ships its own GPU e-dist so this is optional
  coexpr_resid: true           # port of obj_009.3 rsc_coexpr_resid — the specificity-beyond-coexpression axis
  n_jobs: 8
runs: [ {tag, input_kind, path, graph: null, n_seeds, generation, substrate, notes}, ... ]
```

---

## Architecture

Six modules in `src/`:

1. **`io_adapters.py`** — one adapter per input serialization. `load_per_cell(dir)` (Mode A, proto-compatible)
   and `load_mean_lfc(npz)` (Mode B, RHIZO contract). A registry `ADAPTERS: {input_kind -> loader}` returning a
   `ScoredUnit`. The fire-sale discovery hands each found artifact to the matching adapter.
2. **`gpu_kernels.py`** — the compute core. Imports the validated obj_003.2 kernel
   (`OBJECTS/obj_003_grn_bio_evaluator_v2/src/gpu_stat_kernels.py` — Mann-Whitney U + Wasserstein, scipy-exact,
   int64 rank arithmetic; **import, do not fork** unless a change is required, in which case clone it). Adds:
   torch-vectorized Pearson/Spearman/cosine/MAE/RMSE over the `[n_pert,n_gene]` matrices; a GPU **energy
   distance** (cross- minus within-group pairwise L2, cell mode) with a rapids-singlecell/pertpy-GPU parity
   check; batched top-k argsort for the DEG sweep. A `--device cpu` path that calls scipy/numpy and must match
   the GPU path bit-for-bit on the shared tiers (the obj_003.2 exactness doctrine — assert 0 flips in tests).
3. **`metrics.py`** — the metric panel, thin wrappers over `gpu_kernels`, grouped by the PerturBench taxonomy
   (see Data section). Each metric is a pure function `(ScoredUnit) -> {name: value}`; a metric that requires
   cells raises `CellTierUnavailable` in Mode B (caught, recorded as NaN + skip flag). Includes the
   `coexpr_resid` partial-correlation (build the train-only co-expression baseline `Z`, residualize, correlate —
   port `obj_009.3/src/metrics_v3f.py:30-90` and reuse verbatim where possible).
4. **`scorer.py`** — orchestrates one tag: adapter → ScoredUnit → run enabled tiers on `--device` → assemble the
   record → upsert CSV/JSON. Per-seed loop; per-perturbation batched on GPU. Crash-resilient: write each seed's
   JSON immediately; the CSV upsert is atomic (temp-file + replace).
5. **`firesale.py`** — the corpus runner. (a) **Discovery:** walk a configured set of roots
   (`OBJECTS/obj_009`, `obj_009.1`, `obj_009.2`, `obj_009.3`, `DATA/EXPERIMENTS/exp_025*`, and any
   `predictions_root`s) and classify each candidate artifact by a signature (npz with the RHIZO keys → mean_lfc;
   `pred_*.h5ad`/`true_*.h5ad` pair → per_cell; unknown → logged to `firesale_out/UNCLASSIFIED.txt`, never
   silently dropped). (b) **Inventory:** write `firesale_out/inventory.csv` (path, detected kind, generation,
   n_seeds, mtime) and STOP for a one-line human confirm on first run (`--resume` skips the stop). (c) **Score:**
   each classified artifact through `scorer`, tagged with its generation. (d) **Ledger:** concatenate all rows
   into the master ledger + parquet; add a `rank_within_metric` helper column set so the "what actually works"
   read is one `groupby`.
6. **`anastomosis.py`** — CLI dispatch (`score` / `firesale`) + config loader + `--legend` (prints the column
   dictionary) + `--list`.

Data flow: `config → discovery/adapter → ScoredUnit → gpu_kernels(device) → metrics panel → record → CSV/JSON
→ (firesale) master ledger`.

---

## Implementation Plan

**Step 0 — Scale & contract lock (read-only, 10 min).** Re-read `build_rpe1_lfc.py` (RHIZO Mode-B contract,
already confirmed: log1p-CP10K, `mean(pert)−mu_ctrl`) and `anastomosis_proto/anastomosis_scorer.py` Tiers 1–5
(the metric definitions to preserve). Write `intermediate/contract_notes.md` fixing the ScoredUnit schema and
the exact space of each metric. **Prereq for everything.**

**Step 1 — `io_adapters.py` + ScoredUnit (src/).** Implement both loaders; unit-test that Mode B on
`rpe1_run`'s LFC npz reconstructs `means−mu_ctrl == LFC` to 0. Artifact: a loaded ScoredUnit for one RHIZO arm.

**Step 2 — `gpu_kernels.py` (src/).** Import obj_003.2's kernel; add the torch mean-tier ops + GPU energy
distance. **Exactness gate:** on a fixed ScoredUnit, assert every shared metric is bit-identical `cuda` vs `cpu`
(Pearson/Spearman/MAE/RMSE/cosine to <1e-6; Mann-Whitney/Wasserstein exact via the obj_003.2 guarantee). VRAM:
batch perts so an RPE1 corpus (≤5000 genes × ≤500 perts) stays < 4 GB. Artifact: `intermediate/exactness.json`.

**Step 3 — `metrics.py` (src/).** Port the proto's 5 tiers verbatim onto `gpu_kernels`; add `coexpr_resid`
(port from `obj_009.3/src/metrics_v3f.py`). Tag each metric with its PerturBench category + a `needs_cells` bool.
Artifact: the panel dict on one arm, cross-checked against the proto's CPU numbers for the shared tiers.

**Step 4 — `scorer.py` (src/).** Wire adapter→panel→upsert. **Cross-validation gate:** Mode-B scoring of the
`rpe1_run` arms must reproduce the RSC already in `rpe1_run/results/rpe1_main.json` (the panel's Spearman/rank
axis == the arbiter's RSC within tolerance) — proves the port is faithful. Artifact: `intermediate/rpe1_rescored.csv`.

**Step 5 — `firesale.py` (src/).** Discovery + inventory + master ledger. Run discovery **dry** first
(`--list`), write `inventory.csv`, and STOP for confirm. Then score the classified corpus. Artifact:
`firesale_out/inventory.csv`, `.../anastomosis_master_ledger.csv`.

**Step 6 — `anastomosis.py` CLI + configs (src/, data/).** Ship `data/anastomosis_config_rpe1.yaml` (local RPE1
paths, mode-B) and `data/firesale.yaml` (the discovery roots). Repoint every path off the proto's
`/scratch/patrick.sheehan/...` HPC paths to local.

**Step 7 — Full test pass (Testing Plan below).** Then write `intermediate/BUILD_REPORT.md` and stop; promotion
is Patrick's call.

**Efficiency target:** the entire historical RHIZO corpus scored in **< 10 min** on the 2070 (Mode-B mean-tier
work is milliseconds/arm; cell tiers only where per-cell artifacts exist). Incremental saves so a crash resumes.

---

## Data and Dependencies

**Python:** numpy, scipy, pandas, scikit-learn, anndata, scanpy, pyyaml, **torch (CUDA)**; optional pertpy +
rapids-singlecell (only if `run_pertpy_metrics: true` — obj_010 ships its own GPU e-distance so this is a parity
check, not a requirement). Pin nothing beyond what the 2070 venv already has; if pertpy is absent, skip that one
parity check and log it (do not fail).

**Reused production code (import, clone only if edited):**
- `OBJECTS/obj_003_grn_bio_evaluator_v2/src/gpu_stat_kernels.py` — GPU Mann-Whitney + Wasserstein (the eval-stage
  primitive; the reason GPU scoring is an hour, not a day).
- `OBJECTS/obj_009.3_rhizo_final/src/metrics_v3f.py` — `build_coexpr_Z` + `rsc_coexpr_resid_per_pert` (the
  co-expression-residualized specificity axis).
- `OBJECTS/obj_010_anastomosis/anastomosis_proto/anastomosis_scorer.py` + `anastomosis_bio.py` — the 5-tier
  metric definitions to port (reference, not imported directly; copy needed functions into `src/metrics.py`).

**Reference inputs (local, already on disk — do not rebuild):**
- RHIZO Mode-B LFC + arms: `DATA/EXPERIMENTS/exp_025_hyphae_vs_shroom_ensemble/rpe1_run/` (`results/rpe1_main.json`,
  the LFC npz from `build_rpe1_lfc.py`, `arms/`).
- Historical RHIZO generations to discover: `OBJECTS/obj_009{,.1,.2,.3}_*`, `DATA/EXPERIMENTS/exp_025*`.

**Metric taxonomy (PerturBench-aligned — cite in the thesis):** error {MAE, RMSE, MSE, L2}; correlation {Pearson,
Spearman, cosine_delta, **coexpr_resid partial-corr**}; DEG recovery {F1@k sweep, precision/recall@k, f1_auc,
AUPRC_rank, AUROC, DEG t-score}; distribution {**energy distance**, MMD — cell mode only}; reference-insensitive
{RMSE@top20, Pearson@top20, Spearman@top20}; Systema {systematic_variation, pearson_systema (perturbed-centroid
reference), n_perts_above_perturbed_mean, centroid_accuracy}.

---

## Testing Plan

1. **Adapter exactness:** Mode-B reconstructs LFC to 0; Mode-A reproduces the proto's `means_true/means_pred`.
2. **GPU==CPU:** every shared metric bit-identical across `--device` (0 threshold-flips) — the hard gate.
3. **Faithful port:** obj_010's rank axis reproduces `rpe1_main.json` RSC within tolerance; Tier-1/4/5 numbers
   match a proto CPU run on one per-cell VCC tag (if a VCC per-cell artifact is reachable; else skip + log).
4. **Null behavior:** the shuffle and reverse arms collapse toward chance on every metric; the empty arm → 0
   (ψ(0)=0); a random-prediction control ≈ 0 correlation.
5. **Coverage honesty:** an arm with missing sources reports `coverage_frac < 1` and is scored only on covered
   perts (no silent credit).
6. **Fire-sale discovery:** inventory classifies every known artifact; nothing lands in `UNCLASSIFIED.txt`
   without a logged reason; master ledger row count == Σ(generation×tag×seed×split).
7. **VRAM:** peak stays < 4 GB on the RPE1 corpus; no shared-memory spill.

---

## Success Criteria (promotion gate)

- Dual-mode scorer runs Mode B (RHIZO) and Mode A (per-cell) end to end.
- GPU==CPU exactness gate passes with 0 flips; full historical RHIZO corpus scores in < 10 min on the 2070.
- obj_010 reproduces the existing RSC leaderboard (faithfulness) AND emits the full orthogonal panel + master
  ledger, so "what actually works" can be read across metrics and generations in one table.
- Nulls collapse; coverage is reported honestly; production code untouched (clone-only verified).

## Promotion Path

On pass, obj_010 becomes the canonical **downstream scoreboard**. Wire-in points (each a separate, explicit
step, not done by the build agent): (a) the RHIZO arbiter (`proto_rhizo_zeroshot.py` / obj_009.3 harness) calls
`Anastomosis` for reporting while keeping its lightweight RSC as the in-loop null gate; (b) the SPECTRA_f eval
(obj_005/obj_007) emits `pred_/true_` h5ads into obj_010's Mode-A layout so the same gauntlet scores real
per-cell runs. Live object dir stays `OBJECTS/obj_010_anastomosis/`; the proto stays archived at
`anastomosis_proto/` (disable-don't-delete).

## Critical References

- **PerturBench** — Wu, Y. et al. Benchmarking ML models for cellular perturbation analysis. *Nat. Methods*
  (2025) / arXiv:2408.10609. Metric taxonomy adopted for the panel. https://arxiv.org/html/2408.10609v4
- **scPerturBench** — bm2-lab. Single-cell perturbation effects prediction benchmark. https://github.com/bm2-lab/scPerturBench
- **pertpy / scPerturb E-distance** — Peidli, S. et al. Pertpy: an end-to-end framework for perturbation
  analysis. *Nat. Methods* (2025). Energy-distance definition + E-test. https://pertpy.readthedocs.io
- **rapids-singlecell GPU E-distance** (`pertpy-GPU`/ptg submodule) — GPU energy-distance reference for the
  cell-mode distribution tier. arXiv:2603.02402.
- Local code read Phase 1: `anastomosis_proto/anastomosis_scorer.py`, `.../anastomosis_bio.py`,
  `anastomosis_config.yaml`; `DATA/EXPERIMENTS/exp_025_hyphae_vs_shroom_ensemble/rpe1_run/build_rpe1_lfc.py`,
  `build_rpe1_arms.py`, `results/rpe1_main.json`, `results/arms_built.json`;
  `OBJECTS/obj_003_grn_bio_evaluator_v2/src/gpu_stat_kernels.py`; `OBJECTS/obj_009.3_rhizo_final/src/metrics_v3f.py`;
  `fungi_hyphae_prep/proto_exp025/src/proto_rhizo_zeroshot.py`.
