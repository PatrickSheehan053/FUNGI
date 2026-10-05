# obj_004_systema_graph_eval — Systematic-Variation-Aware GRN Evaluation on the Dense/Pruned Graph

**Planning doc.** Build directory: `OBJECTS/obj_004_systema_graph_eval/`. Parallel to obj_003 (general causal/biological evaluator); obj_004 is the *targeted systematic-variation* evaluator. Designed 2026-06-29.

---

## Purpose and Scope

obj_004 quantifies how much of a GRN's apparent causal signal is **perturbation-specific** versus **systematic-variation contamination**, directly on the dense graph and any post-FUNGI graph — **without requiring a trained predictor / GNN**. obj_003's `stat_prec` compares each edge's target distribution in regulator-knockdown cells against *control* cells, which (per SYSTEMA, Viñas Torné et al. 2026) credits an edge for any significant shift, including the shared convergent response (e.g., cell-cycle arrest) that occurs under essentially every essential-gene knockdown. obj_004 applies SYSTEMA's core correction at the edge level: it removes the global perturbed centroid (the shared/systematic component) and measures whether each edge's target shows a *regulator-specific* response. It serves the SHROOM (dense substrate) and FUNGI (post-prune) stages, and is the metric that can actually reveal whether a systematic-variation correction (CHITIN, exp_007) helped. **In scope:** edge-level specificity precision, systematic-variation cosine, the stat_prec→spec_prec contamination gap, a K-sweep, multi-cell-line YAML config, batch mode, leakage-safe held-out evaluation. **Out of scope:** anything requiring predicted expression for unseen perturbations (SYSTEMA's prediction-generalization axis stays a GNN-era tool); modifying obj_003 or any production code.

---

## Interface Design

**Entry point:** both a CLI and a `SystemaGraphEvaluator` class, mirroring obj_003's `grn_eval_v2.py`.

```bash
python OBJECTS/obj_004_systema_graph_eval/src/systema_graph_eval.py \
  --grn-parquet <path/to/graph.parquet> \
  --cell-config OBJECTS/obj_004_systema_graph_eval/data/cell_configs/rpe1.yaml \
  --topk 1000,5000,10000,25000,50000,100000 \
  --candidate-id <NAME> \
  --output-json <path/to/results.json> \
  [--distance wasserstein|edistance] [--also-stat-prec] [--panel-source <h5ad>]
```

**Required args:** `--grn-parquet` (cols Regulator/Target/Importance|Weight — auto-detect, same as obj_003), `--cell-config`, `--output-json`. **Optional:** `--topk` (default `1000,5000`), `--distance` (default `wasserstein`; `edistance` = energy distance per scPerturb), `--also-stat-prec` (recompute obj_003 stat_prec inline so the contamination gap is single-call), `--candidate-id`, `--panel-source`.

**Config (YAML, parent/daughter pattern identical to obj_003):** reuse obj_003's `rpe1.yaml` schema verbatim — `cell_line, sc_input, pert_col, control_label, min_cells_per_perturbation, split{test_size,random_state,stratify_col}, stat_metrics{p_threshold,min_cells}` — plus one new block:
```yaml
systema:
  min_cells_per_perturbation: 25     # SYSTEMA-grade coverage for a stable per-pert centroid
  specificity_test: mannwhitney      # test of regulator-specific residual vs control noise
  p_threshold: 0.05
  cosine_on: full_vector             # full_vector (regulator's whole shift) | target_component
```
RPE1 config functional; `k562.yaml`/`h1_hesc.yaml` stubs (mirror obj_003).

**Output (JSON), mirroring obj_003's `by_k` schema** so downstream aggregation is uniform: top-level run metadata (candidate, panel_size, n_edges, cell_config, split, distance), then `by_k`: list of `{k, spec_prec, sysvar_cosine, sysvar_gap, stat_prec (if --also-stat-prec), n_scored, n_skipped_no_coverage, n_specific_significant, mean_specific_shift, mean_systematic_shift}`. Also writes a flat CSV alongside.

---

## Architecture

Four components in `src/`:

1. **`cell_data.py`** — load the cell-config h5ad, build the **same held-out test split** obj_003 uses (reuse obj_003's split logic exactly — import from the cloned `cell_config.py`/`stat_metrics.py` so the test cells are identical and the metric is leakage-safe). Precompute, on the test split: control centroid `mu_ctrl`, per-perturbation centroids `mu_R` (regulators with ≥`systema.min_cells`), and the **global perturbed centroid** `mu_pert` (mean over all perturbed test cells). Cache to `intermediate/` keyed by config hash (the obj_003 disk-cache pattern) so repeated candidate evals reuse it.
2. **`systema_metrics.py`** — the core. For each top-K edge R→T (R covered):
   - **systematic shift** `s_sys = mu_pert − mu_ctrl` (shared component; same for all edges).
   - **regulator shift** `s_R = mu_R − mu_ctrl`.
   - **specific shift** `s_spec = mu_R − mu_pert` (R-specific residual = regulator shift minus the shared component).
   - **spec_prec@K**: fraction of top-K edges whose target component of `s_spec` is statistically significant — test T's distribution in R-knock test cells vs the *global perturbed* test cells for T (`systema.specificity_test`, p<threshold). (vs obj_003 stat_prec which tests R-knock vs *control*.)
   - **sysvar_cosine@K**: per edge, cosine(`s_R`, `s_sys`) (SYSTEMA's definition: cosine of the regulator's specific shift with the average perturbation effect), aggregated as the top-K mean. High → systematic; low → specific. `cosine_on` selects full-vector (regulator-level, weighted by edge count in top-K) vs target-component.
   - **sysvar_gap@K** = stat_prec@K − spec_prec@K (the headline contamination fraction; requires `--also-stat-prec`).
   - Optional `--distance edistance`: replace the per-target Wasserstein/test with scPerturb energy distance between R-knock and global-perturbed cell sets (E-distance), a multivariate specificity measure.
3. **`graph_io.py`** — parquet loader + column auto-detect + top-K slicing (reuse obj_003's exact logic so K-slicing matches; clone it).
4. **`systema_graph_eval.py`** — CLI + `SystemaGraphEvaluator` orchestrator + `batch_eval`-style resume/upsert mode (mirror obj_003's batch_eval.py) for scoring many candidates (e.g., all exp_007 CHITIN arms) in one call.

Data flow: graph parquet + cell-config → cached test-split centroids → per-edge specific/systematic shifts → per-K aggregation → JSON/CSV. External runtime reads: only the cell-config h5ad (and optionally pertpy for E-distance). No network calls.

Cell-type isolation: all dataset-specific values live in the YAML (parent runner, daughter configs), identical to obj_003 — no cell-line logic hardcoded in `src/`.

---

## Implementation Plan

1. **Scaffold + clone obj_003 shared logic** (`src/`, `clone/`). Copy obj_003's `cell_config.py`, `stat_metrics.py`, and the parquet/top-K loader into `clone/` (obj_004 imports the split + loaders from there so test cells are bit-identical to obj_003; never edit obj_003 originals). Output: build dir populated. Prereq: none. ~30 min.
2. **`src/cell_data.py`** — test-split centroid precompute (`mu_ctrl`, `mu_R`, `mu_pert`) with disk cache. Output: cached centroid arrays. Prereq: 1. ~1 h.
3. **`src/systema_metrics.py`** — spec_prec, sysvar_cosine, sysvar_gap, optional E-distance. Output: per-K metric dict. Prereq: 2. ~2 h.
4. **`src/graph_io.py` + `src/systema_graph_eval.py`** — CLI/class/batch orchestrator + JSON/CSV writer matching obj_003 schema. Output: runnable evaluator. Prereq: 3. ~1.5 h.
5. **`data/cell_configs/`** — `rpe1.yaml` (functional, reuse obj_003 values + `systema` block), `k562.yaml`/`h1_hesc.yaml` stubs. Prereq: 1. ~15 min.
6. **Tests** (see Testing Plan) → `intermediate/` + `logs/`. Prereq: 4–5. ~1.5 h.
7. **Optional E-distance** via pertpy/scPerturb if `--distance edistance` is wanted; otherwise pure numpy/scipy. Prereq: 3.

---

## Data and Dependencies

**Python:** numpy, scipy (stats: mannwhitneyu, wasserstein_distance), pandas, anndata, scanpy, pyarrow. Optional: `pertpy` (E-distance / `pt.tl.Distance("edistance")`) only if `--distance edistance`.
**Reference data:** none new — reuses the cell-config h5ad (`DATA/SPORE_outputs/RPE1/splits/RPE1_5k_essential_train.h5ad`) and the obj_003 split. No gold-standard databases needed (this is a perturbation-data-intrinsic metric, not a database-overlap metric).
**Production/obj scripts:** clone obj_003's `cell_config.py`, `stat_metrics.py`, parquet loader (modification = importing/extending → copy to `clone/`). Calls no FUNGI/SHROOM production code.
**Cell-line data:** RPE1 train split (above); K562/H1 splits when those datasets are processed (stubs until then).

---

## Testing Plan

- **Known-answer / control reproduction:** run with `--also-stat-prec` on the exp_004d `control_qprior_mintra` champion; confirm the inline stat_prec reproduces obj_003's value (±0.005) — proves the cloned split/loader is faithful.
- **Synthetic positive:** a graph of edges R→T where T is a pure cell-cycle/arrest gene (high systematic, low specific) must yield **high stat_prec but low spec_prec** (large sysvar_gap). A graph of known specific regulatory edges must yield spec_prec ≈ stat_prec (small gap). These two synthetic cases validate the decontamination direction.
- **Negative control:** score-shuffled edges → spec_prec and stat_prec both collapse to base rate; sysvar_cosine → undefined/neutral.
- **Edge cases:** regulators with no test-split coverage (skipped, counted in `n_skipped_no_coverage`); targets absent from panel (skipped); K larger than n_edges (clip); all-zero weights (deterministic ordering).
- **Cross-tool consistency:** on 3–4 exp_006/exp_008 graphs, confirm sysvar_gap ranks candidates sensibly (md_gate, which is causally strongest, should not necessarily be the most *specific*).
- **Performance:** centroid precompute cached once per config; per-candidate eval target <10 min at K=100k (matches obj_003).

---

## Success Criteria (promotion gate)

1. Inline stat_prec reproduces obj_003 within ±0.005 on ≥2 shared graphs (faithful split).
2. Both synthetic cases behave correctly (high-systematic graph → large sysvar_gap; specific graph → ~zero gap).
3. Negative control collapses to base rate.
4. Runs the full K-sweep on a real champion in <10 min, leakage-safe (test-split only — verified no train cells enter the centroids).
5. Independent re-implementation of spec_prec on one graph matches to ≤1e-3.

---

## Promotion Path

Once validated, obj_004 lives at `OBJECTS/obj_004_systema_graph_eval/` and is *called* (not merged) by experiments. Integration points: (a) **exp_007** — add an obj_004 call beside the obj_003 grn_eval_v2 step (Step 5/6) so every CHITIN arm reports stat_prec AND sysvar_gap/spec_prec; (b) **commands.md** — add an obj_004 invocation section; (c) future reports cite spec_prec/sysvar_gap as the targeted-systematic-variation readout alongside stat_prec. No production pipeline file changes; obj_004 is an evaluation tool, parallel to obj_001/obj_003. (Per Patrick: the executing agent builds obj_004 during exp_007 downtime/free GPU, then a follow-up prompt wires it into the running investigation.)

---

## Critical References

[1] Viñas Torné, R., Wiatrak, M., Piran, Z., Fan, S., Jiang, L., Teichmann, S.A., Nitzan, M., Brbić, M. "Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation." *Nature Biotechnology* 44:1050–1059, 2026. https://doi.org/10.1038/s41587-025-02777-8 — the systematic-variation definition (cosine of perturbation-specific shift vs average perturbation effect; centroid subtraction) obj_004 ports to the edge level. Uploaded: `s41587-025-02777-8.pdf`.
[2] Peidli, S., et al. "scPerturb: harmonized single-cell perturbation data." *Nature Methods* 21:531–540, 2024. https://doi.org/10.1038/s41592-023-02144-y — E-distance (energy distance) as a perturbation-effect metric; the optional `--distance edistance` multivariate specificity measure. NEW external resource.
[3] Heumos, L., et al. "Pertpy: an end-to-end framework for perturbation analysis." (scverse) https://pertpy.readthedocs.io — maintained library providing E-distance and Mixscape (local perturbation signature = subtract KNN-control average per cell, the per-cell analog of obj_004's edge-level decontamination); obj_004 can use its E-distance rather than reimplementing. NEW external resource.
[4] Chevalley, M., et al. "A large-scale benchmark for network inference from single-cell perturbation data." *Communications Biology* 8:412, 2025. https://doi.org/10.1038/s42003-025-07764-y — stat_prec/wass_test provenance (the metric obj_004 decontaminates).
[5] Replogle, J.M., et al. "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq." *Cell* 185:2559–2575, 2022. https://doi.org/10.1016/j.cell.2022.05.013 — RPE1 perturbation data underlying the centroids.
[6] Local code read (Phase 1; cloned, never edited): `OBJECTS/obj_003_grn_bio_evaluator_v2/src/{grn_eval_v2.py, stat_metrics.py, cell_config.py, batch_eval.py}` (stat_prec machinery, split logic, JSON schema, batch mode obj_004 mirrors); `OBJECTS/obj_003_grn_bio_evaluator_v2/data/cell_configs/rpe1.yaml` (config schema reused); `markdowns/objects/obj_003_grn_bio_evaluator_v2.md` (design conventions); `markdowns/reports/Report_002_CHITIN_Investigation.md` (the systematic_variation_cosine prototype + the "systematic variation is partly genuine causal biology" caveat obj_004 must surface, not hide).
