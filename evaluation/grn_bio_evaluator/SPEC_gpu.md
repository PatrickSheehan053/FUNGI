# obj_003.2 — GPU-Accelerated Statistical Evaluation Kernel (GPU port of obj_003 v2.1 + obj_004)

**Planning doc.** A **performance object**, not a new metric. Build/validate in `OBJECTS/obj_003.2_grn_eval_gpu/`. Named per Patrick's `obj_00N.M` versioning (this is the GPU sibling of obj_003.1 → **obj_003.2**, superseding the Claude Code agent's informal "obj_005_grn_eval_gpu" suggestion). Designed 2026-07-01.

---

## Purpose and Scope

The evaluation stage is now the pipeline's CPU bottleneck: SHROOM and FUNGI already use the GPU, but obj_003 v2.1 (`stat_prec`/`wass_test`) and obj_004 (`spec_prec`/`sysvar`) run scipy on the CPU — a per-regulator `groupby` loop calling `scipy.stats.mannwhitneyu(..., axis=0)` plus a per-target Python loop calling `scipy.stats.wasserstein_distance`. obj_003 is ~3 min/graph (knock vs ~1.6k control cells); obj_004 is ~12 min/graph (knock vs the ~28k global-perturbed centroid cells) — and both are re-run for every graph at 6 k-values. exp_014 (the MD-estimator bake-off) will hammer the evaluator far harder than exp_013 did, so a GPU kernel is worth building as its own validated object. **obj_003.2 delivers a single shared GPU statistical kernel** (`mannwhitneyu_gpu`, `wasserstein1d_gpu`) plus the wiring to call it from cloned copies of obj_003's `stat_metrics.py` and obj_004's `systema_metrics.py`, behind an opt-in `--device cuda` flag. **In scope:** GPU port of the two rank-based kernels (two-sided asymptotic Mann-Whitney U with continuity + tie correction; exact 1D Wasserstein), batched over targets and regulators; shared one-time precompute (fixed control-cell and global-perturbed-cell rank/sort structures reused across all graphs); a reproducibility validator proving the GPU output matches scipy; thermal safeguards (the WHEA 0x124 crashes in session 13 were triggered by sustained GPU load). **Out of scope:** the Tier-1 causal panel (already fast — pure set algebra, stays CPU), the Tier-2 motif validator (unchanged), and **any change to a metric's definition** — obj_003.2 must produce numerically identical verdicts, only faster. It serves the SHROOM/FUNGI/CHITIN evaluation stage that every experiment depends on.

## Interface Design

**Entry points (all in cloned evaluators — production obj_003/obj_004 untouched):**
```bash
# obj_003 v2.1 GPU path (clone)
python OBJECTS/obj_003.2_grn_eval_gpu/clone/obj_003_src/grn_eval_v2.py \
  --grn-parquet <g.parquet> --cell-config <rpe1.yaml> --topk 1000,5000,10000,25000,50000,100000 \
  --output-json <out.json> --device cuda [--gpu-batch-size 2048] [--cooldown-ms 250]

# obj_004 GPU path (clone)
python OBJECTS/obj_003.2_grn_eval_gpu/clone/obj_004_src/systema_graph_eval.py \
  --batch --candidates-json <m.json> --cell-config <rpe1.yaml> --topk ... --device cuda

# standalone reproducibility validator (the promotion gate)
python OBJECTS/obj_003.2_grn_eval_gpu/src/validate.py \
  --grn-parquet <g.parquet> --cell-config <rpe1.yaml> --topk 1000,5000,100000
```
- `--device {cpu,cuda}` — **default `cpu`** (safe/unchanged behavior); `cuda` opts into the GPU kernel. If `cuda` is requested but unavailable, log a clear warning and fall back to CPU (never silently error).
- `--gpu-batch-size N` (default 2048) — number of (regulator×target) columns processed per GPU batch; caps VRAM.
- `--cooldown-ms M` (default 0; recommend 250 for long sweeps) — sleep between GPU batches to bound sustained load (thermal guard).
- `--exact-fallback {on,off}` (default `on`) — route the small-n/no-tie columns scipy would compute *exactly* to CPU scipy (guarantees byte-exact match; see Architecture).
- **Config:** new optional `gpu` block in each cell_config YAML (`device`, `gpu_batch_size`, `cooldown_ms`, `dtype: float32`); absent block → CPU defaults, fully backward-compatible.
- **Output:** identical JSON/CSV schema to obj_003 v2.1 / obj_004 — same keys, same values (within the documented tolerance), plus a `_gpu_meta` block (device, dtype, n_columns_gpu, n_columns_exact_cpu, wall_time_s) for provenance.

## Architecture

**One shared kernel module, two thin integrations.**

1. **`src/gpu_stat_kernels.py`** — the reusable core, pure PyTorch (no differentiable/soft-sort dependency; native `torch.sort`/`torch.argsort`/`torch.searchsorted` are exact):
   - `mannwhitneyu_gpu(ref: Tensor[n_ref, C], intv: Tensor[n_int, C]) -> pvals: Tensor[C]` — replicates scipy's **asymptotic, two-sided, `use_continuity=True`** path. Per column: concatenate the two groups (n_ref+n_int rows), compute **average ranks** of the combined sample (ties → mean rank), sum the ranks of the `intv` group → U statistic; z = (U − n_ref·n_int/2 ± 0.5) / sqrt(σ²) with the **tie-corrected variance** σ² = (n_ref·n_int/12)·[(N+1) − Σ(tᵢ³−tᵢ)/(N(N−1))], N=n_ref+n_int, tᵢ = size of each tie group (computed per column from run-lengths of the sorted values); two-sided p = 2·Φ_sf(|z|) via `torch.special.ndtr`/`erfc`. Everything batched across C columns.
   - `wasserstein1d_gpu(a: Tensor[nA, C], b: Tensor[nB, C]) -> Tensor[C]` — exact 1D W₁: sort each column of a and b, build the merged support, integrate |CDF_a − CDF_b| over the deltas (the `torch.searchsorted` cumulative-CDF construction POT uses for its torch backend). Deterministic, no approximation.
   - Both accept a `chunk`/batch size and optional `cooldown_ms` to bound VRAM and sustained load.
2. **Reproducibility hybrid (`--exact-fallback on`).** scipy `mannwhitneyu(method="auto")` uses the **exact** null distribution when `min(n_ref, n_int) ≤ 8` **and** there are no ties, otherwise **asymptotic**. Single-cell expression is dense with zeros → ties almost everywhere → scipy picks asymptotic ~always, which the GPU kernel matches. The rare column that qualifies for exact (a regulator with few knock cells `min_cells`..8 **and** a tie-free target column) is detected on-device (min-n check + a per-column tie test) and routed to CPU `scipy.stats.mannwhitneyu(method="exact")`. This guarantees the GPU path reproduces scipy `auto` **exactly**, preserving the cross-experiment additivity guarantee (every prior experiment's regression was "diff = 0.0").
3. **Shared one-time precompute (`src/gpu_context.py`).** The control-cell test matrix (~1.6k×5k) and the global-perturbed matrix (~28k×5k) are **fixed per dataset**, independent of the graph. Load once, move to GPU float32 (1.6k×5k≈32 MB, 28k×5k≈560 MB — trivial on the 8 GB RTX 2070), and reuse across all graphs in a batch. Also share the single `stat_prec` computation between obj_003 and obj_004 (both compute it today — compute once, pass through). This is a large part of the win independent of the kernel itself.
4. **Two integrations (clones only).** Copy obj_003 v2.1 `src/*` → `clone/obj_003_src/` and obj_004 `src/*` → `clone/obj_004_src/`; in the cloned `stat_metrics.py` and `systema_metrics.py`, replace the two scipy calls with a dispatch on `--device` (cpu → original scipy; cuda → `gpu_stat_kernels`). The per-regulator grouping stays identical (only the inner statistical call changes), so the exact same edges are tested in the exact same order.

**Data flow:** graph parquet + cell_config → (unchanged) causal panel on CPU + (unchanged) edge grouping → GPU batched MWU + Wasserstein over the fixed on-GPU reference matrices → same JSON. Thermal guard: bursty batched work + optional cooldown + `--gpu-batch-size` cap; recommend cooldown for the long exp_014 sweep.

## Implementation Plan

1. **Scaffold + clone.** Copy obj_003 v2.1 `src/` → `clone/obj_003_src/`, obj_004 `src/` → `clone/obj_004_src/`, and the three `data/cell_configs/*.yaml` → `clone/cell_configs/`. Edit only clones. **~20 min. Prereq: none.**
2. **`src/gpu_stat_kernels.py`** — implement `mannwhitneyu_gpu` (asymptotic + continuity + tie-corrected variance) and `wasserstein1d_gpu` (exact). Unit-test each against scipy on random + tie-heavy + sparse-zero synthetic matrices before any integration. **~5 h. Prereq: 1.**
3. **`src/gpu_context.py`** — one-time GPU load/caching of the control + global-perturbed matrices and the shared `stat_prec`. **~2 h. Prereq: 2.**
4. **Integrate into `clone/obj_003_src/stat_metrics.py` + `grn_eval_v2.py`** — `--device`/`--gpu-batch-size`/`--cooldown-ms`/`--exact-fallback` flags; dispatch; `_gpu_meta`. **~3 h. Prereq: 2,3.**
5. **Integrate into `clone/obj_004_src/systema_metrics.py` + `systema_graph_eval.py`** — same dispatch (reuses the same kernel + the cached ~28k matrix). **~3 h. Prereq: 2,3.**
6. **`src/validate.py`** — run CPU and GPU paths on the same graphs/k, report per-metric max abs diff, `stat_prec` count-of-threshold-flips, `wass_test` rtol, and wall-time speedup. **~2 h. Prereq: 4,5.**
7. **Tests + benchmark** → `intermediate/` + `logs/`. **~3 h. Prereq: 6.**

## Data and Dependencies

**Python:** `torch` (CUDA build for the RTX 2070 — CUDA 11.8/12.x, already used by SHROOM/FUNGI), numpy, scipy (kept — the exact-fallback + the CPU reference path use it), pandas, pyarrow. **No `torchsort`** (that is *soft/differentiable* ranking; we need exact hard ranks — native torch suffices). Optional alt backend: CuPy/RAPIDS (not required; torch is already in the stack). POT (`python-ot`) is a *reference* for the torch 1D-Wasserstein construction, not a runtime dependency.
**Reference data:** none new — reuses the RPE1 test-split h5ad (control + perturbed cells) already loaded by obj_003/obj_004, and the same graphs.
**Production scripts:** cloned (modified) — obj_003 v2.1 `src/*` and obj_004 `src/*`. Calls nothing in SHROOM/FUNGI.
**Hardware:** NVIDIA RTX 2070 (8 GB). Memory budget fits both reference matrices + a 2048-column batch in float32 with wide margin.

## Testing Plan

- **Kernel unit tests (before integration):** `mannwhitneyu_gpu` vs `scipy.stats.mannwhitneyu(axis=0)` on (a) continuous no-tie data, (b) tie-heavy integer data, (c) sparse-zero matrices like real expression, (d) small n (3..8) — p-values match to rtol 1e-6 wherever scipy uses asymptotic; the exact-method columns are the ones routed to CPU. `wasserstein1d_gpu` vs `scipy.stats.wasserstein_distance` to rtol 1e-6 on the same inputs.
- **End-to-end reproducibility (the gate):** on ≥3 real graphs (`sc_control_w25`, `mbk_k5_w25`, a CHITIN arm) at k∈{1k,5k,100k}, GPU vs CPU: **`stat_prec` must be bit-identical (0 threshold-flip count)** — it is (#pvals < p_threshold)/k, an integer ratio, so identical unless a p-value crosses the threshold by float noise; report any flip explicitly. `wass_test` within rtol 1e-6; `spec_prec`/`sysvar_gap` likewise.
- **Cross-check against history:** GPU `stat_prec` for `sc_control_w25` matches obj_003.1's `prod_v21_verify.json` (0.991) and the exp_013 CPU values — the additivity guarantee holds across the GPU port.
- **Negative/edge cases:** empty regulator group; n_knock < min_cells (skipped identically); all-zero target column (tie group = N; variance→0 guard → p=1.0, matching scipy); k>n_edges; `--device cuda` with no GPU → clean CPU fallback.
- **Performance benchmark:** wall-time CPU vs GPU per graph for obj_003 and obj_004; target obj_004 <1 min/graph (from ~12), obj_003 <30 s/graph (from ~3). Record VRAM peak and, with `--cooldown-ms 250`, confirm a full multi-graph sweep runs without a WHEA event.

## Success Criteria (promotion gate)

1. **Exact `stat_prec`** (0 threshold-flips) and **`wass_test`/`spec_prec`/`sysvar` within rtol 1e-6** vs CPU scipy on the ≥3-graph validation set — the additivity guarantee is preserved.
2. **≥5× speedup** on obj_004 (the bigger bottleneck) and **≥4×** on obj_003, measured end-to-end.
3. **Thermal safety:** a full validation-set sweep completes with `--cooldown-ms 250` and no WHEA/driver crash; VRAM peak < 6 GB.
4. **Safe default:** `--device cpu` reproduces current behavior byte-for-byte; GPU is strictly opt-in.

## Promotion Path

On passing the gate, promote the **shared kernel as a common module both evaluators import**: copy `src/gpu_stat_kernels.py` + `src/gpu_context.py` into a new shared location (`OBJECTS/obj_003_grn_bio_evaluator_v2/src/gpu_stat_kernels.py`, imported by both obj_003 and obj_004 via a relative/`sys.path` shim, or a small shared `OBJECTS/_common/`), and merge the `--device`/`--gpu-batch-size`/`--cooldown-ms`/`--exact-fallback` dispatch into production `stat_metrics.py`, `grn_eval_v2.py`, `systema_metrics.py`, `systema_graph_eval.py`, plus the optional `gpu` cell-config block. Bump obj_003 to **v2.2** and note obj_004's GPU capability in its SESSION_LOG. Back up the edited production files first (the project's `_pre_*_backup_<ts>/` convention). Default stays `cpu`; **exp_014 opts into `--device cuda --cooldown-ms 250`** for its large sweep. This is a pure performance/infra change — no experiment's verdict changes, only its runtime.

## Critical References

[1] Mann-Whitney U asymptotic normal approximation with tie + continuity correction — the exact z/variance formula obj_003.2 reproduces: z = (U − n₁n₂/2 ± 0.5)/√σ², σ² = (n₁n₂/12)[(N+1) − Σ(t³−t)/(N(N−1))]. SciPy reference: `scipy.stats.mannwhitneyu` manual. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html — documents the `method="auto"` exact-vs-asymptotic switch (min-n≤8 & no ties → exact) that the `--exact-fallback` hybrid must honor for byte-exact reproduction.
[2] Akinshin, A. "Confusing tie correction in the classic Mann-Whitney U test implementation." https://aakinshin.net/posts/mw-confusing-tie-correction/ — precise treatment of the tie-correction term, to get the GPU variance identical to scipy on tie-heavy expression data. NEW resource.
[3] Python Optimal Transport (POT) — `ot.lp.solver_1d` / "Wasserstein 1D with PyTorch" torch backend: the `torch.searchsorted` cumulative-CDF construction for exact batched 1D Wasserstein on GPU. https://pythonot.github.io/master/auto_examples/backends/plot_wass1d_torch.html and https://pythonot.github.io/_modules/ot/lp/solver_1d.html — reference implementation for `wasserstein1d_gpu`. NEW resource.
[4] Koker, T. "torchsort: Fast, differentiable sorting and ranking in PyTorch." https://github.com/teddykoker/torchsort — evaluated and **rejected**: it provides *soft/differentiable* ranking; obj_003.2 needs exact hard ranks (native `torch.sort`/`argsort`), so torchsort is not a dependency. Documented to prevent a wrong turn. NEW resource.
[5] Pratapa, A., et al. (2020). "BEELINE." *Nature Methods* 17, 147–154 — the EPR metric obj_003.1's causal panel uses (unchanged here; context for why the eval stage is run at scale).
[6] Chevalley, M., et al. (2025). "CausalBench." *Communications Biology* 8, 412 — provenance of the `stat_prec`/Mann-Whitney perturbation-response evaluation being accelerated.
[7] Local code read (Phase 1; cloned, not edited in place): `OBJECTS/obj_003_grn_bio_evaluator_v2/src/stat_metrics.py` (the `groupby("Regulator", observed=True)` loop, `stats.mannwhitneyu(obs_slice, int_slice, axis=0)`, the per-target `stats.wasserstein_distance` inner loop, `min_cells`/`p_threshold`/`fraction_scored` bookkeeping) and `grn_eval_v2.py`; `OBJECTS/obj_004_systema_graph_eval/src/systema_metrics.py` (identical `stats.mannwhitneyu(ref_slice, int_slice, axis=0)` pattern; global-perturbed centroid reference ~28k cells) and `systema_graph_eval.py` (`_X_pert_test`/`_mu_pert` caching, `--batch`); `OBJECTS/obj_003.1_causal_evidence_panel/SESSION_LOG.md` (v2.1 validation baselines the GPU port must match); the Claude Code agent's GPU-feasibility dialog (2026-07-01) framing the kernel targets and the reproducibility/thermal risks.
