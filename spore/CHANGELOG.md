# SPORE one-stop rebuild — change log (exp_009.1)

Goal: make `python spore_light.py --config <dataset>.yaml` produce a HYPHAE-ready substrate + coverage/
firewall proof in **one run**, with no manual patch scripts and no re-runs. Every change below is inside
`HPC_K562_SPORE_launch/` (a clone). Production SPORE at the repo root was NOT touched. Backup of the
pre-rebuild package: `_pre_rebuild_backup_<ts>/`.

## Defect → fix map

**D1 — no on-disk HVG-panel raw file.** Phase 8 now emits `{name}_{split}_hvgraw.h5ad` (HVG panel, RAW
counts) right after HVG selection and before Phase 9 normalizes the in-memory splits
(`spore_light.py::emit_hvgraw_splits`). This is THE builder input; Phase 12 reads it.

**D2 — held-out targets silently under-covered.** Phase 8 now auto-force-carries EVERY val/test
perturbation-target identity from `{name}_split_indices.json` into the HVG panel, independent of any curated
protected list (`phase08_hvg.py::load_heldout_target_ids` + `force_include_into_hvg`, wired into both the
small path `select_hvgs` and the large/worker path `run_worker_logic`). The worker's `--protected-gene-list`
plumbing (which raised a TypeError and was never passed by the launcher) is fixed. A loud coverage report is
printed after Phase 8 (`report_heldout_coverage`): present/missing held-out targets + the
perturbed-but-never-measured ceiling.

**D3 — Phase 12 built the wrong (discarded) metacell.** Phase 12 is rewritten as the native
control-preserving MBK k=2 builder (blessed recipe ported verbatim from `metacell_sweep.py`/`agg_variants.py`).
It emits `{name}_allsplits_metacell_ctrlpreserved.h5ad` + `{name}_train_hybrid.h5ad` +
`{name}_allsplits_metacell_split_indices.json` directly, with the hard-fail guards (controls single-cell,
split disjoint+complete, one-pert-per-metacell) and the firewall subset inside SPORE. The old standalone
builder/verify scripts are retired to `_superseded_by_phase12/`.

**D4 — broken/missing packaging.** Added `requirements.txt` (every third-party import across the workflow,
incl. `joblib`/`threadpoolctl`/`harmonypy`/`h5py`/`mygene`). The `clone/` import chain is gone (recipe is
inline in Phase 12). Fixed a latent break: `phase08_hvg.py`'s top-level `from .utils` import crashed the
large/K562 subprocess worker — now falls back to an absolute import.

**D5 — config silently mismatches data.** `spore_light.py::validate_config_against_data` runs at startup and
HARD-FAILS on: perturbation_col/control_label not in obs; batch_key (now read from
`phase10_confounders.batch_correction.batch_key`, not the dead `dataset.batch_col`) not in obs when batch
correction is enabled; gene_id_format contradicting the var_names; test_n+val_n leaving no train perts.
Gene-ID harmonization now uses an in-file symbol column (`gene_name`) offline instead of the online mygene
query when available.

**D6 — CHITIN dead code.** Deleted `phase13_chitin.py` + `engine.py` (ChitinModel); stripped
`chitin_output_dir` from configs + both path resolvers; removed the `report_phase13_*` diagnostics and the
`_chitin_output` references.

**D7 — no internal verification gate.** `spore_light.py::final_verification_gate` runs at the end and refuses
success unless (1) coverage is at the measurable ceiling (missing ⊆ never-measured), (2) the train-hybrid
firewall holds (0 val/test units), (3) the substrate is HVG-panel RAW counts. Writes `COVERAGE_REPORT.{md,json}`
and the process exits non-zero on FAIL.

## Acceptance test
`python spore_light.py --config <dataset>.yaml` → `{name}_allsplits_metacell_ctrlpreserved.h5ad` +
`{name}_train_hybrid.h5ad` + `COVERAGE_REPORT.md`, no manual steps. Validated on the local RPE1 raw
(247,914 cells; same in-process path VCC uses).
