# Known issues (repo assembled 5 Oct 2026)

## 1. Co-expression worker v4 is a rebuild
The worker that ran on HPC1 on 9 July is not in this repo. `coexpression_generator/coexpression_worker_v4_rebuilt_UNTESTED.py`
is reconstructed from the handoff notes. See `coexpression_generator/README.md`.

## 2. Hard-coded paths
These files still point at `C:/Users/studi/...`, `/scratch/...` or the thesis `DATA/EXPERIMENTS` tree. They need
CLI arguments or config keys before anyone else can run them:

- `spore/configs/spore_config_rpe1.yaml`, `spore/configs/spore_config_rpe1_canary.yaml`, `spore/src/phase04_doublets.py`
- `supergraph/build_super_parent.py` (`REPO`, `LFC`, `DENSE_CAUSAL`, `DENSE_COEXP`)
- `fungi/configs/*.yaml` (`sc_data_path`, graph paths)
- `baselines/build_rpe1_graphs.py`, `baselines/build_nulls.py`
- `rhizo/src/build_cellhalf_lfc.py`, `rhizo/src/build_rhizomorph.py`
- `evaluation/anastomosis_scorer/panel.py`, `evaluation/anastomosis_scorer/gpu_kernels.py`

Already rewired to repo-relative defaults: `spore/spore.py` (finds its own `src/`), `spore/run_k562_pipeline.sh`,
`fungi/scripts/fungi_prune.py`, `fungi/scripts/build_fungi_cache.py`.

## 3. Which version of each stage is here
| stage | version in repo | source |
|---|---|---|
| SPORE | one-stop coverage-fixed rebuild (8 Jul): automatic held-out coverage, native MBK k=2 Phase 12, config validation, final verification gate, CHITIN removed | `OBJECTS/obj_009.1.../HPC_K562_SPORE_launch` |
| Causality generator | PSGRN Self-Train production (`shroom.py`, 23 Jun) | `SHROOM/` |
| Super-graph | exp_035 dense fusion (`--causal-norm`, `--fusion-wc`). Thesis exp_033 used a Borda fusion of the two FUNGI-pruned pillars instead (`baselines/graph_tools.py`). | `exp_035/src`, `exp_033/src` |
| FUNGI | kernel_firepower_v2 clone (19 Jul): production kernel plus `sigma_scber`, `zeta_s`, `tau_indeg`, `eta_out`, `theta_pa`, `kappa_frac` levers. Each lever reproduces production byte-identical at its identity value. Production thesis config kept as `fungi_config_thesis_production.yaml`. | `exp_035/kernel_firepower_v2/clone` |
| RHIZO | obj_009.3 tree + patch3 data builders + patch4 gauntlet | `exp_033/clone/harness_gauntlet_src`, `exp_031*/patch*` |
| Scorer | obj_010 + patch6 (f1_auc fix) | `OBJECTS/obj_010_anastomosis`, `exp_031b/patch6` |

The exact RHIZO tree that ran the thesis results (`/scratch/patrick.sheehan/final_weekend/ship_patch2/` on hpc2)
had live edits that are not on the laptop. Pull it when access returns and diff against `rhizo/`.

## 4. Thesis names still inside the code
Log strings, conda env names, config keys and docstrings still say HYPHAE, SHROOM, SPORE_light, MYCELIUM,
obj_00n and exp_0NN. Only file and folder names were changed. A full rename pass is a follow-up.

## 5. No data in the repo
Data and large files (`.h5ad`, `.parquet`, `.npz`, `.pkl`, archives, anything over 5 MB) were excluded on purpose.
Graphs and substrates should go to Zenodo or similar and be linked from the README.
