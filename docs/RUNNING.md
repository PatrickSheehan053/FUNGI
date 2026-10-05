# Running FUNGI

The step-by-step guide to running the full pipeline, from raw Perturb-seq counts to a scored graph. Back to the [main README](../README.md).

Run the stages in order. Each stage writes a file the next stage reads, and the gene panel and split produced by SPORE must stay fixed from step 1 onward. Several scripts still contain paths from the original development machine, so check [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) before the first run and point every config at your own data.

## Step 1: clean and split the data with SPORE

```bash
pip install -r spore/requirements.txt
python spore/spore.py --config spore/configs/spore_config_rpe1.yaml
```

Set `paths.raw_h5ad`, `perturbation_col`, `control_label`, and the test and validation sizes in the config. SPORE checks the config against the data at startup and stops immediately if they disagree. The outputs you need are `{name}_train_hybrid.h5ad` (the graph-building substrate, training perturbations and controls only), `{name}_allsplits_metacell_ctrlpreserved.h5ad`, the split file, and `COVERAGE_REPORT.md`.

## Step 2a: build the co-expression graph

The co-expression generator is a SLURM array job. Each of 10 tasks takes a slice of the gene list and fits its genes in parallel. Copy `{name}_train_hybrid.h5ad` into `coexpression_generator/input/`, select it in the `SELECT DATASET` block of the submit script, and submit.

```bash
cd coexpression_generator
conda env create -f environment.yml
sbatch submit_coexpression.sh
# when all 10 tasks finish
python consolidate_graphs.py --chunk_dir chunks_<run_name> --output_file coexp_dense.parquet \
    --total_tasks 10 --h5ad_file input/<name>_train_hybrid.h5ad
```

Every gene is checkpointed to its own file as it finishes, so a task that dies or hits its wall time resumes where it stopped when resubmitted. This is the heaviest stage. Expect roughly 18 hours across 10 nodes of 32 cores on a 5,000-gene metacell substrate, and budget memory by measured per-worker cost rather than matrix size (see `coexpression_generator/README.md`). The output columns are `Regulator, Target, Importance`.

## Step 2b: build the causal graph

```bash
python causality_generator/causality_generator.py \
    --mc-input <name>_train_hybrid.h5ad --sc-input <name>_train_hybrid.h5ad \
    --output causal_dense.parquet --pert-col <perturbation column> \
    --control-label non-targeting --device cuda
```

Feed raw counts. The generator normalizes internally. It runs on one GPU in a fraction of the co-expression generator's cluster time.

## Step 3: fuse the two graphs into the Super graph

```bash
python supergraph/build_super_parent.py --mode fuse_then_prune --out super_dense.parquet
```

Set `DENSE_CAUSAL` and `DENSE_COEXP` at the top of the script to the two dense graphs from step 2. The default operator rank-normalizes each graph within each source gene and keeps the larger of the two ranks for every edge (rank-max union). The script restricts causal sources to training-perturbed regulators, so no held-out perturbation contributes an out-edge, and it writes a QC report next to the graph. `--causal-norm` and `--fusion-wc` switch to the alternative fusion operators. The thesis recipe, which prunes each graph first and fuses the two pruned graphs by Borda count, lives in `baselines/graph_tools.py`.

## Step 4: prune with FUNGI

```bash
python fungi/scripts/build_fungi_cache.py --graph super_dense.parquet --cache-dir cache/super --tag super
python fungi/scripts/fungi_prune.py --cache-dir cache/super --tag super \
    --lam-lo 40 --lam-hi 48 --max-edge-count 400000 --refine
```

The first command runs FUNGI's diagnostic calibration and builds the candidate pool once, and the second runs the search and writes the champion graph (`Regulator, Target, Weight`) to `fungi/outputs/fungi/champions_full/`, with the run record in `fungi/outputs/fungi/<tag>/`. `--lam-lo` and `--lam-hi` bound the density in edges per gene, so 40 to 48 on a 5,000-gene panel gives a graph of roughly 200,000 to 240,000 edges. The topology bands, hyperparameter ranges, and priors are set in `fungi/configs/`. The same two commands prune a co-expression graph on its own.

## Step 5: test the graph with RHIZO and score it

```bash
cd rhizo
python src/build_rpe1_gauntlet_data.py --root <bundle>
python src/build_caches_v3f.py --arms fungi_bio,top_weight,knn
RHIZO_ONLY_ARMS=fungi_bio,top_weight,knn \
    python src/gauntlet_v3f.py --config configs/core_anchor_b8.json --device cuda
python ../evaluation/anastomosis_scorer/score_preds.py --preds results/preds
```

The data builder reads each graph to test from `<bundle>/arms/<name>.npz` together with the held-out expression changes, and writes the training package and `data/arm_graphs/<name>.npz` that RHIZO reads. It currently targets the RPE1 layout, so a new dataset needs its own copy of this step. The gauntlet trains RHIZO on every graph under identical settings and seeds, writes one resumable result file per cell, and dumps predictions that the scorer turns into `pearson_systema`, DEG-AUPRC, and the raw correlation metrics. RHIZO fits an 8 GB laptop GPU at a small batch size. The full multi-seed comparison runs fastest on a data-centre GPU. `rhizo/docs/README_RHIZO_FINAL.md` describes the configs and the control arms.

---
