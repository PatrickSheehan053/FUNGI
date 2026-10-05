# FUNGI

FUNGI builds sparse, biologically calibrated gene regulatory networks (GRNs) from single-cell CRISPRi
Perturb-seq data. It fuses a causal graph and a co-expression graph into one dense Super-graph, then prunes it
with the DASH kernel into a sparse graph that downstream perturbation-prediction models can use.

## Pipeline

| step | folder | what it does |
|---|---|---|
| 1 | `spore/` | Cleans and splits the raw Perturb-seq data, keeps every held-out perturbation target in the gene panel, and builds the control-preserving MBK k=2 metacell substrate. Also works as a standalone preprocessing tool. |
| 2a | `coexpression_generator/` | Dense co-expression graph: one LightGBM regression per target gene (GENIE3/GRNBoost2 style), bootstrapped. CPU / SLURM array. |
| 2b | `causality_generator/` | Causal graph from perturbation responses (PSGRN Self-Train, Song et al. 2026). GPU. |
| 3 | `supergraph/` | Fuses the causal and co-expression graphs into one dense directed Super-graph. |
| 4 | `fungi/` | Prunes the Super-graph with the DASH kernel and Sobol search into a sparse graph that hits literature topology targets. |
| 5 | `rhizo/` | Zero-ontology directed GNN that predicts perturbation responses from the graph alone. Used as the arbiter to compare graphs. |
| — | `baselines/` | Edge-matched Top-Weight and KNN pruned graphs, plus null graphs (shuffle, reverse, empty). |
| — | `evaluation/` | De-confounded downstream scorer (`pearson_systema`), GRN bio-evaluator (gold-standard panels), SYSTEMA graph eval. |

Data flows: raw `.h5ad` → **SPORE** → train substrate `.h5ad` → **co-expression** + **causality** dense graphs
(`Regulator, Target, Importance/Weight` parquet) → **Super-graph** → **FUNGI** sparse graph → **RHIZO** + **evaluation**.

## Status (October 2026)

This repo is assembled from the thesis working tree. Read `docs/KNOWN_ISSUES.md` before running anything. In short:
- The co-expression worker that ran on the org HPC on 9 July is not in this repo yet (see `coexpression_generator/README.md`).
- Many scripts still contain hard-coded paths from the original machine and cluster.
- Internal identifiers, log strings and config keys still use the thesis names (HYPHAE = co-expression generator,
  SHROOM = causality generator, MYCELIUM = whole pipeline).

`docs/PROVENANCE.csv` maps every file to its original location, modification time and md5.
