# FUNGI  Functional Unravelling of Network Geometry for Inference

**FUNGI** is a topology-aware graph sparsification pipeline that converts a dense, LightGBM-inferred Gene Regulatory Network (GRN) into a sparse, biologically plausible network optimized for downstream perturbation prediction via Graph Neural Networks. It sits at the heart of the **MYCELIUM** single-cell perturbation prediction framework.

```
Raw .h5ad → SPORE → CHITIN → PSGRN (LightGBM) → FUNGI → SPECTRA (GNN)
```

---

## The Problem FUNGI Solves

LightGBM-based GRN inference (e.g. GENIE3/GuanLab) produces dense parent graphs with tens of millions of scored edges across thousands of genes. Feeding such a graph directly into a GNN is computationally intractable and biologically noisy the vast majority of edges are spurious co-expression correlations, not causal regulatory relationships. FUNGI prunes this parent graph to a sparse candidate network of ~80–100K edges that satisfies quantitative biological topology constraints while preserving the most causally informative regulatory structure.

---

## Pipeline Overview

FUNGI operates in seven sequential phases, each building on the last.

### Phase 0  Diagnostic Calibration

Before any graph manipulation, FUNGI characterizes the biological signal in the expression data. For each CRISPRi perturbation in the training set, it runs Wilcoxon differential expression to identify statistically significant downstream DEGs. From these results it constructs:

- An **impact array:** the DEG count per perturbation, used to estimate per-gene regulatory output
- A **DEG matrix:** a binary (n_genes × n_genes) sparse matrix where entry [i, j] = 1 if gene j is a significant DEG when gene i is silenced
- An **LFC matrix:** log-fold-change vectors per perturbation, used for topology probe estimation

These structures feed six independent **topology probes** that estimate biologically grounded target bounds for the graph search:

| Parameter | Probe Method |
|-----------|-------------|
| α (scale-free exponent) | Power-law MLE on per-perturbation DEG counts with diffusion shift |
| Gini (hub inequality) | Panel-aware LFC-magnitude Gini for small panels; coherence-aware IDF for large |
| $S_{max}$ (largest hub fraction) | Hybrid DEG-directness + participation ratio estimator |
| Q (modularity) | Multi-resolution Louvain modularity on the LFC-causal graph |
| C (clustering coefficient) | Topweight transitivity on the parent graph at λ_eff edges/gene |
| ρ (assortativity) | Spearman of per-perturbation LFC L2 norms vs per-gene column L2 norms |

Each probe returns a confidence-weighted target interval [lo, hi]. The six intervals define the **utopian bounds**  the region of topology space that is simultaneously biologically plausible and data-consistent. Probe confidences are normalized into loss weights that scale each term's contribution to the search objective.

Phase 0 also estimates **λ_eff**  the expected edges per gene in the pruned graph  using a specificity-weighted median regulatory reach formula. λ_eff anchors the λ search range in Phase 3.

### Phase 1  Data Ingestion and Pre-filtering

The parent GRN parquet is loaded and converted to a weighted CSR sparse matrix. FUNGI uses **LightGBM Importance** (raw gain) as the structural backbone weight  this is non-negotiable, as it is the only signal that has been validated against gene expression prediction. Experimental columns (md_confidence, stability, consensus_sign, pert_efficiency) from enhanced PSGRN runs are carried forward as a separate dataframe for use as modulatory signals in Phase 2.

The parent graph is then pre-filtered to the top 10% of edges by Importance weight using O(N) argpartition. This reduces the candidate pool from ~25M to ~2.5M edges while retaining all high-confidence regulatory structure.

### Phase 2  Graph Normalization and Pre-computation

Phase 2 computes all the per-edge and per-gene scoring arrays that DASH will use throughout the search. These are computed once and held in memory (or distributed via Ray object store) for the duration of the search.

**W_q  Source-Quantile Weights**

Raw LightGBM Importance values vary enormously in scale across source genes. A hub gene with thousands of high-confidence outgoing edges would dominate a peripheral gene with a handful of weak edges if raw importance were used directly. W_q normalizes each edge's importance *within its source gene's outgoing distribution*, assigning rank-based values in (0, 1]:

```
W_q(e) = 1 - (rank_within_source + 0.5) / n_edges_for_source
```

This makes β (the W_q exponent) a meaningful gradient across the actual signal range for every gene, regardless of its absolute importance scale. Crucially, this normalization intentionally preserves cross-source inequality  genes with more outgoing edges achieve slightly higher rank-1 W_q values (approaching 1.0 as n→∞), which correctly allows hubs to claim more of the fixed edge budget and produce scale-free topology.

**PageRank κ Multipliers**

Power iteration PageRank is run on the pre-filtered graph. The top 1% of genes by PageRank score receive a hub_multiplier × kappa_base cap on outgoing edges, allowing genuine regulatory hubs to exceed the standard per-source edge cap when the data supports it.

**Perturbation Impact Prior (π)**

For each CRISPRi perturbation target gene, the number of DEGs it causes  normalized by its out-degree in the parent graph  gives a regulatory *efficiency* signal. Genes that cause many DEGs per available outgoing edge have high π. This is computed as:

```
π_s = (impact_s / mean_impact) / (deg_out_s / mean_deg_out)
```

π is floored at 1.0  perturbation genes with weak causal signal are treated as neutral, never penalized. Genes not in the perturbation set default to π = 1.0. The ψ hyperparameter then controls how aggressively this prior influences edge selection: `π_s^ψ`.

**Perturbation Pleiotropy Prior (χ)**

χ captures how frequently each gene appears as a *downstream DEG* across all training perturbations, derived from the column sums of the Phase 0 DEG matrix:

```
χ(s) = max(1.0, 1 + log(col_sum_s / g))^ζ
```

where g is the mean column sum across genes ever observed as a DEG, and ζ = 0.5 (fixed, not a hyperparameter). χ ranges [1.0, ~1.8]. A gene that appears as a DEG in many independent CRISPRi experiments is a regulatory convergence point  boosting edges to it reflects its biological centrality. This is a global signal defined for all 5,024 genes.

**DEG Row-Sum Prior (ρ)**

The complement to χ, ρ captures each perturbation gene's outgoing causal productivity from the row sums of the DEG matrix  how many targets it causally affects, normalized by out-degree. Like χ, it is log-damped with a floor at 1.0 and a fixed exponent φ = 0.5. ρ provides a boost-only correction for high-efficiency perturbation sources.

**Experimental Gate (G)**

When PSGRN is run in experimental mode with Mean Difference scoring and bootstrap stability estimation, FUNGI builds a per-edge multiplicative gate from two components:

- **Stability gate**  Beta-Binomial log-odds of bootstrap stability. Edges that appear consistently across LightGBM bootstrap iterations are boosted; unstable edges are penalized.
- **MD gate**  Per-source rank-normalized md_score × sign_agreement. Edges with strong, sign-consistent causal evidence from the CRISPRi perturbation data receive a boost proportional to their evidence strength.

These gates multiply together to form G_st ∈ [0.5, 2.0]. For standard GRNs without experimental columns, G = 1.0 (identity).

**Source-Conditioned Bridge Effective Resistance (SCBER)**

FUNGI implements a structurally-motivated edge scoring term based on approximate effective resistance, computed via a Johnson-Lindenstrauss sketch of the graph Laplacian. Rather than applying ER globally (which systematically over-selects inter-community bridges at the expense of intra-module hub→effector edges), FUNGI applies the ER boost selectively:

```
SCBER(s→t) = R_st^η_inter   if community(s) ≠ community(t)
           = 1.0             if community(s) == community(t)
```

Community membership is determined by Leiden community detection on the symmetrized pre-filtered graph, with resolution tuned to match the modularity target Q. Intra-module edges are left completely untouched. η_inter = 0.20 by default, giving a maximum ~1.82× spread between the most and least structurally critical inter-module bridges.

### Phase 3  Expansive Search

FUNGI evaluates a large quasi-random **Sobol sequence** of 4,096 hyperparameter configurations across the full 6D space. The Sobol sequence provides better coverage than random sampling by construction  it fills the space with low-discrepancy quasi-uniformity.

Each configuration is evaluated by the **DASH kernel** (see below), producing a utopia loss score. Results are sharded to disk for crash recovery. Shattered graphs (those violating hard structural constraints) are identified and excluded.

### Phase 4  Spatial Niching

The top-performing graphs from Phase 3 are clustered in normalized 6D hyperparameter space using K-Means. One local champion per cluster is extracted as an anchor coordinate. This ensures Phase 5 begins from a diverse set of promising regions rather than a tight cluster around the single best point.

### Phase 5  ML+GMM Refinement

Phase 5 runs a multi-round adaptive search around the Phase 3 anchors using a combination of:

- **Random Forest classifier** trained on Phase 3 shatter/viable labels to filter proposed candidates before evaluation
- **Gaussian Mixture Model** fitted to the top-performing viable graphs to propose new candidates in promising regions
- **Basin-aware density search**  once a sufficient zero-loss pool is established, DBSCAN detects distinct zero-loss basins in 6D space, each receiving a quality-proportional round and sample budget. A high-λ bias steers each density round toward the dense frontier of its basin.

The refinement terminates by patience-based exhaustion  when no meaningful edge-count improvement is observed across consecutive rounds.

### Phase 6  Cohort Selection

A champion graph and up to four topologically diverse alternates are selected from all evaluations via farthest-point sampling in normalized topology space. The champion is the densest zero-loss graph (or lowest-loss if no zero-loss exists). Alternates cover meaningfully different regions of topology space for downstream comparison or ensemble use.

### Phase 7  Graph Output

The champion's hyperparameter recipe is used to exactly reconstruct its edge list from the pre-sorted candidate pool. Output is a parquet file with columns Regulator, Target, Weight (and Sign if experimental GRN is active), ready for ingestion by SPECTRA.

---

## The DASH Kernel

DASH (Degree-Aware Sparsification Heuristic) is FUNGI's core edge scoring function. For each candidate edge (s→t) in the pre-filtered pool, DASH computes:

```
ω(s→t) = W_q^β × exp(δ × T̃_st) × π_s^ψ × G_st × SCBER_st × χ_s × ρ_s
```

| Term | Description |
|------|-------------|
| W_q^β | Source-quantile importance rank raised to the importance-steepness exponent |
| exp(δ × T̃_st) | Feed-Forward Loop (FFL) triangle density, degree-normalized via GraphBLAS |
| π_s^ψ | Perturbation impact prior for source gene s, floored at 1.0 |
| G_st | Experimental gate (stability × MD signal); identity for standard GRNs |
| SCBER_st | Source-conditioned bridge effective resistance |
| χ_s | Perturbation pleiotropy prior (col-sum DEG frequency) |
| ρ_s | DEG row-sum causal output efficiency prior |

Edges are selected by descending ω subject to:
- A global edge budget of λ × N_genes total edges
- A per-source hub cap of κ_base × PageRank_multiplier × N_genes outgoing edges
- Protection of the top-3 outgoing edges for each CRISPRi perturbation target

The **FFL term** `T̃_st` is computed using GraphBLAS sparse matrix multiplication to count degree-normalized triangles. An edge participates in a FFL if there exists an intermediate gene k such that s→k and k→t both exist among the top k_core × N_genes edges by weight. The degree normalization prevents high-degree hubs from accumulating inflated triangle counts purely by volume.

---

## The Utopia Loss Function

Given a selected graph, FUNGI measures how far its topology is from the utopian bounds estimated in Phase 0. For each of the six topology parameters, a smooth penalty function is applied:

```
L_total = √(Σ_p  w_p × f(observed_p, [lo_p, hi_p]))
```

where `f` is zero inside the bounds and grows as a sigmoid-gated squared distance outside them, with a 10% buffer zone at each boundary to prevent harsh gradients near the edges.

The six topology parameters and their biological interpretations:

| Parameter | What it measures | Target range |
|-----------|-----------------|--------------|
| α (scale-free exponent) | Degree distribution power-law fit  how scale-free the network is | [1.27, 1.66] |
| Gini | Inequality of hub out-degree  how dominated by a few masters | [0.46, 0.66] |
| S_max | Fraction of genes connected to the single largest hub | [0.04, 0.065] |
| Q (modularity) | Functional module separation  how distinct the regulatory communities are | [0.24, 0.44] |
| C (clustering) | Feed-forward loop density  triangular motif prevalence | [0.05, 0.12] |
| ρ (assortativity) | Hub-to-effector connectivity bias  whether hubs connect to effectors, not other hubs | [-0.10, -0.02] |

A graph with utopia_loss = 0 satisfies all six topology targets simultaneously. The loss function's smooth penalties and confidence-weighted terms allow the optimizer to navigate near-boundary regions without hard failures.

An additional **connectivity penalty** is applied if the giant weakly connected component covers less than 45% of all genes  ensuring the selected graph is not a fragmented collection of isolated subgraphs.

---

## The Six Hyperparameters

| Parameter | Symbol | Range | Role |
|-----------|--------|-------|------|
| Importance steepness | β | [1.0, 4.0] | Controls how aggressively high-importance edges beat lower ones. Higher β concentrates budget on the very top edges. |
| FFL boost | δ | [0.0, 1.5] | Controls the reward for feed-forward loop participation. Higher δ selects more triangulated, modular structure. |
| Hub cap | κ | [0.02, 0.15] | Maximum fraction of N_genes outgoing edges any source gene can claim (before PageRank multiplier). Controls S_max directly. |
| FFL window | k_core | [8, 28] | Number of top edges per gene considered when counting FFL triangles. Wider window = richer FFL context. |
| Edge density | λ | [10, 35] edges/gene | Total edges in the output graph = λ × N_genes. Primary driver of global connectivity. |
| Perturbation weight | ψ | [0.0, 3.0] | Exponent on π_s. Higher ψ more aggressively boosts well-connected perturbation source genes. |

---

## Shatter Detection

Before computing utopia loss, FUNGI checks four hard structural constraints. A graph that fails any of these is marked as shattered and assigned loss = 999:

- **Density collapse**  edge count exceeds the λ_max × N_genes ceiling
- **Orphan collapse**  more than 15% of genes have zero in- and out-degree
- **GWCC percolation**  the giant weakly connected component covers less than 50% of genes
- **Clustering collapse**  global clustering coefficient falls below a data-derived floor

---

## Implementation Notes

- **Parallelism**  Phase 3 and 5 evaluations run in parallel via [Ray](https://ray.io), with each worker receiving pre-sorted edge arrays via Ray's object store. Crash-safe shard checkpointing allows interrupted runs to resume without restarting.
- **GraphBLAS**  FFL triangle computation uses [python-graphblas](https://github.com/python-graphblas/python-graphblas) for sparse semiring operations, achieving orders-of-magnitude speedup over dense adjacency multiplication.
- **Sobol sampling**  Phase 3 uses scrambled Sobol sequences via `scipy.stats.qmc` for low-discrepancy 6D coverage.
- **Community detection**  SCBER uses igraph's built-in Leiden algorithm with a brief resolution sweep targeting Q ≈ 0.5.

---

## Dependencies

```
numpy  scipy  pandas  anndata  scanpy
lightgbm  joblib  ray
graphblas (python-graphblas)
igraph  networkx
scikit-learn  tqdm  pyyaml
```

---

## Repository Structure

```
FUNGI/
├── FUNGI.ipynb                  # Main pipeline notebook
├── fungi_config.yaml            # All configuration and hyperparameter bounds
├── src/
│   ├── engine.py                # DASH kernel, loss function, graph reconstruction
│   ├── diagnostics.py           # Phase 0: topology probes, DEG matrix, λ_eff
│   ├── filtering.py             # Pre-filter (adaptive threshold)
│   ├── graph_utils.py           # Graph loading, structural metrics
│   ├── effective_resistance.py  # SCBER computation (Leiden + JL sketch)
│   ├── search.py                # Sobol generation, SearchEvaluator, Ray wiring
│   ├── niching.py               # Phase 4: spatial niching, anchor extraction
│   ├── refinement.py            # Phase 5: GMM+RF refinement, basin search
│   └── topology.py              # Persistent homology, EPR (audit utilities)
└── data/
    └── output/                  # Pipeline outputs (created at runtime)
```

---

## Citation

If you use FUNGI in your research, please cite:

> Sheehan, P. *FUNGI: Functional Unravelling of Network Geometry for Inference.*
> Human Technopole / University of Milan. 2026.

---

## License

MIT License  see `LICENSE` for details.
