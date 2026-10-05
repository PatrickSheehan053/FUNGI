# How FUNGI works

The stages in more depth, the input restriction every user needs to know, and the mechanics of the FUNGI pruner. Back to the [main README](../README.md).

## The problem FUNGI solves

A gene regulatory network is the map of which genes control which others, and it is what a biologist reaches for to reason about how a cell will respond when a gene is switched off. Perturb-seq now measures the transcriptome-wide response to knocking down thousands of individual genes across hundreds of thousands of cells [5], which turns that question into a concrete prediction problem. Given a regulatory map, can a model predict the response to a perturbation it has never seen?

Three obstacles make the problem hard. No ground-truth GRN exists to train against, so every inferred network is an estimate that can only be judged indirectly. The inference methods that produce those estimates return dense graphs in which almost every gene pair carries some weight, and a downstream graph model run over that density blends every gene's signal into every other's. The perturbation data is also dominated by a large shared response that imitates regulatory biology and inflates any naive accuracy score [6].

FUNGI takes the three on together. It builds the densest, most informative candidate graph the data supports, prunes it against explicit, measurable topology targets instead of a fixed threshold, and judges the result with an evaluation that strips out the shared response before scoring.

---

## The pipeline at a glance

FUNGI runs as five stages. Each stage is also a standalone tool, so a user who only needs a clean co-expression graph, or who already has a dense graph and only wants it pruned, can run that piece alone.

| step | folder | what it does | hardware |
|---|---|---|---|
| 1 | `spore/` | Cleans the raw data, keeps every held-out perturbation target in the gene panel, builds gene-disjoint splits, and emits the control-preserving metacell substrate. | CPU, workstation |
| 2a | `coexpression_generator/` | Infers a dense co-expression graph with one gradient-boosted regression per gene. Every gene gets out-edges. | CPU cluster |
| 2b | `causality_generator/` | Infers a dense causal graph from the measured knockdown responses. Edges point from a silenced gene to the genes it moves. | single GPU |
| 3 | `supergraph/` | Fuses the causal and co-expression graphs into one directed Super graph. | CPU |
| 4 | `fungi/` | Prunes the Super graph (or a co-expression graph) to a sparse graph that sits inside literature topology bands. | single GPU |
| 5 | `rhizo/` | Predicts perturbation responses from the graph alone, holding the model fixed so any change in accuracy belongs to the graph. | single GPU |
| | `evaluation/` | Scores predictions on the de-confounded `pearson_systema` axis, plus graph-level bio and SYSTEMA evaluators. | GPU or CPU |
| | `baselines/` | Builds edge-matched Top-Weight and KNN graphs and null graphs (shuffle, reverse, empty) for comparison. | CPU |

The stage names come from the thesis this work grew out of, and the code still uses them. HYPHAE is the co-expression generator, SHROOM is the causality generator, and MYCELIUM was the name of the whole pipeline.

### SPORE: Systematic Preprocessing and Optimization for Robust Evaluation

SPORE prepares the substrate every later stage reads. It runs as a fixed sequence of numbered phases (ingestion, detection, cell triage, ambient RNA, doublets, escaper filtering, gene triage, splitting, HVG selection, normalization, confounders, cell-line separation, metacells), each reading the last phase's output, so a run is auditable and can resume mid-way. Two of its jobs matter most downstream. SPORE splits the data by perturbation, so no perturbed gene seen in training appears in evaluation, and it force-carries every held-out perturbation target into the gene panel, so the graph contains a node for every gene the evaluation will ask about. It then builds the control-preserving metacell substrate, pooling each perturbation's cells into small metacells (MiniBatchKMeans, k=2) while keeping control cells single, which denoises dropout without blurring the perturbation-versus-control contrast. SPORE refuses to report success unless coverage and the leakage firewall both pass, and it writes a `COVERAGE_REPORT.md` as proof. It also works as a general-purpose cleaner for any Perturb-seq dataset.

### The co-expression generator (HYPHAE)

The co-expression generator infers edges the way GENIE3 does [1]. For each target gene it fits a regression that predicts that gene from every other gene, and it reads each predictor's feature importance as the weight of a `Regulator → Target` edge. It uses the gradient-boosted form popularized by GRNBoost2 [2], fitting one LightGBM model per gene, bootstrapping each gene over repeated 80/20 splits with early stopping, and averaging the gain importances. Because every gene serves as a predictor for every other gene, every gene receives out-edges, including genes that were never perturbed. That full coverage is its defining strength. Its weakness follows from the same association basis. Co-expression is close to symmetric, so the direction of an edge is weakly identified, and two genes driven by a common third gene score as strongly as a true regulator and its target.

### The causality generator (SHROOM)

The causality generator is a GPU implementation of PSGRN self-training [3]. It describes every directed gene pair with four numbers. Two are control baselines for the source and the target, and two are interventional, the source and the target's expression in the cells where the source itself was knocked down. A single classifier, trained on correlation-derived pseudolabels, scores every pair from those features. The interventional features read what actually changes when the source is silenced, so the score of an edge depends on which gene was the intervention and the edge carries a direction. That makes the causal graph highly regulator-specific. It also ties its coverage to the perturbation set, which the next section explains.

### The Super graph

The two generators fail in opposite places. The co-expression graph covers every gene but cannot orient its edges, and the causal graph orients its edges but only for perturbed genes. DREAM5 showed that no single inference method wins across networks and that aggregating complementary methods is more robust than any one of them [4], and NIMEFI carried that result into rank-based aggregation of ensemble GRN methods [7]. The Super graph applies the same wisdom-of-crowds logic to one causal and one co-expression graph. Each graph's weights are first converted to normalized ranks, because gain importances and classifier probabilities live on different scales, and the ranks are then fused so the result inherits causal direction where it exists and co-expression coverage everywhere else.

![Super graph construction](figures/super_graph_construction.png)

*The causal core contributes a few specific hubs, the co-expression scaffold contributes coverage of every gene, and the fused Super graph carries both.*

### FUNGI: the pruner

The FUNGI pruner turns the dense Super graph into a usable one. Instead of keeping the top-weighted edges or capping every gene's degree, it treats the shape of a real GRN as an explicit objective and searches for the sparse graph that matches it. The section [How the FUNGI pruner works](#how-the-fungi-pruner-works) gives the detail.

### RHIZO: Regulatory Hop Inference, Zero Ontology

RHIZO is the downstream model. It takes a graph and the identity of a perturbed gene and predicts the genome-wide expression response, propagating the perturbation one regulatory hop at a time through a lightweight directed message-passing network. It carries zero ontology. RHIZO uses no gene-ontology graph, no pretrained gene embeddings, and no per-gene identity features, so every edge feature it reads is computed from the graph it is given. When RHIZO predicts better on one graph than on another, the graph is the only thing that changed. That property makes RHIZO the arbiter for every graph comparison in this repository.

---

## Co-expression alone works, causal alone does not

FUNGI accepts two kinds of input graph. A co-expression graph can be pruned and used on its own. A causal graph cannot serve as the only input, and this distinction matters for anyone bringing their own data.

A causal edge is scored from what happens to the target when the source is knocked down, so only a gene that was actually perturbed can carry an out-edge. Every gene that was never perturbed becomes a sink, with incoming edges and no outgoing ones. On the RPE1 panel, 866 of the 5,000 genes (17.3%) were perturbation sources. On the h1-hESC panel the figure falls to 88 of 5,000 (1.8%). A causal-only graph therefore leaves more than 80% of the panel unable to regulate anything.

![Who can be a source](figures/who_can_be_a_source.png)

*The co-expression generator can place an out-edge from every gene in the panel. The causal generator can only place one from a perturbed gene. On the held-out perturbations of the RPE1 test, the causal graph has out-edges for 0 of 373, so it has nothing to propagate.*

The consequence reaches the downstream model directly. RHIZO is built so that a gene the perturbation never reaches predicts exactly zero, which keeps it from inventing signal that the graph does not carry. A perturbed gene with no out-edges reaches nothing, so on a causal-only graph every held-out perturbation produces a flat, unscoreable prediction.

![RHIZO predicts zero for unreached genes](figures/rhizo_unreached_gene_zero.png)

*The perturbation (star) propagates along out-edges. Genes it never reaches stay at exactly zero. A source with no out-edges reaches nothing.*

FUNGI will mechanically prune a causal-only graph, and inside the perturbed set that graph is highly regulator-specific. However, it cannot speak for any regulator that was never perturbed, so it cannot serve as a general-purpose GRN. The only setting where a causal-only graph stands alone is a dataset in which every gene in the panel was perturbed, which is the regime PSGRN was originally designed and validated in [3]. Genome-scale panels with a few hundred to a thousand perturbations are far from that regime. The supported inputs are therefore a co-expression graph on its own, or the Super graph, which keeps the causal core's specificity and borrows coverage from the co-expression scaffold. The Super graph is the recommended input.

---

---

## How the FUNGI pruner works

### Topology targets

A real GRN differs from a random graph in measurable ways. A small set of transcription-factor hubs regulate many targets while most genes regulate few or none, regulators cluster into feed-forward motifs, hubs tend to connect to low-degree genes, and most regulation runs in one direction. FUNGI turns each of these signatures into a target band and prunes toward all of them at once.

| target | what it measures | literature band |
|---|---|---|
| `alpha` | power-law exponent of the degree distribution (scale-free hub hierarchy) | 2.0 to 2.8 |
| `gini_in` | inequality of in-degree across genes | 0.45 to 0.60 |
| `S_max` | out-degree of the largest hub as a fraction of the panel | 0.09 to 0.18 |
| `C` | clustering, which tracks feed-forward-loop density | 0.08 to 0.20 |
| `rho` | degree assortativity (negative means hubs attach to low-degree genes) | -0.35 to -0.05 |
| `reciprocity` | fraction of edges whose reverse also exists | 0.02 to 0.12 [8] |

Out-degree inequality and modularity are also defined, and they are disabled by default because the current dense parents cannot reach them. Each band can come from one of three sources, chosen per target. By default FUNGI runs a diagnostic probe suite on the dense parent and derives the band from the data. A curated literature band can replace any probe, and the user can override any band directly in the config. That choice is what makes FUNGI dataset-agnostic, because the definition of a good graph travels with the data instead of being hard-coded.

### The utopian loss

The utopian loss collapses the targets into one number. A target inside its band contributes nothing, and a target outside it contributes a weighted penalty that grows with the distance, so a loss of zero means every engaged target is satisfied at once. FUNGI only removes edges and never adds one the parent did not contain, so the reachable graphs are bounded by the parent. A loss floor that stays above zero is therefore a diagnostic. It says either that the target is wrong for this biology or that the parent lacks the structure to express it.

### The DASH kernel

The loss scores whole graphs. The DASH kernel, adapted from the domain-knowledge pruning approach of Hossain et al. [9], scores individual edges. For a candidate edge $e = (s \to t)$ it combines every source of evidence into one score,

$$\omega(e) = W_q(e)^{\beta} \cdot e^{\delta \tilde{T}_{st}} \cdot \pi(s)^{\psi} \cdot R(s)^{\nu} \cdot m_{\text{intra}}^{\mathbb{1}[\text{intra}(e)]} \cdot \text{ER}(e)^{\eta} \cdot \chi_s(s)\,\chi_t(t) \cdot \rho_c(s)$$

where $W_q(e)$ is the inferred edge weight re-scaled by quantile within its source gene, $\tilde{T}_{st}$ counts the feed-forward-loop triangles the edge closes, $\pi(s)$ is the measured perturbation impact of the source, $R(s)$ is a reachability-diffusion prior that rewards sources with broad, diverse regulatory reach, $m_{\text{intra}}$ boosts edges inside a community, $\text{ER}(e)$ is a source-conditioned effective-resistance term that protects bridges between modules, $\chi_s$ and $\chi_t$ are pleiotropy priors on the source and target, and $\rho_c(s)$ is a causal-output prior on the source. The product form means no single term can veto an edge the rest of the evidence supports, and any factor drops out cleanly at its neutral value. FUNGI keeps the top $\lfloor n\lambda \rceil$ edges by $\omega$, with $n$ genes and $\lambda$ edges per gene, under a per-gene hub cap.

### The search

A run moves through a fixed sequence of phases.

* 1) **Ingestion** aligns the dense parent and the expression data on one gene index.
* 2) **Diagnostic calibration** measures perturbation impact and sets the target bands and their weights, once per run.
* 3) **Candidate pool** filters the parent and pre-computes every prior on the GPU.
* 4) **Expansive search** samples thousands of hyperparameter settings with a Sobol sequence and scores each resulting graph against the utopian loss.
* 5) **Niching and refinement** cluster the best settings and search around them with a random-forest surrogate.
* 6) **Champion selection** returns the single graph the run commits to, the densest zero-loss graph when the search finds one.

| hyperparameter | role | default range |
|---|---|---|
| `beta` | steepness of the weight term | 0.5 to 4.0 |
| `delta` | feed-forward-loop boost | 0.0 to 3.0 |
| `kappa` | per-gene hub cap | 0.02 to 0.35 |
| `k_core` | feed-forward-loop window | 8 to 28 |
| `lambda` | density, edges per gene | set per run |
| `psi` | perturbation-impact weight | 0.0 to 3.0 |
| `nu` | reachability-diffusion prior weight | 0.0 to 2.0 |
| `m_intra` | within-community boost | 1.0 to 4.0 |

Several fixed priors can also be opened to the search (`sigma_scber`, `zeta_s`, `tau_indeg`, `eta_out`, `theta_pa`). Each reproduces the base kernel exactly at its neutral value.

---

---

## References

1. Huynh-Thu, V. A., Irrthum, A., Wehenkel, L. & Geurts, P. Inferring regulatory networks from expression data using tree-based methods. *PLoS One* 5, e12776 (2010).
2. Moerman, T. et al. GRNBoost2 and Arboreto: efficient and scalable inference of gene regulatory networks. *Bioinformatics* 35, 2159–2161 (2019).
3. Song, X., Deng, K., Chen, M. & Guan, Y. PSGRN: gene regulatory network inference from single-cell perturbational data through self-training with synthetic gold standards. *Sci. Adv.* 12, eaeb3376 (2026).
4. Marbach, D. et al. Wisdom of crowds for robust gene network inference. *Nat. Methods* 9, 796–804 (2012).
5. Replogle, J. M. et al. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell* 185, 2559–2575 (2022).
6. Viñas Torné, R. et al. Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation. *Nat. Biotechnol.* (2025).
7. Ruyssinck, J. et al. NIMEFI: gene regulatory network inference using multiple ensemble feature importance algorithms. *PLoS One* 9, e92709 (2014).
8. Garlaschelli, D. & Loffredo, M. I. Patterns of link reciprocity in directed networks. *Phys. Rev. Lett.* 93, 268701 (2004).
9. Hossain, I., Fischer, J., Burkholz, R. & Quackenbush, J. Pruning neural network models for gene regulatory dynamics using data and domain knowledge. Preprint at arXiv:2403.04805 (2024).
10. Roohani, Y. et al. Virtual Cell Challenge: toward a Turing test for the virtual cell. *Cell* 188, 3370–3374 (2025).
11. Ahlmann-Eltze, C., Huber, W. & Anders, S. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nat. Methods* 22, 1657–1661 (2025).
