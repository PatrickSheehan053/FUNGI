# FUNGI results

The evidence behind the headline claims, with the controls. Back to the [main README](../README.md).

All results below come from two CRISPRi Perturb-seq screens. RPE1 is the genome-scale screen of Replogle et al. [5], run on a 5,000-gene panel with 1,514 perturbations split by gene into 1,058 training, 152 validation, and 304 test perturbations. h1-hESC is the human embryonic stem cell screen from the Virtual Cell Challenge [10], with roughly ten times sparser causal coverage. The two lines share the same panel size, so a difference between them reflects biology and coverage.

## Naive accuracy is fooled, so FUNGI scores on a de-confounded axis

Every perturbation in these screens shares a large common response, and a model that predicts that average response scores well on raw correlation without knowing anything about regulation [6]. FUNGI's evaluation therefore uses `pearson_systema`, the per-perturbation correlation between predicted and true expression change after the shared response is projected out. It credits only signal that is specific to the perturbed regulator, and its sign is diagnostic.

![Naive versus de-confounded scoring](figures/naive_vs_deconfounded.png)

*On the naive metric (left) every graph looks good, co-expression included. On the de-confounded metric (right) the three co-expression graphs drop below zero while every causal and Super graph stays strongly positive.*

## The substrate decides regulator-specificity

At matched edge count and in distribution, the composition of the graph decides whether it carries regulator-specific signal. Every causal and Super graph is strongly positive on RPE1 (+0.42 to +0.54) and every co-expression graph is negative (-0.05 to -0.09). The FUNGI Super graph scores +0.472, against +0.198 for a ridge-regression baseline, the simple linear model that recent deep perturbation predictors have repeatedly failed to beat [11]. The ordering replicates on h1-hESC, where the Super graphs top the table with the gap compressed by the sparser coverage.

![Specificity against direction agreement, RPE1, all densities](figures/super_master_scatter.png)

*Every graph, every seed and all three edge densities on RPE1. The x-axis is regulator-specificity and the y-axis is agreement with the direction of the true response. Super graphs (orange) sit top-right, causal graphs (red) just below, and co-expression graphs (purple) bottom-left, and the three groups stay separated at every density.*

![Substrate by pruner heatmap](figures/substrate_heatmap.png)

*Mean regulator-specificity for every pruner and substrate on both cell lines. Columns (substrate) separate cleanly. Rows (pruner) are close.*

## The signal lives in the directed wiring

Control graphs show the advantage comes from regulatory structure. Scrambling the targets while preserving every gene's degree collapses the advantage toward the co-expression floor, and so does reversing every edge. Replacing every edge weight with 1 leaves it intact. The signal therefore sits in which regulator points at which target, and in which direction, rather than in degree or in fine weights.

![Mechanism controls](figures/mechanism_controls.png)

## FUNGI produces a legible graph at predictive parity

On prediction, the choice of pruner matters far less than the choice of substrate. At matched edge count FUNGI ties or slightly trails Top-Weight and KNN, with no pruner winning consistently, and that parity survives a doubled-capacity RHIZO and a full architecture sweep, so it reflects the graphs and not an under-powered model. What separates the pruners is the shape of the graph they produce. FUNGI is the only one of the three whose Super graphs keep both a hub hierarchy (out-degree Gini 0.72 on RPE1, 0.74 on h1-hESC) and low reciprocity (0.12 and 0.09) on both cell lines. Top-Weight over-reciprocates (0.14 and 0.21), and KNN flattens the hub hierarchy on RPE1 (out-degree Gini 0.57). FUNGI delivers a graph with the hubs and directionality of a real regulatory network that predicts as well as a structure-blind prune of the same parent.

FUNGI's full objective asks for more than these two signatures. It scores six literature targets at once, and on the Super graph it does not yet satisfy all six together. The best coverage-preserving Super graph so far lands five of six in band, and closing the last gap is the main active development target.

![FUNGI in the literature zone](figures/fungi_literature_zone.png)

## Post-thesis: a coverage-preserving Super graph doubles zero-shot specificity

Work after the thesis targeted the hardest regime, held-out perturbations the graph never saw. The thesis Super graph pruned each input first and fused the pruned graphs, which left a third of the held-out perturbations without an out-edge. Fusing the dense graphs first with the rank-max operator and then pruning with FUNGI preserves that coverage. On the 373 held-out RPE1 perturbations, the coverage-preserving Super graph reaches a `pearson_systema` of 0.089 against 0.042 for the earlier construction (5 seeds), and it covers all 373 held-out perturbations against 247. An independent re-run on a laptop GPU reproduced the result at 0.099 (3 seeds).

![Super graph versus incumbent at zero-shot](figures/super_vs_incumbent_zeroshot.png)

Reversing every edge of that graph drives its score negative (-0.054 against +0.098), which confirms the directed wiring carries the signal. The raw correlation metric rates the reversed graph higher than the real one (0.454 against 0.246), the clearest demonstration of why the de-confounded axis is the decision axis.

![Reverse-null gate](figures/reverse_null_gate.png)

---

## Boundaries

A causal edge exists only for a perturbed regulator, so at held-out perturbations a purely causal graph has nothing to propagate, and in that regime every method in this work converges toward the linear-baseline ceiling that the field currently reports [11]. Added coverage widens what a graph can predict without sharpening regulator-specificity on its own, which is why every comparison here is read on the de-confounded axis. The coverage-preserving Super graph is the current route into the zero-shot regime, and the next step is a causal graft that gives held-out genes their own directed out-edges.

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
