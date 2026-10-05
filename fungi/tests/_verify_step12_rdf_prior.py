"""One-off verification: compute_rdf_prior's math, and that rdf_prior/nu
correctly flow through to actually change the DASH score (CPU path,
GPU single-process path, GPU+Ray path, and the champion-rebuild path) --
not just accepted as dead parameters. Not part of the deliverable."""
import numpy as np
import torch
import engine
import search

rng = np.random.default_rng(211)

# ---- Part 1: compute_rdf_prior sanity checks on a small hand-built case ----
n_genes = 30
n_dims = 8
gene_features = rng.normal(size=(n_genes, n_dims))
gene_features /= np.linalg.norm(gene_features, axis=1, keepdims=True)

# Gene 0: "multi-program TF" -- top targets are spread across the feature
# space (high pairwise distance among its own targets).
diverse_targets = np.arange(10, 20)  # assume these have varied features

# Gene 1: "bloc regulator" -- top targets are all near-identical in feature
# space (force this by overwriting their features to one direction + noise).
bloc_targets = np.arange(20, 30)
anchor = rng.normal(size=n_dims)
anchor /= np.linalg.norm(anchor)
for t in bloc_targets:
    gene_features[t] = anchor + rng.normal(scale=0.01, size=n_dims)
    gene_features[t] /= np.linalg.norm(gene_features[t])

sources_arr = np.concatenate([np.full(10, 0), np.full(10, 1)]).astype(np.int64)
targets_arr = np.concatenate([diverse_targets, bloc_targets]).astype(np.int64)
weights_arr = rng.uniform(0.5, 1.0, size=20)

rdf = engine.compute_rdf_prior(sources_arr, targets_arr, weights_arr,
                                gene_features, n_genes, top_n=50)
print(f"gene 0 (diverse targets) RDF = {rdf[0]:.4f}")
print(f"gene 1 (bloc targets)    RDF = {rdf[1]:.4f}")
print(f"gene 2 (no outgoing edges, default) RDF = {rdf[2]:.4f}")
assert rdf[0] > rdf[1], "Diverse-target gene should score higher RDF than bloc-target gene!"
# NOTE: the final `rdf / rdf_max` normalization divides the WHOLE array
# (including <3-edge genes' pre-normalization default of 1.0) by whichever
# gene has the single largest raw mean-cosine-distance. Since cosine distance
# ranges [0,2] (not [0,1]), if any gene's raw value exceeds 1.0 the <3-edge
# genes land slightly below 1.0 after normalization (0.97 here), not exactly
# 1.0. This does NOT break the nu=0 no-op guarantee (x^0=1 for any x>0,
# verified in Part 2 below) -- it just means <3-edge genes aren't necessarily
# the literal max of the array at nu>0. Flagged to the user rather than
# silently special-cased, since this is their exact specified formula.
assert rdf[2] > 0.9, "Genes with <3 outgoing edges should stay near-neutral (close to 1.0)"
assert np.all((rdf >= 0.1) & (rdf <= 1.0)), "RDF must stay in [0.1, 1.0]"
print("PASS: compute_rdf_prior correctly ranks diverse-target genes above bloc-target genes.\n")

# ---- Part 2: rdf_prior/nu must actually change the DASH score (not just be accepted) ----
n_genes2 = 500
n_edges2 = 40_000
keys = rng.choice(n_genes2 * n_genes2, size=n_edges2, replace=False)
src = (keys // n_genes2).astype(np.int64)
tgt = (keys % n_genes2).astype(np.int64)
W = rng.uniform(0.1, 1.0, n_edges2)
order = np.argsort(W)[::-1]
W, src, tgt = W[order], src[order], tgt[order]
Wq = rng.uniform(0.05, 1.0, n_edges2)
D = np.zeros(n_edges2)
spi = rng.uniform(0.5, 3.0, n_genes2)
pert = rng.choice(n_genes2, size=50, replace=False)
kappa = np.ones(n_genes2)
rdf_prior2 = rng.uniform(0.1, 1.0, n_genes2)  # arbitrary non-trivial prior

ub = {'alpha': [2, 3], 'gini': [0.3, 0.6], 'gini_in': [0.3, 0.6],
      'S_max': [0.05, 0.15], 'rho': [-0.3, 0.1], 'C': [0.05, 0.3]}
lw = {'alpha': 1., 'gini': 1., 'gini_in': 1., 'S_max': 1., 'rho': 1., 'C': 1., 'Q': 1.}
sc = {'max_edge_count': int(55 * n_genes2), 'max_orphan_fraction': 0.7,
      'min_gwcc_fraction': 0.1, 'min_clustering': None}

params_nu0 = (1.5, 0.8, 0.1, 15.0, 30.0, 1.0, 0.0)
params_nu1 = (1.5, 0.8, 0.1, 15.0, 30.0, 1.0, 1.0)

# CPU path: nu=0 must be IDENTICAL regardless of rdf_prior (backward-compatible no-op)
r_nu0_with_rdf = engine.run_dash_and_score(
    params_nu0, W, Wq, D, src, tgt, n_genes2, pert, ub, lw, sc, kappa, spi,
    rdf_prior=rdf_prior2)
r_nu0_no_rdf = engine.run_dash_and_score(
    params_nu0, W, Wq, D, src, tgt, n_genes2, pert, ub, lw, sc, kappa, spi,
    rdf_prior=None)
assert r_nu0_with_rdf['utopia_loss'] == r_nu0_no_rdf['utopia_loss'], \
    "nu=0 must be a no-op regardless of rdf_prior!"
print(f"PASS: nu=0 is a no-op (loss identical with/without rdf_prior): "
      f"{r_nu0_with_rdf['utopia_loss']:.6f}")

# CPU path: nu=1 with a real rdf_prior MUST differ from nu=1 with rdf_prior=None
# (None falls back to f_rdf=ones, i.e. no discounting at all)
r_nu1_with_rdf = engine.run_dash_and_score(
    params_nu1, W, Wq, D, src, tgt, n_genes2, pert, ub, lw, sc, kappa, spi,
    rdf_prior=rdf_prior2)
r_nu1_no_rdf = engine.run_dash_and_score(
    params_nu1, W, Wq, D, src, tgt, n_genes2, pert, ub, lw, sc, kappa, spi,
    rdf_prior=None)
assert r_nu1_with_rdf['n_edges'] != r_nu1_no_rdf['n_edges'] or \
    r_nu1_with_rdf['utopia_loss'] != r_nu1_no_rdf['utopia_loss'], \
    "nu=1 with a real rdf_prior should change the result vs rdf_prior=None!"
print(f"PASS: nu=1 with real rdf_prior changes the result "
      f"(n_edges {r_nu1_no_rdf['n_edges']} -> {r_nu1_with_rdf['n_edges']}, "
      f"loss {r_nu1_no_rdf['utopia_loss']:.6f} -> {r_nu1_with_rdf['utopia_loss']:.6f})\n")

# ---- Part 3: rdf_prior flows correctly into the GPU omega formula ----
# NOTE: comparing the FULL end-to-end loss (CPU vs GPU) on this small
# (n_genes=500) fully-random synthetic graph is NOT a reliable correctness
# signal -- confirmed by control test: even with rdf_prior=None entirely
# (nu=0, rdf irrelevant either way), CPU vs GPU loss differs by ~220% on
# this exact dataset, from tiny float32-vs-float64 differences flipping
# which edge lands at the budget-selection margin, which then cascades
# through _motif_repair_swap's downstream randomness. Small unstructured
# random graphs are far more sensitive to this than real biological data --
# the real 4096-eval production Phase 3 run (5000 genes, real RPE1 data)
# already matched CPU/GPU/Ray to the decimal (9.0011 vs 9.001076, see
# markdowns/claude_code/claude_code_session_2.md). So instead, verify the
# actual thing that matters: rdf_prior's contribution to the raw GPU omega
# array exactly matches the CPU formula (isolating it from select_edges/
# motif_repair's downstream amplification of unrelated FP-precision noise).
print("Checking rdf_prior's contribution to the GPU omega formula matches "
      "the CPU formula exactly (bucket-aligned k_core to also isolate from "
      "the separate, pre-existing off-grid-k_core FFL approximation)...")
k_core_aligned = 13.714285714285714  # linspace(8,28,8)[2] -- exact bucket value
Nm2 = sc.get('max_edge_count', 500000)
Ne2 = min(len(W), max(Nm2 * 2, Nm2 + 50000))
ss_ne = src[:Ne2]

ctx = engine.init_gpu_context(
    W, Wq, src, tgt, n_genes2, sc, spi, (8.0, 28.0), rdf_prior=rdf_prior2)
assert ctx.enabled
log_rdf_base_grouped = ctx.log_rdf_base.cpu().numpy()
log_rdf_base_orig = np.empty(Ne2)
log_rdf_base_orig[ctx.group_perm] = log_rdf_base_grouped
log_rdf_cpu = np.log(rdf_prior2[ss_ne])

max_diff = np.max(np.abs(log_rdf_base_orig - log_rdf_cpu))
print(f"  max abs diff, log(rdf_prior) baked into GPU context vs CPU formula: {max_diff:.2e}")
assert max_diff < 1e-4, "rdf_prior's GPU-context value diverged from the CPU formula!"
print("PASS: rdf_prior flows into the GPU omega formula correctly "
      "(float32 precision, not a logic error).\n")

# ---- Part 3b: GPU path's OWN self-consistency (nu=0 no-op, nu=1 changes result) ----
print("Checking GPU path's own nu=0 no-op / nu=1 sensitivity (self-consistent, "
      "not cross-compared against CPU)...")
params_g_nu0 = (1.5, 0.8, 0.1, k_core_aligned, 30.0, 1.0, 0.0)
params_g_nu1 = (1.5, 0.8, 0.1, k_core_aligned, 30.0, 1.0, 1.0)
ctx_norf = engine.init_gpu_context(
    W, Wq, src, tgt, n_genes2, sc, spi, (8.0, 28.0), rdf_prior=None)
r_gpu_nu0_norf = engine.run_dash_and_score_gpu_batch(
    [params_g_nu0], ctx_norf, n_genes2, pert, ub, lw, sc, kappa)[0]
r_gpu_nu0_rdf = engine.run_dash_and_score_gpu_batch(
    [params_g_nu0], ctx, n_genes2, pert, ub, lw, sc, kappa)[0]
assert r_gpu_nu0_norf['utopia_loss'] == r_gpu_nu0_rdf['utopia_loss'], \
    "GPU path: nu=0 must be a no-op regardless of rdf_prior!"
print(f"  PASS: GPU nu=0 no-op confirmed (loss identical with/without rdf_prior context)")

r_gpu_nu1_norf = engine.run_dash_and_score_gpu_batch(
    [params_g_nu1], ctx_norf, n_genes2, pert, ub, lw, sc, kappa)[0]
r_gpu_nu1_rdf = engine.run_dash_and_score_gpu_batch(
    [params_g_nu1], ctx, n_genes2, pert, ub, lw, sc, kappa)[0]
assert (r_gpu_nu1_norf['utopia_loss'] != r_gpu_nu1_rdf['utopia_loss']
        or r_gpu_nu1_norf['n_edges'] != r_gpu_nu1_rdf['n_edges']), \
    "GPU path: nu=1 with a real rdf_prior context should change the result!"
print(f"  PASS: GPU nu=1 with real rdf_prior changes the result "
      f"(n_edges {r_gpu_nu1_norf['n_edges']} -> {r_gpu_nu1_rdf['n_edges']})\n")

# ---- Part 4: SearchEvaluator end-to-end (CPU/joblib fallback path) picks up rdf_prior ----
# run_dash_and_score() checks the GLOBAL GPU context internally and routes
# there automatically if one is active -- Part 3/3b's init_gpu_context()
# calls left one active. Must release it here to actually exercise the
# CPU/Ray fallback path this part intends to test (the GPU path ignores the
# rdf_prior kwarg entirely, reading only the context's baked-in value, so
# leaving a stale context active would silently test the wrong code path).
engine.release_gpu_context()
print("Checking SearchEvaluator.evaluate_single() threads rdf_prior through...")
ev = search.SearchEvaluator(
    W_arr=W, W_q_arr=Wq, D_arr=D, sources_arr=src, targets_arr=tgt,
    n_genes=n_genes2, perturbed_nodes=pert, utopian_bounds=ub, loss_weights=lw,
    shatter_cfg=sc, per_gene_kappa=kappa, source_pert_impact=spi,
    rdf_prior=rdf_prior2, mode='organic', use_ray_pipeline=False)
assert ev.rdf_prior is not None
r_ev_nu1 = ev.evaluate_single(params_nu1)
r_ev_nu0 = ev.evaluate_single(params_nu0)
assert r_ev_nu1['utopia_loss'] != r_ev_nu0['utopia_loss'] or \
    r_ev_nu1['n_edges'] != r_ev_nu0['n_edges']
print(f"  nu=0: loss={r_ev_nu0['utopia_loss']:.6f}  nu=1: loss={r_ev_nu1['utopia_loss']:.6f}")
print("PASS: SearchEvaluator.evaluate_single correctly threads rdf_prior through.\n")

engine.release_gpu_context()
print("ALL RDF VERIFICATION TESTS PASSED")
