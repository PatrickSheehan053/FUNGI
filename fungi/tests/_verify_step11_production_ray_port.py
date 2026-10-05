"""One-off verification: the Ray pipeline ported from FUNGI/ray_dev/ into
production engine.py/search.py. Confirms (1) _evaluate_gpu_ray works
correctly from its new home in src/, (2) evaluate()'s new routing logic
actually goes through Ray when use_ray_pipeline=True and matches
_evaluate_gpu exactly, (3) use_ray_pipeline=False correctly forces the
single-process path. Small scale -- this is a routing/correctness check,
not a timing benchmark (see ray_dev/ for the real benchmark)."""
import numpy as np
import torch
import engine
import search

rng = np.random.default_rng(151)
n_genes = 800
n_edges_full = 90_000

all_pairs = rng.choice(n_genes * n_genes, size=n_edges_full, replace=False)
sources_full = (all_pairs // n_genes).astype(np.int64)
targets_full = (all_pairs % n_genes).astype(np.int64)
W_full = rng.uniform(0.01, 1.0, n_edges_full).astype(np.float64)
order0 = np.argsort(W_full)[::-1]
W_full, sources_full, targets_full = W_full[order0], sources_full[order0], targets_full[order0]
W_q_full = rng.uniform(0.05, 1.0, n_edges_full).astype(np.float64)
D_full = np.zeros(n_edges_full, dtype=np.float64)

source_pert_impact = rng.uniform(0.5, 3.0, n_genes).astype(np.float64)
chi_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)
chi_t_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)
rho_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)
perturbed_nodes = rng.choice(n_genes, size=120, replace=False)
per_gene_kappa = np.ones(n_genes)

utopian_bounds = {
    "alpha": [2.0, 3.0], "gini": [0.3, 0.6], "gini_in": [0.3, 0.6],
    "S_max": [0.05, 0.15], "rho": [-0.3, 0.1], "C": [0.05, 0.3],
}
loss_weights = {"alpha": 1.0, "gini": 1.0, "gini_in": 1.0, "S_max": 1.0,
                "rho": 1.0, "C": 1.0, "Q": 1.0}
shatter_cfg = {
    "max_edge_count": int(55.0 * n_genes),
    "max_orphan_fraction": 0.7, "min_gwcc_fraction": 0.1, "min_clustering": None,
}
k_core_bounds = (8.0, 28.0)
kernel_flags = {"weight": True, "ffl": True, "pert_impact": True, "rdf": True,
                "scber": True, "chi_s": True, "chi_t": True, "rho": True}

print("Initializing GPU context...")
ctx = engine.init_gpu_context(
    W_full, W_q_full, sources_full, targets_full, n_genes, shatter_cfg,
    source_pert_impact, k_core_bounds, rdf_prior=None, chi_prior=chi_prior,
    chi_t_prior=chi_t_prior, rho_prior=rho_prior, kernel_flags=kernel_flags,
    n_ffl_buckets=8)
assert ctx.enabled
print(f"  Ne={ctx.Ne:,}  max_seg_len={ctx.max_seg_len}")

hp_cfg = {"beta": [0.5, 4.0], "delta": [0.0, 3.0], "kappa": [0.02, 0.35],
          "k_core": list(k_core_bounds), "lambda_density": [0.006, 0.010],
          "psi": [0.0, 3.0], "nu": [0.0, 2.0]}
sobol_params, lower, upper = search.generate_sobol_samples(
    n_genes=n_genes, n_samples=40, hp_cfg=hp_cfg, seed=23)
param_list = list(sobol_params)

# ---- Test 1: use_ray_pipeline=True -> evaluate() should route through Ray ----
ev_ray = search.SearchEvaluator(
    W_arr=W_full, W_q_arr=W_q_full, D_arr=D_full,
    sources_arr=sources_full, targets_arr=targets_full,
    n_genes=n_genes, perturbed_nodes=perturbed_nodes,
    utopian_bounds=utopian_bounds, loss_weights=loss_weights,
    shatter_cfg=shatter_cfg, per_gene_kappa=per_gene_kappa,
    source_pert_impact=source_pert_impact, mode='organic',
    n_workers=4, use_ray_pipeline=True)

print("\nCalling evaluate() with use_ray_pipeline=True...")
df_via_router_ray = ev_ray.evaluate(param_list, chunk_size=10, show_progress=False)
assert ev_ray._ray is not None, "evaluate() did not initialize Ray -- routing failed silently!"
print(f"  Ray was initialized: {ev_ray._ray.is_initialized()}")
print(f"  {len(df_via_router_ray)} results returned")

# ---- Test 2: use_ray_pipeline=False -> evaluate() should NOT touch Ray ----
ev_single = search.SearchEvaluator(
    W_arr=W_full, W_q_arr=W_q_full, D_arr=D_full,
    sources_arr=sources_full, targets_arr=targets_full,
    n_genes=n_genes, perturbed_nodes=perturbed_nodes,
    utopian_bounds=utopian_bounds, loss_weights=loss_weights,
    shatter_cfg=shatter_cfg, per_gene_kappa=per_gene_kappa,
    source_pert_impact=source_pert_impact, mode='organic',
    n_workers=4, use_ray_pipeline=False)

print("\nCalling evaluate() with use_ray_pipeline=False...")
df_via_router_single = ev_single.evaluate(param_list, chunk_size=10, show_progress=False)
assert ev_single._ray is None, "evaluate() touched Ray even though use_ray_pipeline=False!"
print(f"  Ray left uninitialized (correct): {ev_single._ray is None}")
print(f"  {len(df_via_router_single)} results returned")

# ---- Compare: both routing paths must produce identical results ----
cols_exact = ['is_shattered', 'n_edges']
cols_float = ['utopia_loss', 'alpha', 'Gini', 'rho', 'C', 'S_max']
mismatches = 0
max_diffs = {}
for i in range(len(param_list)):
    a, b = df_via_router_single.iloc[i], df_via_router_ray.iloc[i]
    for c in cols_exact:
        if int(a[c]) != int(b[c]):
            mismatches += 1
            print(f"  row {i}: {c} MISMATCH single={a[c]} ray={b[c]}")
    for c in cols_float:
        max_diffs[c] = max(max_diffs.get(c, 0.0), abs(float(a[c]) - float(b[c])))

print(f"\n{len(param_list) - mismatches}/{len(param_list)} rows exact-matched on is_shattered/n_edges")
print("max abs diff per metric:", max_diffs)
assert mismatches == 0
assert max(max_diffs.values()) < 1e-9
print("\nPASS: production evaluate() routing (both Ray and single-process) verified correct.")

engine.release_gpu_context()
try:
    ev_ray._ray.shutdown()
except Exception:
    pass
print("Released GPU context and shut down Ray.")
