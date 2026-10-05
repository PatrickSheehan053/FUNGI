"""
One-off at-scale benchmark on SYNTHETIC data sized to match real config
derivation (n_genes=5000, lambda_max=55 -> Ne~550K). Not part of the
deliverable -- fungi_config.yaml's graph_path is still a TODO placeholder,
so this is the closest available stand-in until real data is wired in.
"""
import time
import numpy as np
import torch
import engine
import search

rng = np.random.default_rng(99)
n_genes = 5000
n_edges_full = 2_500_000  # ~target_density=0.10 prefilter scale

print(f"Generating synthetic graph: {n_genes} genes, {n_edges_full:,} edges...")
t0 = time.perf_counter()
all_pairs = rng.choice(n_genes * n_genes, size=n_edges_full, replace=False)
sources_full = (all_pairs // n_genes).astype(np.int64)
targets_full = (all_pairs % n_genes).astype(np.int64)
W_full = rng.uniform(0.01, 1.0, n_edges_full).astype(np.float64)
order0 = np.argsort(W_full)[::-1]
W_full, sources_full, targets_full = W_full[order0], sources_full[order0], targets_full[order0]
W_q_full = rng.uniform(0.05, 1.0, n_edges_full).astype(np.float64)
D_full = np.zeros(n_edges_full, dtype=np.float64)
print(f"  done in {time.perf_counter()-t0:.1f}s")

source_pert_impact = rng.uniform(0.5, 3.0, n_genes).astype(np.float64)
rdf_prior = rng.uniform(0.1, 2.0, n_genes).astype(np.float64)
chi_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)
chi_t_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)
rho_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)
perturbed_nodes = rng.choice(n_genes, size=300, replace=False)
per_gene_kappa = np.ones(n_genes)

utopian_bounds = {
    "alpha": [2.0, 3.0], "gini": [0.3, 0.6], "gini_in": [0.3, 0.6],
    "S_max": [0.05, 0.15], "rho": [-0.3, 0.1], "C": [0.05, 0.3],
}
loss_weights = {"alpha": 1.0, "gini": 1.0, "gini_in": 1.0, "S_max": 1.0,
                "rho": 1.0, "C": 1.0, "Q": 1.0}
shatter_cfg = {
    "max_edge_count": int(55.0 * n_genes),  # lambda_max=55, matches real config
    "max_orphan_fraction": 0.7, "min_gwcc_fraction": 0.1, "min_clustering": None,
}
k_core_bounds = (8.0, 28.0)  # matches real hyperparameter_bounds.k_core
kernel_flags = {"weight": True, "ffl": True, "pert_impact": True, "rdf": True,
                "scber": True, "chi_s": True, "chi_t": True, "rho": True}

print("\nInitializing GPU context...")
t0 = time.perf_counter()
ctx = engine.init_gpu_context(
    W_full, W_q_full, sources_full, targets_full, n_genes, shatter_cfg,
    source_pert_impact, k_core_bounds, rdf_prior=rdf_prior, chi_prior=chi_prior,
    chi_t_prior=chi_t_prior, rho_prior=rho_prior, kernel_flags=kernel_flags,
    n_ffl_buckets=8)
print(f"  init_gpu_context: {time.perf_counter()-t0:.2f}s")
assert ctx.enabled
print(f"  Ne={ctx.Ne:,}  max_seg_len={ctx.max_seg_len}")

B = search._choose_gpu_batch_size(ctx, vram_budget_gb=5.0)
print(f"  chosen batch size B={B}")

evaluator = search.SearchEvaluator(
    W_arr=W_full, W_q_arr=W_q_full, D_arr=D_full,
    sources_arr=sources_full, targets_arr=targets_full,
    n_genes=n_genes, perturbed_nodes=perturbed_nodes,
    utopian_bounds=utopian_bounds, loss_weights=loss_weights,
    shatter_cfg=shatter_cfg, per_gene_kappa=per_gene_kappa,
    source_pert_impact=source_pert_impact, mode='organic')

n_test = 256  # representative slice of the real 4096-point Sobol search
hp_cfg = {"beta": [0.5, 4.0], "delta": [0.0, 3.0], "kappa": [0.02, 0.35],
          "k_core": list(k_core_bounds), "lambda_density": [0.006, 0.010],
          "psi": [0.0, 3.0], "nu": [0.0, 2.0]}
sobol_params, lower, upper = search.generate_sobol_samples(
    n_genes=n_genes, n_samples=n_test, hp_cfg=hp_cfg, seed=42)

print(f"\nRunning {n_test} evaluations through evaluator.evaluate() (GPU path)...")
torch.cuda.synchronize()
t0 = time.perf_counter()
df = evaluator.evaluate(list(sobol_params), desc="GPU benchmark")
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
print(f"\n{n_test} evals in {elapsed:.2f}s -> {elapsed/n_test*1000:.2f} ms/eval")
print(f"Peak VRAM during run: {peak_vram_gb:.2f} GB")
print(f"Extrapolated to 4096-point Sobol search: {elapsed/n_test*4096:.1f}s "
      f"({elapsed/n_test*4096/60:.2f} min)")
print(f"n_viable: {int((df['is_shattered']==0).sum())}/{len(df)}")
print(f"Result columns: {list(df.columns)}")
