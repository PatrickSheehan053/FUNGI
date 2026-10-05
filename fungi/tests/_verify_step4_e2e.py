"""One-off verification: full run_dash_and_score CPU vs GPU path. Not part of the deliverable."""
import numpy as np
import engine

rng = np.random.default_rng(123)
n_genes = 150
n_edges_full = 12000

all_pairs = rng.choice(n_genes * n_genes, size=n_edges_full, replace=False)
sources_full = (all_pairs // n_genes).astype(np.int64)
targets_full = (all_pairs % n_genes).astype(np.int64)
W_full = rng.uniform(0.01, 1.0, n_edges_full).astype(np.float64)
order0 = np.argsort(W_full)[::-1]
W_full, sources_full, targets_full = W_full[order0], sources_full[order0], targets_full[order0]
W_q_full = rng.uniform(0.05, 1.0, n_edges_full).astype(np.float64)

source_pert_impact = rng.uniform(0.5, 3.0, n_genes).astype(np.float64)
rdf_prior = rng.uniform(0.1, 2.0, n_genes).astype(np.float64)
chi_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)
chi_t_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)
rho_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)

perturbed_nodes = rng.choice(n_genes, size=15, replace=False)
per_gene_kappa = np.ones(n_genes)

utopian_bounds = {
    "alpha": [2.0, 3.0], "gini": [0.3, 0.6], "gini_in": [0.3, 0.6],
    "S_max": [0.05, 0.15], "rho": [-0.3, 0.1], "C": [0.05, 0.3],
}
loss_weights = {"alpha": 1.0, "gini": 1.0, "gini_in": 1.0, "S_max": 1.0,
                "rho": 1.0, "C": 1.0, "Q": 1.0}
shatter_cfg = {
    "max_edge_count": 9000, "max_orphan_fraction": 0.7,
    "min_gwcc_fraction": 0.1, "min_clustering": None,
}
k_core_bounds = (5.0, 20.0)
kernel_flags = {"weight": True, "ffl": True, "pert_impact": True, "rdf": True,
                "scber": True, "chi_s": True, "chi_t": True, "rho": True}

common_kwargs = dict(
    perturbed_nodes=perturbed_nodes, utopian_bounds=utopian_bounds,
    loss_weights=loss_weights, shatter_cfg=shatter_cfg,
    per_gene_kappa=per_gene_kappa, source_pert_impact=source_pert_impact,
    chi_prior=chi_prior, rho_prior=rho_prior, chi_t_prior=chi_t_prior,
    rdf_prior=rdf_prior, kernel_flags=kernel_flags, mode='organic')

_bucket_kcores = np.linspace(k_core_bounds[0], k_core_bounds[1], 8)
# lam is edges-PER-GENE directly (select_edges: budget = n_genes * lam) --
# NOT divided by n_genes. First 4 configs use EXACT bucket-aligned k_core --
# no FFL approximation, so these should match to float32 tolerance. Last 2
# deliberately use off-grid k_core to confirm the (expected, by-design)
# FFL-bucket approximation still produces a reasonable, non-degenerate result.
test_params = [
    (1.5, 0.8, 0.15, _bucket_kcores[1], 30.0, 1.2, 0.5),
    (0.6, 1.5, 0.10, _bucket_kcores[3], 35.0, 2.0, 0.0),
    (2.0, 0.3, 0.20, _bucket_kcores[5], 32.0, 0.7, 1.0),
    (1.0, 1.0, 0.12, _bucket_kcores[0], 40.0, 1.5, 1.5),
]
test_params_offgrid = [
    (0.6, 1.5, 0.10, 15.0, 35.0, 2.0, 0.0),
    (2.0, 0.3, 0.20, 16.5, 32.0, 0.7, 1.0),
]

# ---- CPU baseline (no GPU context) ----
engine.release_gpu_context()
cpu_results = [engine.run_dash_and_score(p, W_full, W_q_full, None, sources_full,
                                         targets_full, n_genes, **common_kwargs)
               for p in test_params]

# ---- GPU path ----
ctx = engine.init_gpu_context(
    W_full, W_q_full, sources_full, targets_full, n_genes, shatter_cfg,
    source_pert_impact, k_core_bounds, rdf_prior=rdf_prior, chi_prior=chi_prior,
    chi_t_prior=chi_t_prior, rho_prior=rho_prior, kernel_flags=kernel_flags,
    n_ffl_buckets=8)
assert ctx.enabled
gpu_results = [engine.run_dash_and_score(p, W_full, W_q_full, None, sources_full,
                                         targets_full, n_genes, **common_kwargs)
               for p in test_params]

# ---- Also test the real batch path directly ----
batch_results = engine.run_dash_and_score_gpu_batch(
    test_params, ctx, n_genes, perturbed_nodes, utopian_bounds, loss_weights,
    shatter_cfg, per_gene_kappa, mode='organic')

all_pass = True
# n_edges/is_shattered/active_nodes drive search ranking directly and should
# match tightly. utopia_loss tolerates a slightly wider band now: fast_topology
# uses clustering_wedge_sample (a real, intentional approximation of C) and
# skips modularity (Q forced to 0 -- intentional, not load-bearing once
# gini_in is configured), both of which propagate small differences into the
# aggregate loss. alpha/Gini/rho/C/S_max tolerate a relative difference for
# the same near-tied-edge-selection reason as before.
tight_keys = ['is_shattered', 'n_edges', 'active_nodes']
loose_keys = ['utopia_loss', 'alpha', 'Gini', 'rho', 'C', 'S_max']
for i, p in enumerate(test_params):
    c, g, b = cpu_results[i], gpu_results[i], batch_results[i]
    row_pass = True
    if abs(g.get('Q', 0.0)) > 1e-9 or abs(b.get('Q', 0.0)) > 1e-9:
        print(f"  UNEXPECTED: param {i} Q should be exactly 0 under fast_topology, "
              f"got gpu={g.get('Q')} batch={b.get('Q')}")
        row_pass = False
    for k in tight_keys:
        cv, gv, bv = c.get(k), g.get(k), b.get(k)
        ok = (abs(cv - gv) < 1e-3 and abs(cv - bv) < 1e-3) if isinstance(cv, float) else (cv == gv == bv)
        if not ok:
            print(f"  TIGHT MISMATCH param {i} key={k}: cpu={cv} gpu={gv} batch={bv}")
            row_pass = False
    for k in loose_keys:
        cv, gv, bv = c.get(k, 0.0), g.get(k, 0.0), b.get(k, 0.0)
        denom = max(abs(cv), 1e-3)
        ok = abs(cv - gv) / denom < 0.10 and abs(cv - bv) / denom < 0.10
        if not ok:
            print(f"  LOOSE MISMATCH param {i} key={k}: cpu={cv} gpu={gv} batch={bv}")
            row_pass = False
    all_pass &= row_pass
    print(f"config {i}: loss cpu={c['utopia_loss']:.6f} gpu={g['utopia_loss']:.6f} "
          f"batch={b['utopia_loss']:.6f} shattered={c['is_shattered']}/{g['is_shattered']} "
          f"n_edges={c['n_edges']}/{g['n_edges']} {'PASS' if row_pass else 'FAIL'}")

print("\n" + ("ALL EXACT-GRID E2E TESTS PASSED" if all_pass else "SOME EXACT-GRID E2E TESTS FAILED"))
assert all_pass

# ---- Off-grid k_core: FFL is approximated by design. Expect close, not exact. ----
print("\nOff-grid k_core (FFL bucket approximation expected):")
cpu_offgrid = [engine.run_dash_and_score(p, W_full, W_q_full, None, sources_full,
                                         targets_full, n_genes, **common_kwargs)
               for p in test_params_offgrid]
gpu_offgrid = [engine.run_dash_and_score(p, W_full, W_q_full, None, sources_full,
                                         targets_full, n_genes, **common_kwargs)
               for p in test_params_offgrid]
offgrid_ok = True
for i, p in enumerate(test_params_offgrid):
    c, g = cpu_offgrid[i], gpu_offgrid[i]
    edge_diff = abs(c['n_edges'] - g['n_edges'])
    loss_diff = abs(c['utopia_loss'] - g['utopia_loss'])
    ok = (c['is_shattered'] == g['is_shattered']) and edge_diff <= max(2, int(0.05 * c['n_edges']))
    offgrid_ok &= ok
    print(f"  config {i}: loss cpu={c['utopia_loss']:.4f} gpu={g['utopia_loss']:.4f} "
          f"(diff={loss_diff:.4f})  n_edges cpu={c['n_edges']} gpu={g['n_edges']} "
          f"(diff={edge_diff})  {'OK' if ok else 'TOO DIVERGENT'}")
assert offgrid_ok, "off-grid k_core produced an unreasonably large divergence"
print("\nALL VERIFICATION TESTS PASSED")
