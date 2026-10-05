"""One-off verification: GPU segmented argsort vs np.lexsort. Not part of the deliverable."""
import numpy as np
import torch
import engine

rng = np.random.default_rng(7)
n_genes = 80
n_edges_full = 6000

all_pairs = rng.choice(n_genes * n_genes, size=n_edges_full, replace=False)
sources_full = (all_pairs // n_genes).astype(np.int64)
targets_full = (all_pairs % n_genes).astype(np.int64)
W_full = rng.uniform(0.01, 1.0, n_edges_full).astype(np.float64)
order0 = np.argsort(W_full)[::-1]
W_full, sources_full, targets_full = W_full[order0], sources_full[order0], targets_full[order0]
W_q_full = rng.uniform(0.05, 1.0, n_edges_full).astype(np.float64)

source_pert_impact = rng.uniform(0.5, 3.0, n_genes).astype(np.float64)
rdf_prior = rng.uniform(0.1, 2.0, n_genes).astype(np.float64)

shatter_cfg = {"max_edge_count": 400}  # Ne = min(6000, max(800,450)) = 800
k_core_bounds = (5.0, 20.0)
kernel_flags = {"weight": True, "ffl": True, "pert_impact": True, "rdf": True,
                "scber": False, "chi_s": False, "chi_t": False, "rho": False}

ctx = engine.init_gpu_context(
    W_full, W_q_full, sources_full, targets_full, n_genes, shatter_cfg,
    source_pert_impact, k_core_bounds, rdf_prior=rdf_prior,
    kernel_flags=kernel_flags, n_ffl_buckets=8)
Ne = ctx.Ne
print(f"Ne={Ne}")

bucket_kcores = ctx.k_core_bucket_values
test_configs = [
    dict(beta=1.5, delta=0.8, psi=1.2, nu=0.5, k_core=bucket_kcores[1]),
    dict(beta=0.6, delta=1.5, psi=2.0, nu=0.0, k_core=bucket_kcores[5]),
]

dev = torch.device("cuda")
for cfg in test_configs:
    beta, delta, psi, nu, k_core = cfg["beta"], cfg["delta"], cfg["psi"], cfg["nu"], cfg["k_core"]

    # ---- CPU reference ----
    T_local = engine.compute_dynamic_topology(W_full, sources_full, targets_full, k_core, n_genes)
    Ws, Wqs = W_full[:Ne], W_q_full[:Ne]
    ss, ts = sources_full[:Ne], targets_full[:Ne]
    Ts = T_local[:Ne]
    pi_s = np.power(source_pert_impact[ss], psi)
    f_weight = Wqs ** beta
    f_ffl = np.exp(delta * Ts)
    f_rdf = np.power(rdf_prior[ss], nu) if nu > 1e-8 else np.ones(Ne)
    num_cpu = f_weight * f_ffl * pi_s * f_rdf  # scber/chi_s/chi_t/rho OFF in this test

    cpu_order = np.lexsort((-num_cpu, ss))
    so_cpu, to_cpu, Wo_cpu, no_cpu = ss[cpu_order], ts[cpu_order], Ws[cpu_order], num_cpu[cpu_order]

    # ---- GPU ----
    beta_t = torch.tensor([beta], device=dev, dtype=torch.float32)
    delta_t = torch.tensor([delta], device=dev, dtype=torch.float32)
    psi_t = torch.tensor([psi], device=dev, dtype=torch.float32)
    nu_t = torch.tensor([nu], device=dev, dtype=torch.float32)
    k_core_eff = np.array([k_core])

    omega_batch = engine.compute_omega_batch_gpu(ctx, beta_t, delta_t, psi_t, nu_t, k_core_eff)
    order_grouped = engine.segmented_argsort_batch_gpu(ctx, omega_batch)

    so_gpu = ctx.ss.gather(0, order_grouped[0]).cpu().numpy()
    to_gpu = ctx.ts.gather(0, order_grouped[0]).cpu().numpy()
    Wo_gpu = ctx.W.gather(0, order_grouped[0]).cpu().numpy()
    no_gpu = omega_batch[0].gather(0, order_grouped[0]).cpu().numpy()

    src_match = np.array_equal(so_cpu, so_gpu)
    tgt_match = np.array_equal(to_cpu, to_gpu)
    w_match = np.allclose(Wo_cpu, Wo_gpu, rtol=1e-4)
    omega_rel_err = np.abs(no_gpu - no_cpu) / np.maximum(np.abs(no_cpu), 1e-8)

    print(f"beta={beta} delta={delta} psi={psi} nu={nu}: "
          f"src_match={src_match} tgt_match={tgt_match} w_match={w_match} "
          f"max_omega_rel_err={omega_rel_err.max():.2e}")
    assert src_match and tgt_match and w_match, "sorted edge content mismatch!"
    assert omega_rel_err.max() < 1e-3, "omega value mismatch in sorted order"

print("\nALL SORT VERIFICATION TESTS PASSED")
