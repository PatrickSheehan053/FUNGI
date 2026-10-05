"""One-off verification: GPU omega vs CPU numpy reference. Not part of the deliverable."""
import numpy as np
import torch
import engine

rng = np.random.default_rng(42)
n_genes = 60
n_edges_full = 3000

all_pairs = rng.choice(n_genes * n_genes, size=n_edges_full, replace=False)
sources_full = (all_pairs // n_genes).astype(np.int64)
targets_full = (all_pairs % n_genes).astype(np.int64)
W_full = rng.uniform(0.01, 1.0, n_edges_full).astype(np.float64)
order = np.argsort(W_full)[::-1]
W_full, sources_full, targets_full = W_full[order], sources_full[order], targets_full[order]
W_q_full = rng.uniform(0.05, 1.0, n_edges_full).astype(np.float64)

source_pert_impact = rng.uniform(0.5, 3.0, n_genes).astype(np.float64)
rdf_prior = rng.uniform(0.1, 2.0, n_genes).astype(np.float64)
chi_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)
chi_t_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)
rho_prior = rng.uniform(0.5, 1.5, n_genes).astype(np.float64)

shatter_cfg = {"max_edge_count": 200}  # Ne = min(3000, max(400,250)) = 400
k_core_bounds = (5.0, 20.0)
n_buckets = 8
kernel_flags = {"weight": True, "ffl": True, "pert_impact": True, "rdf": True,
                "scber": True, "chi_s": True, "chi_t": True, "rho": True}

ctx = engine.init_gpu_context(
    W_full, W_q_full, sources_full, targets_full, n_genes, shatter_cfg,
    source_pert_impact, k_core_bounds,
    rdf_prior=rdf_prior, chi_prior=chi_prior, chi_t_prior=chi_t_prior,
    rho_prior=rho_prior, kernel_flags=kernel_flags, n_ffl_buckets=n_buckets)

assert ctx.enabled
Ne = ctx.Ne
print(f"Ne={Ne}")

# Pick test configs whose k_core lands exactly on bucket grid points -- makes
# the FFL term an EXACT comparison too, not just "close to nearest bucket".
bucket_kcores = ctx.k_core_bucket_values
test_configs = [
    dict(beta=1.5, delta=0.8, psi=1.2, nu=0.5, k_core=bucket_kcores[0]),
    dict(beta=2.0, delta=0.3, psi=0.7, nu=1.0, k_core=bucket_kcores[3]),
    dict(beta=0.6, delta=1.5, psi=2.0, nu=0.0, k_core=bucket_kcores[-1]),
]

for cfg in test_configs:
    beta, delta, psi, nu, k_core = cfg["beta"], cfg["delta"], cfg["psi"], cfg["nu"], cfg["k_core"]

    # ---- CPU reference (mirrors run_dash_and_score exactly) ----
    T_local = engine.compute_dynamic_topology(W_full, sources_full, targets_full, k_core, n_genes)
    Ws, Wqs = W_full[:Ne], W_q_full[:Ne]
    ss, ts = sources_full[:Ne], targets_full[:Ne]
    Ts = T_local[:Ne]
    pi_s = np.power(source_pert_impact[ss], psi)
    chi = chi_prior[ss]
    rho = rho_prior[ss]
    chi_t = chi_t_prior[ts]
    f_weight = Wqs ** beta
    f_ffl = np.exp(delta * Ts)
    f_pi = pi_s
    f_rdf = np.power(rdf_prior[ss], nu) if nu > 1e-8 else np.ones(Ne)
    num_cpu = f_weight * f_ffl * f_pi * f_rdf * chi * rho * chi_t

    # ---- GPU result ----
    dev = torch.device("cuda")
    beta_t = torch.tensor([beta], device=dev, dtype=torch.float32)
    delta_t = torch.tensor([delta], device=dev, dtype=torch.float32)
    psi_t = torch.tensor([psi], device=dev, dtype=torch.float32)
    nu_t = torch.tensor([nu], device=dev, dtype=torch.float32)
    k_core_eff = np.array([k_core])

    omega_gpu = engine.compute_omega_batch_gpu(ctx, beta_t, delta_t, psi_t, nu_t, k_core_eff)
    omega_gpu_np = omega_gpu[0].cpu().numpy()

    # un-permute GPU (grouped order) back to original [:Ne] order for comparison
    inv_perm = np.argsort(ctx.group_perm)
    omega_gpu_unpermuted = omega_gpu_np[inv_perm]

    rel_err = np.abs(omega_gpu_unpermuted - num_cpu) / np.maximum(np.abs(num_cpu), 1e-8)
    max_rel_err = rel_err.max()
    print(f"beta={beta} delta={delta} psi={psi} nu={nu} k_core={k_core:.2f}: "
          f"max_rel_err={max_rel_err:.2e}  {'PASS' if max_rel_err < 1e-3 else 'FAIL'}")
    assert max_rel_err < 1e-3, f"omega mismatch too large: {max_rel_err}"

print("\nALL OMEGA VERIFICATION TESTS PASSED")
