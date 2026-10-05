"""One-off verification: fast topology functions vs igraph/powerlaw. Not part of the deliverable."""
import numpy as np
import igraph as ig
import powerlaw
import engine

rng = np.random.default_rng(55)
n_genes = 800

for trial in range(5):
    n_edges = rng.integers(3000, 8000)
    all_pairs = rng.choice(n_genes * n_genes, size=n_edges, replace=False)
    ss = (all_pairs // n_genes).astype(np.int64)
    st = (all_pairs % n_genes).astype(np.int64)
    sw = rng.uniform(0.1, 1.0, n_edges)

    # ---- exact (igraph) ----
    edges = list(zip(ss.tolist(), st.tolist()))
    ig_g = ig.Graph(n=n_genes, edges=edges, directed=True, edge_attrs={'weight': sw.tolist()})
    rho_exact = ig_g.assortativity_degree(directed=True)
    ig_u = ig_g.as_undirected(mode="collapse", combine_edges=dict(weight="sum"))
    c_exact = ig_u.transitivity_undirected()

    od = np.bincount(ss, minlength=n_genes)
    cap = int(n_genes * 0.15)
    cd = od[(od > 0) & (od < cap)]
    alpha_exact = 1.0
    if len(cd) >= 20 and len(np.unique(cd)) >= 3:
        fit_ao = powerlaw.Fit(cd, discrete=True, verbose=False)
        xmin_ao = fit_ao.power_law.xmin
        if int(np.sum(cd >= xmin_ao)) < 50:
            fit_ao = powerlaw.Fit(cd, xmin=6, discrete=True, verbose=False)
        alpha_exact = fit_ao.power_law.alpha

    # ---- fast (numpy) ----
    rho_fast = engine.assortativity_fast(ss, st, n_genes)
    c_fast = engine.clustering_wedge_sample(ss, st, n_genes, n_samples=50000,
                                            rng=np.random.default_rng(1))
    alpha_fixed = engine.hill_alpha_fast(od, n_genes)
    alpha_auto = engine.fast_auto_xmin_alpha(od, n_genes)

    rho_diff = abs(rho_exact - rho_fast) if np.isfinite(rho_exact) else float('nan')
    c_diff = abs(c_exact - c_fast) if np.isfinite(c_exact) else float('nan')

    print(f"trial {trial} (n_edges={n_edges}): "
          f"rho exact={rho_exact:.4f} fast={rho_fast:.4f} diff={rho_diff:.4f} | "
          f"C exact={c_exact:.4f} fast={c_fast:.4f} diff={c_diff:.4f} | "
          f"alpha exact={alpha_exact:.4f} fixed-xmin={alpha_fixed:.4f} "
          f"(diff={abs(alpha_exact-alpha_fixed):.4f})  auto-xmin={alpha_auto:.4f} "
          f"(diff={abs(alpha_exact-alpha_auto):.4f})")
