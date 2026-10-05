"""One-off verification: fully vectorized select_edges (protection list +
kappa enforcement, both loop-free) vs the current loop-based engine.select_edges.
Not part of the deliverable -- confirms the vectorization changes nothing but
speed before it's wired into engine.py.

IMPORTANT FINDING (this session): real RPE1 5k has 868 perturbation nodes in
the gene panel (confirmed by reading the train_metacell.h5ad obs column
directly). At that scale the PERTURBATION PROTECTION loop -- not the kappa-cap
loop the original brief's Step 5 named -- is the dominant cost inside
select_edges: ~220ms/eval at n_pert=868, Ne~550k (measured directly against
the unmodified engine.select_edges). That is ~15 minutes alone across a
4096-eval Sobol sweep. Both loops are vectorized here using the invariant
that src/omega arriving at select_edges are already grouped source-ascending,
omega-descending-within-source (true for both the GPU and CPU presort paths --
see engine.py's FungiGPUContext docstring and run_dash_and_score's
np.lexsort((-num, ss)) call) -- so "top-3-by-omega per gene" is just the first
(<=3) elements of each gene's contiguous segment, no per-gene loop needed.

Compares output as SETS of (src,tgt) pairs since internal ordering is not
semantically meaningful (downstream consumers are all order-independent:
bincount, set-membership, igraph construction)."""
import time
import numpy as np
import engine


def select_edges_vectorized(omega, W, src, tgt, pert_nodes, n, lam,
                            per_gene_kappa, kappa_base):
    """
    Fully vectorized replacement for engine.select_edges -- no Python loop
    over pert_nodes, no Python loop over over-capacity genes. Relies on
    src/omega being grouped source-ascending, omega-descending-within-source
    (true of every caller in this codebase).
    """
    budget = int(np.round(n * lam))
    n_total = len(omega)

    effective_caps = np.maximum(
        (per_gene_kappa * kappa_base * n).astype(np.int64), 1)

    # ---- Vectorized protection list: top-3-by-omega per pert gene ----
    # src is grouped source-ascending, omega descending within group, so the
    # top-(<=3)-by-omega edges for gene p are just the first <=3 positions of
    # that gene's contiguous segment -- no per-gene scan needed.
    if len(pert_nodes) > 0:
        uniq_src, first_idx, counts = np.unique(src, return_index=True, return_counts=True)
        pert_arr = np.asarray(pert_nodes, dtype=np.int64)
        pos = np.searchsorted(uniq_src, pert_arr)
        pos = np.clip(pos, 0, len(uniq_src) - 1)
        found = uniq_src[pos] == pert_arr
        pert_pos = pos[found]
        starts = first_idx[pert_pos]
        sizes = np.minimum(counts[pert_pos], 3)

        max_k = 3
        offs = np.arange(max_k)
        idx_grid = starts[:, None] + offs[None, :]
        valid = offs[None, :] < sizes[:, None]
        prot_raw = np.unique(idx_grid[valid].astype(np.int64))
    else:
        prot_raw = np.array([], dtype=np.int64)

    max_prot = max(1, int(budget * 0.05))
    if len(prot_raw) > max_prot:
        prot = prot_raw[np.argsort(omega[prot_raw])[-max_prot:]]
    else:
        prot = prot_raw

    rem = budget - len(prot)
    if rem > 0:
        m = np.ones(n_total, dtype=bool)
        if len(prot) > 0:
            m[prot] = False
        av = np.where(m)[0]
        if rem < len(av):
            fi = av[np.argpartition(omega[av], -rem)[-rem:]]
        else:
            fi = av
        sel = np.concatenate([prot, fi]) if len(prot) > 0 else fi
    else:
        sel = prot[:budget]

    # ---- Vectorized kappa enforcement: per-gene rank via group boundaries ----
    ss = src[sel]
    nc = np.bincount(ss, minlength=n)
    over_cap = nc > effective_caps

    if np.any(over_cap):
        om_sel = omega[sel]
        order = np.lexsort((-om_sel, ss))
        ss_s = ss[order]
        sel_s = sel[order]

        change = np.empty(len(ss_s), dtype=bool)
        change[0] = True
        change[1:] = ss_s[1:] != ss_s[:-1]
        group_start = np.where(change)[0]
        group_sizes = np.diff(np.append(group_start, len(ss_s)))
        rank = np.arange(len(ss_s)) - np.repeat(group_start, group_sizes)

        caps_s = effective_caps[ss_s]
        keep_s = rank < caps_s
        sel = sel_s[keep_s]

        freed = budget - len(sel)
        if freed > 0:
            cm = np.ones(n_total, dtype=bool)
            cm[sel] = False
            cands = np.where(cm)[0]
            if len(cands) > 0:
                nf = min(freed, len(cands))
                sel = np.concatenate(
                    [sel, cands[np.argpartition(omega[cands], -nf)[-nf:]]])

    return src[sel], tgt[sel], W[sel]


def edge_set(s, t):
    return set(zip(s.tolist(), t.tolist()))


rng = np.random.default_rng(31)
n_genes = 5000
mismatches = 0
old_total_t, new_total_t = 0.0, 0.0
N_TRIALS = 15

for trial in range(N_TRIALS):
    Ne = int(rng.integers(400_000, 600_000))
    zipf_src = rng.zipf(a=1.3, size=Ne) - 1
    src = np.clip(zipf_src, 0, n_genes - 1).astype(np.int64)
    src = np.sort(src)
    omega = rng.exponential(1.0, size=Ne)
    order = np.lexsort((-omega, src))
    src, omega = src[order], omega[order]
    tgt = rng.integers(0, n_genes, size=Ne).astype(np.int64)
    W = rng.uniform(0.1, 1.0, size=Ne)

    # Vary n_pert around the real RPE1 5k value (868) plus edge cases (0, tiny, huge)
    n_pert = int(rng.choice([0, 1, 50, 868, 868, 1200, 3454]))
    pert_nodes = rng.choice(n_genes, size=min(n_pert, n_genes), replace=False)

    lam = rng.uniform(30.0, 50.0)  # real units: edges-per-gene (see note above)
    kappa_base = rng.uniform(0.02, 0.35)
    per_gene_kappa = np.ones(n_genes)
    hub_idx = rng.choice(n_genes, size=max(1, n_genes // 100), replace=False)
    per_gene_kappa[hub_idx] = 3.0

    t0 = time.perf_counter()
    old_s, old_t, old_w = engine.select_edges(
        omega, W, src, tgt, pert_nodes, n_genes, lam, per_gene_kappa, kappa_base)
    t1 = time.perf_counter()
    new_s, new_t, new_w = select_edges_vectorized(
        omega, W, src, tgt, pert_nodes, n_genes, lam, per_gene_kappa, kappa_base)
    t2 = time.perf_counter()
    old_total_t += (t1 - t0)
    new_total_t += (t2 - t1)

    old_set = edge_set(old_s, old_t)
    new_set = edge_set(new_s, new_t)
    match = (len(old_s) == len(new_s)) and (old_set == new_set)
    if not match:
        mismatches += 1
        print(f"trial {trial}: MISMATCH n_old={len(old_s)} n_new={len(new_s)} "
              f"sym_diff={len(old_set ^ new_set)} Ne={Ne} n_pert={n_pert}")
    else:
        print(f"trial {trial}: OK n_edges={len(old_s)} Ne={Ne} n_pert={n_pert:4d} "
              f"old={1000*(t1-t0):7.2f}ms new={1000*(t2-t1):6.2f}ms "
              f"speedup={(t1-t0)/max(t2-t1,1e-9):5.1f}x")

print(f"\n{N_TRIALS - mismatches}/{N_TRIALS} trials matched exactly (same edge set & count)")
print(f"total time -- old: {1000*old_total_t:.1f}ms  new: {1000*new_total_t:.1f}ms  "
      f"speedup: {old_total_t/max(new_total_t,1e-9):.2f}x")
assert mismatches == 0, "Vectorized select_edges diverged from the original!"
print("\nPASS: fully vectorized select_edges produces identical edge sets, "
      f"{old_total_t/max(new_total_t,1e-9):.1f}x faster overall.")
