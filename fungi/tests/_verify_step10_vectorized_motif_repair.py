"""One-off verification: vectorized _motif_repair_swap vs the current
engine._motif_repair_swap. Not part of the deliverable.

IMPORTANT FINDING (this session): _motif_repair_swap is the dominant
remaining bottleneck after fixing alpha (Step 1) and select_edges -- the
sparse matmul A2 = adj @ adj is fast (~70-90ms at n_survived=200k,
n_genes=5000), but the Python `for ci,cj,cv in zip(A2_coo.row, A2_coo.col,
A2_coo.data): if (ci,cj) not in selected_set...` loop over A2's nonzeros
costs ~3.7 SECONDS at the same scale, because 2-hop reachability densifies
fast (6.8M nonzeros out of 25M possible entries on a 200k-edge, 5000-node
graph). This single loop accounts for nearly all of the measured 2.9-7.1s/eval
cost of this function at realistic scale -- bigger than every other per-eval
cost in the pipeline combined (alpha: <1ms, select_edges: ~220ms at the real
RPE1 n_pert=868).

The fix here goes beyond "remove the Python loop": rather than materializing
ALL of A2's millions of nonzero entries as candidates and then filtering,
this version starts from the (much smaller) UNSELECTED portion of the full
omega-scored edge pool (Ne - n_selected edges, e.g. ~350k) and queries A2
directly for those pairs only. Candidates not in the pool always have
omega=0 in the original algorithm (via _lookup_omega's zero-default) and can
never win a swap against a selected edge (which always has omega>0) -- so
skipping them entirely changes nothing about which swaps get made, it just
stops generating and sorting millions of candidates that were always going
to be discarded."""
import time
import numpy as np
import scipy.sparse as sp
import engine


def motif_repair_swap_vectorized(surv_s, surv_t, surv_W, omega_full, src_full,
                                  tgt_full, n_genes, max_swap_fraction=0.03,
                                  rng=None):
    n_selected = len(surv_s)
    budget_swaps = max(1, int(n_selected * max_swap_fraction))

    if n_selected < 10 or budget_swaps < 1:
        return surv_s, surv_t, surv_W

    try:
        adj = sp.coo_matrix(
            (np.ones(n_selected, dtype=bool), (surv_s, surv_t)),
            shape=(n_genes, n_genes)).tocsr()
        A2 = adj @ adj  # bool dtype: ~2x faster fancy-indexing below

        # Mark which full-pool edges are already selected (single searchsorted
        # of Ne queries against the n_selected sorted keys -- cheap).
        selected_keys = np.sort(
            surv_s.astype(np.int64) * n_genes + surv_t.astype(np.int64))
        full_keys = src_full.astype(np.int64) * n_genes + tgt_full.astype(np.int64)
        ins = np.searchsorted(selected_keys, full_keys)
        ins = np.clip(ins, 0, len(selected_keys) - 1)
        is_selected_full = selected_keys[ins] == full_keys

        # Candidates can only ever be unselected pool edges -- anything not in
        # the pool defaults to omega=0 and can never beat a selected edge
        # (always omega>0), so there is no need to ever generate it.
        unsel_src = src_full[~is_selected_full]
        unsel_tgt = tgt_full[~is_selected_full]
        unsel_omega = omega_full[~is_selected_full]

        vals = np.asarray(A2[unsel_src, unsel_tgt]).ravel()
        keep = (vals > 0) & (unsel_src != unsel_tgt)
        candidate_src = unsel_src[keep]
        candidate_tgt = unsel_tgt[keep]
        close_scores = unsel_omega[keep]

        if len(candidate_src) == 0:
            return surv_s, surv_t, surv_W

        src_keys_full = full_keys
        sort_order_full = np.argsort(src_keys_full)
        keys_sorted_full = src_keys_full[sort_order_full]
        omega_sorted_full = omega_full[sort_order_full]

        def _lookup_omega(s_arr, t_arr):
            q = s_arr.astype(np.int64) * n_genes + t_arr.astype(np.int64)
            ins2 = np.searchsorted(keys_sorted_full, q)
            ins2 = np.clip(ins2, 0, len(keys_sorted_full) - 1)
            matched = keys_sorted_full[ins2] == q
            scores = np.zeros(len(q), dtype=np.float64)
            scores[matched] = omega_sorted_full[ins2[matched]]
            return scores

        cand_s = candidate_src.astype(np.int64)
        cand_t = candidate_tgt.astype(np.int64)

        sel_s = surv_s.astype(np.int64)
        sel_t = surv_t.astype(np.int64)
        selected_scores = _lookup_omega(sel_s, sel_t)

        cand_order = np.argsort(close_scores)[::-1]
        sel_order = np.argsort(selected_scores)

        selected_mask = np.ones(n_selected, dtype=bool)
        new_src, new_tgt = [], []
        n_swapped = 0

        for i in range(min(budget_swaps, len(cand_order), len(sel_order))):
            ci_idx = cand_order[i]
            sel_idx = sel_order[i]
            if close_scores[ci_idx] <= selected_scores[sel_idx]:
                break
            selected_mask[sel_idx] = False
            new_src.append(int(cand_s[ci_idx]))
            new_tgt.append(int(cand_t[ci_idx]))
            n_swapped += 1

        if n_swapped == 0:
            return surv_s, surv_t, surv_W

        keep_idx = np.where(selected_mask)[0]
        final_s = np.concatenate([surv_s[keep_idx],
                                   np.array(new_src, dtype=surv_s.dtype)])
        final_t = np.concatenate([surv_t[keep_idx],
                                   np.array(new_tgt, dtype=surv_t.dtype)])
        final_W = np.concatenate([surv_W[keep_idx],
                                   np.zeros(n_swapped, dtype=surv_W.dtype)])
        return final_s, final_t, final_W

    except Exception:
        return surv_s, surv_t, surv_W


def edge_set(s, t):
    return set(zip(s.tolist(), t.tolist()))


rng = np.random.default_rng(61)
n_genes = 5000
mismatches = 0
old_total_t, new_total_t = 0.0, 0.0
N_TRIALS = 10

for trial in range(N_TRIALS):
    # Real edge arrays have unique (src,tgt) pairs (one row per Regulator/Target
    # gene pair) -- deduplicate the synthetic pool to respect that invariant.
    Ne_draw = int(rng.integers(400_000, 600_000))
    keys = rng.choice(n_genes * n_genes, size=Ne_draw, replace=False)
    src_full = (keys // n_genes).astype(np.int64)
    tgt_full = (keys % n_genes).astype(np.int64)
    Ne = len(src_full)
    omega_full = rng.exponential(1.0, size=Ne)

    n_survived = int(rng.integers(120_000, 260_000))
    sel_idx = rng.choice(Ne, size=min(n_survived, Ne), replace=False)
    surv_s = src_full[sel_idx]
    surv_t = tgt_full[sel_idx]
    surv_W = rng.uniform(0.1, 1.0, size=len(sel_idx))

    seed = int(rng.integers(0, 1_000_000))
    t0 = time.perf_counter()
    old_s, old_t, old_w = engine._motif_repair_swap(
        surv_s, surv_t, surv_W, omega_full, src_full, tgt_full, n_genes,
        max_swap_fraction=0.03, rng=np.random.default_rng(seed))
    t1 = time.perf_counter()
    new_s, new_t, new_w = motif_repair_swap_vectorized(
        surv_s, surv_t, surv_W, omega_full, src_full, tgt_full, n_genes,
        max_swap_fraction=0.03, rng=np.random.default_rng(seed))
    t2 = time.perf_counter()
    old_total_t += (t1 - t0)
    new_total_t += (t2 - t1)

    old_set = edge_set(old_s, old_t)
    new_set = edge_set(new_s, new_t)
    match = (len(old_s) == len(new_s)) and (old_set == new_set)
    if not match:
        mismatches += 1
        print(f"trial {trial}: MISMATCH n_old={len(old_s)} n_new={len(new_s)} "
              f"sym_diff={len(old_set ^ new_set)} n_survived={len(sel_idx)} Ne={Ne}")
    else:
        print(f"trial {trial}: OK n_edges={len(old_s)} n_survived={len(sel_idx):7d} Ne={Ne:7d} "
              f"old={1000*(t1-t0):8.1f}ms new={1000*(t2-t1):7.2f}ms "
              f"speedup={(t1-t0)/max(t2-t1,1e-9):6.1f}x")

print(f"\n{N_TRIALS - mismatches}/{N_TRIALS} trials matched exactly (same edge set & count)")
print(f"total time -- old: {1000*old_total_t:.1f}ms  new: {1000*new_total_t:.1f}ms  "
      f"speedup: {old_total_t/max(new_total_t,1e-9):.1f}x")
assert mismatches == 0, "Vectorized _motif_repair_swap diverged from the original!"
print("\nPASS: vectorized _motif_repair_swap produces identical results, "
      f"{old_total_t/max(new_total_t,1e-9):.1f}x faster overall.")
