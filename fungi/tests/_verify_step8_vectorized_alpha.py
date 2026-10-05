"""One-off verification: vectorized fast_auto_xmin_alpha vs the old loop-based
reference implementation it replaced, plus both vs exact powerlaw.Fit. Not
part of the deliverable -- confirms the Step 1 vectorization changed nothing
but speed."""
import numpy as np
import powerlaw
import engine


def old_fast_auto_xmin_alpha(od, n_genes, cap_frac=0.15, min_tail=50, fallback_xmin=6):
    """Loop-based reference -- this is exactly what engine.fast_auto_xmin_alpha
    was before the Step 1 vectorization. Kept here only for this comparison."""
    cap = int(n_genes * cap_frac)
    cd = np.sort(od[(od > 0) & (od < cap)])
    n_total = len(cd)
    if n_total < 20 or len(np.unique(cd)) < 3:
        return 1.0

    candidates = np.unique(cd)
    counts_ge = n_total - np.searchsorted(cd, candidates, side="left")
    keep = counts_ge >= 20
    candidates, counts_ge = candidates[keep], counts_ge[keep]
    if len(candidates) == 0:
        return 1.0

    def _fit(xmin_val):
        tail = cd[cd >= xmin_val].astype(np.float64)
        denom = np.sum(np.log(tail / (xmin_val - 0.5)))
        if denom <= 0:
            return None, None
        alpha_val = 1.0 + len(tail) / denom
        sorted_tail = np.sort(tail)
        emp_ccdf = 1.0 - np.arange(len(sorted_tail)) / len(sorted_tail)
        fit_ccdf = (sorted_tail / xmin_val) ** (1.0 - alpha_val)
        ks = float(np.max(np.abs(emp_ccdf - fit_ccdf)))
        return alpha_val, ks

    best_ks, best_alpha, best_xmin = np.inf, 1.0, candidates[0]
    for xmin_val in candidates:
        alpha_val, ks = _fit(xmin_val)
        if alpha_val is not None and ks < best_ks:
            best_ks, best_alpha, best_xmin = ks, alpha_val, xmin_val

    if int(np.sum(cd >= best_xmin)) < min_tail:
        alpha_fb, _ = _fit(fallback_xmin)
        if alpha_fb is not None:
            best_alpha = alpha_fb
    return float(best_alpha)


def make_pareto_od(rng, n_genes, alpha=2.3, xmin=3):
    u = rng.random(n_genes)
    raw = xmin * (1 - u) ** (-1.0 / (alpha - 1))
    return np.clip(np.round(raw), 0, n_genes - 1).astype(np.int64)


def make_uniform_od(rng, n_genes, lo=1, hi=40):
    return rng.integers(lo, hi, size=n_genes).astype(np.int64)


rng = np.random.default_rng(7)
n_genes = 5000

max_old_new_diff = 0.0
max_new_exact_diff = 0.0
n_trials = 0

shapes = (
    [("pareto", lambda r: make_pareto_od(r, n_genes, alpha=2.0 + 0.5 * r.random()))] * 10
    + [("uniform", lambda r: make_uniform_od(r, n_genes))] * 10
)

print(f"{'shape':9s} {'old':>8s} {'new':>8s} {'old-new':>10s} {'exact':>8s} {'new-exact':>10s}")
for shape_name, gen in shapes:
    od = gen(rng)
    old_a = old_fast_auto_xmin_alpha(od, n_genes)
    new_a = engine.fast_auto_xmin_alpha(od, n_genes)
    diff_old_new = abs(old_a - new_a)
    max_old_new_diff = max(max_old_new_diff, diff_old_new)
    n_trials += 1

    cap = int(n_genes * 0.15)
    cd = od[(od > 0) & (od < cap)]
    exact_a = 1.0
    if len(cd) >= 20 and len(np.unique(cd)) >= 3:
        fit_ao = powerlaw.Fit(cd, discrete=True, verbose=False)
        xmin_ao = fit_ao.power_law.xmin
        if int(np.sum(cd >= xmin_ao)) < 50:
            fit_ao = powerlaw.Fit(cd, xmin=6, discrete=True, verbose=False)
        exact_a = float(fit_ao.power_law.alpha)
    diff_new_exact = abs(new_a - exact_a)
    max_new_exact_diff = max(max_new_exact_diff, diff_new_exact)

    print(f"{shape_name:9s} {old_a:8.4f} {new_a:8.4f} {diff_old_new:10.6f} "
          f"{exact_a:8.4f} {diff_new_exact:10.6f}")

# Tiny-array edge cases (n_total < 20, < 3 unique values, etc.)
edge_cases = [
    np.zeros(100, dtype=np.int64),
    np.ones(100, dtype=np.int64),
    np.array([0, 0, 1, 1, 2], dtype=np.int64),
    rng.integers(1, 3, size=15).astype(np.int64),
]
for i, od in enumerate(edge_cases):
    old_a = old_fast_auto_xmin_alpha(od, n_genes)
    new_a = engine.fast_auto_xmin_alpha(od, n_genes)
    diff = abs(old_a - new_a)
    max_old_new_diff = max(max_old_new_diff, diff)
    print(f"edge{i:1d}     {old_a:8.4f} {new_a:8.4f} {diff:10.6f}")

print(f"\n{n_trials} random trials + {len(edge_cases)} edge cases")
print(f"max |old - new| (should be ~0, same algorithm just vectorized): {max_old_new_diff:.8f}")
print(f"max |new - exact powerlaw.Fit| (informational only): {max_new_exact_diff:.4f}")
assert max_old_new_diff < 1e-6, "Vectorized fast_auto_xmin_alpha diverged from the loop-based reference!"
print("\nPASS: vectorized fast_auto_xmin_alpha is numerically identical to the old loop-based version.")
