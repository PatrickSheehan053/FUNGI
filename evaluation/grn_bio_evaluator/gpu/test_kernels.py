"""
obj_003.2 kernel unit tests -- gpu_stat_kernels vs scipy (the exactness gate).

Cases: (a) continuous no-tie, (b) tie-heavy integer, (c) sparse-zero float32 (like real expression),
(d) large-ref/small-int (obj_004 shape). Asymptotic regime only (min(n_ref,n_int) > 8) -- the
min<=8 & tie-free exact regime is routed to scipy by the integration, and is checked separately.
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_stat_kernels import mannwhitneyu_gpu, wasserstein1d_gpu

DEV = "cuda"
rng = np.random.default_rng(0)


def scipy_mwu(ref, intv):
    _, p = stats.mannwhitneyu(ref, intv, axis=0)  # two-sided, use_continuity=True default
    return np.asarray(p, dtype=np.float64)


def scipy_wass(a, b):
    return np.array([stats.wasserstein_distance(a[:, c], b[:, c]) for c in range(a.shape[1])])


def check(name, gp, sp, rtol=1e-6, atol=1e-9):
    d = np.abs(gp - sp)
    rel = d / (np.abs(sp) + 1e-12)
    ok = bool(np.all((d <= atol) | (rel <= rtol)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: max_abs_diff={d.max():.2e} max_rel={rel.max():.2e}")
    return ok


def main():
    allok = True
    print("=== mannwhitneyu_gpu vs scipy (asymptotic regime, min_n>8) ===")
    # (a) continuous no-tie float64
    ref = rng.standard_normal((1600, 500)).astype(np.float64)
    intv = rng.standard_normal((40, 500)).astype(np.float64) + 0.3
    allok &= check("a continuous f64", mannwhitneyu_gpu(ref, intv, DEV), scipy_mwu(ref, intv))
    # (b) tie-heavy small-integer float64
    ref = rng.integers(0, 5, (300, 500)).astype(np.float64)
    intv = rng.integers(0, 5, (60, 500)).astype(np.float64)
    allok &= check("b tie-heavy int f64", mannwhitneyu_gpu(ref, intv, DEV), scipy_mwu(ref, intv))
    # (c) sparse-zero float32 (like normalized+log1p expression: ~70% zeros)
    def sparsef32(n, C):
        x = rng.standard_normal((n, C)).astype(np.float32)
        x[rng.random((n, C)) < 0.7] = 0.0
        return np.abs(x)
    ref = sparsef32(1612, 400); intv = sparsef32(37, 400)
    allok &= check("c sparse-zero f32 (obj_003 shape)", mannwhitneyu_gpu(ref, intv, DEV), scipy_mwu(ref, intv))
    # (d) large-ref small-int float64 (obj_004: ~28k perturbed ref)
    ref = sparsef32(28000, 200).astype(np.float64); intv = sparsef32(30, 200).astype(np.float64)
    allok &= check("d large-ref f64 (obj_004 shape)", mannwhitneyu_gpu(ref, intv, DEV), scipy_mwu(ref, intv))

    print("=== wasserstein1d_gpu vs scipy ===")
    ref = rng.standard_normal((1600, 300)).astype(np.float32)
    intv = (rng.standard_normal((40, 300)) + 0.5).astype(np.float32)
    allok &= check("wass sparse-ish f32", wasserstein1d_gpu(ref, intv, DEV), scipy_wass(ref, intv))
    ref = sparsef32(1612, 300); intv = sparsef32(37, 300)
    allok &= check("wass sparse-zero f32", wasserstein1d_gpu(ref, intv, DEV), scipy_wass(ref, intv))

    print("=== stat_prec threshold-flip test (the real metric) ===")
    # p<0.05 boolean must be IDENTICAL (stat_prec is a count of these)
    ref = sparsef32(1612, 2000); intv = sparsef32(45, 2000)
    gp = mannwhitneyu_gpu(ref, intv, DEV); sp = scipy_mwu(ref, intv)
    flips = int(np.sum((gp < 0.05) != (sp < 0.05)))
    print(f"  [{'PASS' if flips == 0 else 'FAIL'}] p<0.05 flips: {flips}/2000")
    allok &= (flips == 0)

    print("=== confirm min_n<=8 tie-free is the exact regime scipy diverges on (justifies routing) ===")
    ref = rng.standard_normal((1600, 50)).astype(np.float64)
    intv = (rng.standard_normal((6, 50)) + 0.5).astype(np.float64)  # n_int=6<=8, continuous->tie-free
    gp = mannwhitneyu_gpu(ref, intv, DEV); sp = scipy_mwu(ref, intv)
    diff = np.abs(gp - sp).max()
    print(f"  n_int=6 tie-free: gpu(asymptotic) vs scipy(exact) max_diff={diff:.2e} "
          f"-> {'DIVERGES as expected (route to scipy)' if diff > 1e-4 else 'matches'}")

    print("\nALL KERNEL TESTS PASS:" , allok)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
