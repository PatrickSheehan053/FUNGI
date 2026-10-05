"""
gpu_stat_kernels.py -- obj_003.2 shared GPU statistical kernels (pure PyTorch, exact).

Two rank-based kernels that reproduce the scipy calls at the heart of obj_003 (stat_prec/wass) and
obj_004 (spec_prec):

  mannwhitneyu_gpu(ref, intv) == scipy.stats.mannwhitneyu(ref, intv, axis=0)
      two-sided, use_continuity=True, ASYMPTOTIC path (with tie + continuity correction).
  wasserstein1d_gpu(a, b)     == scipy.stats.wasserstein_distance(a[:,c], b[:,c]) per column.

EXACTNESS NOTES (load-bearing -- the framework requires stat_prec bit-identical across experiments):
- The sort/ranking runs at the CALLER'S value dtype (obj_003 X is float32, obj_004 X is float64), so
  the tie structure is byte-identical to what scipy sees on the same array. Do NOT upcast values before
  sorting.
- Rank SUMS reach ~4e8 (obj_004: ~28k ref cells x ranks up to ~30k), which exceeds float32's 2^24
  exact-integer range -> ranks are represented as the integer `rank2 = 2*rank` in int64 (exact), and
  U/tie_term are int64; only the final z/p arithmetic is float64.
- Average ranks (scipy's tie handling) are computed exactly as rank = (#strictly-less) + (#equal+1)/2
  via two torch.searchsorted passes (left/right) -- no separate tie-grouping pass needed.
- tie-corrected variance:  var = (n1*n2/12)*[(N+1) - Σ(t^3 - t)/(N(N-1))],  and
  Σ over tie-groups of (t^3 - t) == Σ over all N elements of (t^2 - 1)  (t = #equal for that element).
- continuity: numerator -= sign(numerator)*0.5 ; two-sided p = 2*Φ(-|z|) = 2*ndtr(-|z|), clamped <=1.
- ASYMPTOTIC ONLY: scipy uses the EXACT null when min(n_ref,n_int) <= 8 AND a column is tie-free. Those
  columns must be routed to scipy by the CALLER (see integration: regulators with n_knock<=8 -> scipy).
  For min>8 (or n<=8 with ties) scipy ALWAYS uses asymptotic, which this kernel matches.
"""
import time

import numpy as np
import torch


def _ct(x, device):
    """[n, C] array/tensor -> [C, n] contiguous tensor on device, preserving dtype (float32/float64).
    INPUT-ADAPTER ONLY (no statistical math): accepts a numpy array OR an already-on-device torch
    tensor (the obj_003.2 gpu_context resident-reference path slices columns on-device and passes a
    CUDA tensor here -> no host->device copy). numpy path is unchanged; validated identical."""
    if isinstance(x, torch.Tensor):
        t = x.to(device)
    else:
        t = torch.as_tensor(np.asarray(x), device=device)
    return t.transpose(0, 1).contiguous()


@torch.no_grad()
def mannwhitneyu_gpu(ref, intv, device="cuda", chunk=4096, cooldown_ms=0):
    """Two-sided asymptotic MWU (continuity + tie corrected), batched over columns.
    ref [n_ref, C], intv [n_int, C] (numpy). Returns pvals np.float64 [C]. Values are sorted at their
    native dtype so ties match scipy exactly. ref/intv may be numpy OR an on-device torch tensor
    (gpu_context resident-reference path); slicing + .shape work identically for both."""
    n1, C = ref.shape
    n2 = intv.shape[0]
    N = n1 + n2
    out = np.empty(C, dtype=np.float64)
    mu = n1 * n2 / 2.0
    for s in range(0, C, chunk):
        e = min(s + chunk, C)
        r = _ct(ref[:, s:e], device)     # [c, n1]
        iv = _ct(intv[:, s:e], device)   # [c, n2]
        V = torch.cat([r, iv], dim=1)    # [c, N], caller dtype
        sorted_V, _ = torch.sort(V, dim=1)
        cl = torch.searchsorted(sorted_V, V, right=False)          # int64 [c, N]  (#strictly less)
        cq = torch.searchsorted(sorted_V, V, right=True) - cl      # int64 [c, N]  (#equal)
        rank2 = 2 * cl + cq + 1                                    # int64, == 2*avg_rank (exact)
        R1_2 = rank2[:, :n1].sum(dim=1)                            # int64 [c]  == 2*sum(ranks of ref)
        U1 = R1_2.double() * 0.5 - n1 * (n1 + 1) / 2.0             # float64 [c]
        tie_term = (cq * cq - 1).sum(dim=1).double()              # Σ(t^2-1) == Σ(t^3-t), exact int64
        var = (n1 * n2 / 12.0) * ((N + 1) - tie_term / (N * (N - 1)))
        sd = torch.sqrt(var)
        num = U1 - mu
        num = num - torch.sign(num) * 0.5                         # continuity
        z = num / sd
        p = 2.0 * torch.special.ndtr(-z.abs())
        p = torch.clamp(p, max=1.0)
        p = torch.where(sd > 0, p, torch.ones_like(p))            # all-tied column -> p=1.0
        out[s:e] = p.detach().cpu().numpy()
        if cooldown_ms:
            time.sleep(cooldown_ms / 1000.0)
    return out


@torch.no_grad()
def wasserstein1d_gpu(a, b, device="cuda", chunk=4096, cooldown_ms=0):
    """Exact 1-D W1 per column == scipy.stats.wasserstein_distance(a[:,c], b[:,c]).
    a [nA, C], b [nB, C] (numpy or on-device tensor). Returns np.float64 [C]. Arithmetic in float64
    (matches scipy)."""
    nA, C = a.shape
    nB = b.shape[0]
    out = np.empty(C, dtype=np.float64)
    for s in range(0, C, chunk):
        e = min(s + chunk, C)
        av = _ct(a[:, s:e], device).double()   # [c, nA]
        bv = _ct(b[:, s:e], device).double()    # [c, nB]
        au, _ = torch.sort(av, dim=1)
        bu, _ = torch.sort(bv, dim=1)
        allv = torch.cat([au, bu], dim=1)
        allv, _ = torch.sort(allv, dim=1)                        # [c, nA+nB]
        deltas = allv[:, 1:] - allv[:, :-1]                      # [c, nA+nB-1]
        q = allv[:, :-1]                                         # evaluation points
        # CDFs at q via searchsorted 'right' (scipy convention), / n
        u_cdf = torch.searchsorted(au, q, right=True).double() / nA
        v_cdf = torch.searchsorted(bu, q, right=True).double() / nB
        w = ((u_cdf - v_cdf).abs() * deltas).sum(dim=1)
        out[s:e] = w.detach().cpu().numpy()
        if cooldown_ms:
            time.sleep(cooldown_ms / 1000.0)
    return out
