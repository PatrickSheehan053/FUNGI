"""
gpu_context.py -- obj_003.2 resident-GPU reference cache (the gpu_context refactor, Step 3).

The transfer-bound cost of the first obj_003.2 integration was re-slicing + re-uploading the FIXED
reference matrix (control ~1.6k cells or global-perturbed ~28k cells x genes) to the GPU once PER
REGULATOR (~868 host->device copies per graph). The GPU sort/rank is fast; those copies dominate.

GpuContext uploads each reference matrix to the GPU ONCE (keyed by the numpy array's id, so the same
X_ctrl_test / X_pert_test object is uploaded a single time and reused across all k and all regulators),
and slices the needed target columns ON-DEVICE (index_select) before calling the validated kernel. The
per-regulator knockdown slice (n_knock x n_targets, small) is still uploaded per call by the kernel.

EXACTNESS INVARIANTS (Patrick's Step 2 -- do not break):
  - The resident tensor is uploaded at the CALLER'S value dtype (obj_003 float32 / obj_004 float64):
    torch.as_tensor(X_ref) preserves numpy dtype; NO upcast on load, so the tie structure the kernel
    sorts is byte-identical to what scipy sees.
  - Column slicing uses index_select(1, cols) -> the columns come back in EXACTLY the caller's order
    (the order of group["Target"]); .contiguous() in _ct. stat_prec is a count and wass a mean, both
    column-order-invariant, but order is preserved anyway.
  - The int64 rank2 arithmetic lives entirely inside the (unchanged) kernel.
  - Caller still routes n_knock <= 8 to scipy; GpuContext is only used for n_knock > 8.
"""
import torch

import gpu_stat_kernels as K


def _safe_chunk(n_ref, n_int):
    """VRAM-aware column-batch size. Per chunk the kernel holds ~ chunk * (n_ref+n_int) float64/int64
    across a few buffers; cap the working set well under the 8 GB RTX 2070 (resident ref may be ~2 GB
    for the 28k perturbed matrix)."""
    N = n_ref + n_int
    budget_cols = int(1.2e9 / (N * 40))          # ~1.2 GB / (N * ~40 bytes-per-col-element-buffers)
    return max(256, min(4096, budget_cols))


class GpuContext:
    def __init__(self, device="cuda", cooldown_ms=0):
        self.device = device
        self.cooldown_ms = cooldown_ms
        self._cache = {}   # id(X_ref) -> resident tensor (keeps a ref to X_ref so id stays valid)
        self._keepalive = {}

    def _resident(self, X_ref):
        key = id(X_ref)
        t = self._cache.get(key)
        if t is None:
            t = torch.as_tensor(X_ref, device=self.device)   # preserve dtype; upload ONCE
            self._cache[key] = t
            self._keepalive[key] = X_ref                      # prevent id reuse
        return t

    def _cols(self, target_cols):
        return torch.as_tensor(target_cols, device=self.device, dtype=torch.long)

    def mwu_cols(self, X_ref, target_cols, int_slice):
        """MWU p-values [C] with the reference sliced ON-DEVICE. int_slice is the (small) knockdown
        slice [n_int, C] (numpy). Equivalent to mannwhitneyu_gpu(X_ref[:, target_cols], int_slice)."""
        ref_slice = self._resident(X_ref).index_select(1, self._cols(target_cols))  # [n_ref, C] on GPU
        return K.mannwhitneyu_gpu(ref_slice, int_slice, device=self.device,
                                  chunk=_safe_chunk(ref_slice.shape[0], int_slice.shape[0]),
                                  cooldown_ms=self.cooldown_ms)

    def wass_cols(self, X_ref, target_cols, int_slice):
        ref_slice = self._resident(X_ref).index_select(1, self._cols(target_cols))
        return list(K.wasserstein1d_gpu(ref_slice, int_slice, device=self.device,
                                        chunk=_safe_chunk(ref_slice.shape[0], int_slice.shape[0]),
                                        cooldown_ms=self.cooldown_ms))
