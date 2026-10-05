"""
obj_009.2 — attention.py : bias-free, GRAPH-DERIVED, zero-preserving multi-head attention over INCOMING edges.

For each target t, its incoming messages (from edges s->t) are combined with a learned convex weight
alpha_{s->t} (softmax over t's incoming edges), instead of a plain sum. The attention logits come ONLY from
graph-derived per-edge features phi_e (NO gene identity, NO per-gene params) via a bias-free MLP, and the
message VALUE still reads the (zero-preserving) sender state m_e = a_e ⊙ MLP(a_e ⊙ x_s). Because every message
value is 0 when its sender is 0, an unreached target aggregates Σ alpha·0 = 0 → **ψ(0)=0 preserved regardless
of alpha**. Multi-head = H independent weightings over disjoint channel slices, concatenated.

Anti-leak (verify_v3 V9): since the attention logits use NO target identity, permuting target identities must
NOT change the aggregation pattern for a fixed graph -> the gain must collapse under target-identity-shuffle,
proving attention reads graph structure, not identity. (Kept static-in-state for v3's first pass; the message
value carries the sender state, so this is attention-over-edge-topology, not a receiver-side identity bypass.)
"""
from __future__ import annotations
import torch


def scatter_softmax(scores, index, N):
    """Segment softmax of `scores` (E,H) over groups given by `index` (E,) target ids, group count N.
    Returns alpha (E,H): for each target, its incoming edges' weights sum to 1 (numerically stable)."""
    E, H = scores.shape
    dev = scores.device
    maxes = torch.full((N, H), -1e30, device=dev, dtype=scores.dtype)
    maxes = maxes.scatter_reduce(0, index.view(-1, 1).expand(-1, H), scores, reduce="amax", include_self=True)
    z = torch.exp(scores - maxes[index])                     # (E,H)
    denom = torch.zeros(N, H, device=dev, dtype=scores.dtype).index_add_(0, index, z)
    return z / (denom[index] + 1e-16)


def attn_aggregate(m_e, score_e, src, tgt, N, heads):
    """Aggregate per-edge messages m_e (E,B,d) into node states a (N,B,d) using multi-head attention over
    INCOMING edges. score_e (E,H) = bias-free graph-derived logits. d must be divisible by heads.
    Zero-preserving: m_e=0 (zero sender) -> a=0 for unreached targets, regardless of the (finite) weights."""
    E, B, d = m_e.shape
    H = heads; dh = d // H
    alpha = scatter_softmax(score_e, tgt, N)                  # (E,H)
    a = torch.zeros(N, B, d, device=m_e.device, dtype=m_e.dtype)
    for h in range(H):
        w = alpha[:, h].view(E, 1, 1)                        # (E,1,1)
        m_h = m_e[:, :, h * dh:(h + 1) * dh] * w             # weight this head's channel slice
        a[:, :, h * dh:(h + 1) * dh].index_add_(0, tgt, m_h)
    return a
