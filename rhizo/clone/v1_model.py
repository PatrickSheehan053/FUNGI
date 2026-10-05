"""
obj_009 — model.py : DNPN (Directed Nonlinear Propagation Network).

Directed, weighted, SEED-ONLY nonlinear message-passing GNN with NO node features and ZERO-PRESERVING
layers, per obj_009_BUILD_SPEC.md §Architecture. The ONLY path from "gene p perturbed" to "gene t changed"
runs through the directed edges; an empty graph (or a perturbed node with no out-edges) yields ŝ_g = 0 for
every non-seeded gene BY CONSTRUCTION. This ψ(0)=0 property is the load-bearing invariant (verification #1).

DELIBERATE, DOCUMENTED DEVIATION from the spec's literal pseudocode:
  The spec writes the message as `φ_msg = W₂·GELU(W₁·[x_s ‖ w])` (concatenate sender state with the edge
  weight). That literal form is NOT zero-preserving — `φ_msg([0 ‖ w]) = W₂·GELU(W₁·[0‖w]) ≠ 0` for w≠0, so a
  zero sender would still emit a nonzero message and EVERY node with any in-edge would become nonzero at hop 1,
  regardless of the seed — violating the spec's own ψ(0)=0 / "0 whenever no walk of length ≤K reaches g"
  invariant (§Architecture, §Testing #1). We therefore use the zero-preserving multiplicative form
      m_st = w · MLP(w · x_s),     MLP = W₂·GELU(W₁·)  (bias-free)
  which keeps the spec's INTENT exactly — the edge weight enters TWICE and NONLINEARLY, the message reads the
  SENDER state + weight only (no receiver leak), it is a genuine learned nonlinear message (not a fixed
  operator + nonlinear head) — while guaranteeing m_st=0 whenever x_s=0. This is the minimal change that makes
  the literal pseudocode satisfy the invariant it is specified to satisfy.

All layers bias-free; LayerNorm affine=False; readout bias-free end-to-end (both W_in AND W_out — the spec
emphasises the output layer, but W_in bias would also break ψ(0)=0, so both are bias-free). Over-smoothing
control: JK (concat all hops, default) + always-on per-hop learned gates + optional teleport (α·x⁰).
Batched over perturbations: state x is (N, B, d); messages are (E, B, d). GPU-resident, num_workers=0.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BiasFreeMLP(nn.Module):
    """W₂·GELU(W₁·x), both bias-free ⇒ MLP(0)=0."""
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class DNPN(nn.Module):
    def __init__(self, d=32, d_hidden=64, K=3, oversmooth="jk", dropout=0.1, teleport_alpha=0.1):
        super().__init__()
        self.d = d; self.K = int(K); self.oversmooth = oversmooth
        self.msg = nn.ModuleList([_BiasFreeMLP(d, d_hidden, d) for _ in range(self.K)])
        self.upd = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.K)])
        self.gate = nn.Parameter(torch.ones(self.K))       # per-hop soft-truncation, init 1
        self.teleport_alpha = float(teleport_alpha)
        feat_dim = d * self.K if "jk" in oversmooth else d
        self.read_in = nn.Linear(feat_dim, d_hidden, bias=False)
        self.read_out = nn.Linear(d_hidden, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, seed_nodes, deltas, src, tgt, w, N):
        """seed_nodes (B,) long (perturbed gene idx, -1 if not a node); deltas (B,) float seed magnitude;
        src,tgt (E,) long; w (E,) float. Returns (B, N) predicted specific residual ŝ."""
        B = seed_nodes.shape[0]; d = self.d; dev = w.device
        x0 = torch.zeros(N, B, d, device=dev, dtype=w.dtype)
        valid = seed_nodes >= 0
        if valid.any():
            bidx = torch.arange(B, device=dev)[valid]
            x0[seed_nodes[valid], bidx, :] = deltas[valid].unsqueeze(-1)   # δ broadcast over d
        x = x0
        wcol = w.view(-1, 1, 1)                                            # (E,1,1)
        hops = []
        for k in range(self.K):
            m = wcol * self.msg[k](wcol * x[src])                          # (E,B,d), zero-preserving
            a = torch.zeros(N, B, d, device=dev, dtype=w.dtype).index_add_(0, tgt, m)
            h = F.gelu(self.upd[k](a))
            h = F.layer_norm(h, (d,))                                      # affine=False ⇒ LN(0)=0
            if "teleport" in self.oversmooth:
                h = h + self.teleport_alpha * x0
            x = torch.clamp(self.gate[k], 0.0, 1.0) * h
            hops.append(x)
        f = torch.cat(hops, dim=-1) if "jk" in self.oversmooth else x      # (N,B,feat)
        f = self.drop(f)
        out = self.read_out(F.gelu(self.read_in(f)))                       # (N,B,1), bias-free ⇒ ψ(0)=0
        return out.squeeze(-1).transpose(0, 1).contiguous()               # (B,N)
