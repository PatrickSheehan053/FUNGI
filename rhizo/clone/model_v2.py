"""
obj_009.1 — model_v2.py : DNPN-v2. The v1 DNPN backbone VERBATIM + four surgical, invariant-preserving
architecture knobs behind a `variant` switch. When ALL knobs are off (v1_baseline) the forward is
byte-for-byte the v1 model (same module creation order -> same init RNG stream -> faithful reference).

The four knobs (see obj_009.1 build spec §Architecture):
  V-EDGE  use_edge : replace the scalar out-normalized weight w with a graph-derived d-dim per-edge gate
                     a_e = w + edge_gate(phi_e) (edge_gate's last layer is ZERO-INIT, so a_e = w at init ->
                     V-EDGE is a strict generalization of v1 that starts exactly at v1). message
                     m = a_e ⊙ MLP(a_e ⊙ x_s), so m=0 when x_s=0 -> ψ(0)=0 preserved.
  V-FILM  use_film : multiplicative-only FiLM of the JK readout by node context: gamma = 1 + tanh(film(c_g))
                     (film last layer ZERO-INIT -> gamma=1 at init). f' = gamma ⊙ f. NO additive beta (that
                     would break ψ(0)=0). gamma ⊙ 0 = 0 for unreached genes -> ψ(0)=0 preserved.
  V-DIR   use_dir  : Dir-GNN separate in/out aggregation. in-channel = v1 (scatter to tgt from x[src]);
                     out-channel scatters to src from x[tgt] with a SEPARATE bias-free MLP. Combine
                     x = alpha·a_in + (1-alpha)·a_out, alpha = sigmoid(learnable), init 0.5. Both channels
                     zero-preserving -> ψ(0)=0. Direction PRESERVED (two distinct learned channels, never
                     symmetric). NOTE the mandatory reverse/labelperm control (verify_v2): the out-channel can
                     recover the forward direction on a reversed graph — if reverse stops collapsing, V-DIR is
                     a symmetrization-in-disguise and is DISQUALIFIED (reported, not promoted).
  V-TELE  use_tele : teleport / initial-residual h += alpha·x0 each hop (x0 is 0 off the seed -> ψ(0)=0),
                     combined with JK, alpha swept {0.05,0.15,0.30}. Unlocks K>4 where v1 over-smooths.

VARIANT_FEATURES maps a variant name to the four flags; v_full is composed at run time (data-driven).
All layers bias-free; LayerNorm affine=False; readout bias-free end-to-end.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANT_FEATURES = {
    "v1_baseline": dict(edge=False, film=False, dir=False, tele=False),
    "v_edge":      dict(edge=True,  film=False, dir=False, tele=False),
    "v_film":      dict(edge=False, film=True,  dir=False, tele=False),
    "v_dir":       dict(edge=False, film=False, dir=True,  tele=False),
    "v_tele":      dict(edge=False, film=False, dir=False, tele=True),
    # v_full is set at run time from the positive-ΔGAP subset; default = edge+film+tele (dir excluded: risk).
    "v_full":      dict(edge=True,  film=True,  dir=False, tele=True),
}


class _BiasFreeMLP(nn.Module):
    """W2·GELU(W1·x), both bias-free ⇒ MLP(0)=0. zero_last=True zero-inits W2 (identity-preserving residual)."""
    def __init__(self, d_in, d_hidden, d_out, zero_last=False):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_out, bias=False)
        if zero_last:
            nn.init.zeros_(self.w2.weight)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class DNPNv2(nn.Module):
    def __init__(self, d=32, d_hidden=64, K=3, oversmooth="jk", dropout=0.1, teleport_alpha=0.15,
                 variant="v1_baseline", edge_dim=8, ctx_dim=6, alpha_init=0.5, feats=None):
        super().__init__()
        f = dict(feats) if feats is not None else dict(VARIANT_FEATURES[variant])
        self.variant = variant
        self.use_edge = bool(f["edge"]); self.use_film = bool(f["film"])
        self.use_dir = bool(f["dir"]);   self.use_tele = bool(f["tele"])
        self.d = d; self.K = int(K)
        # V-TELE folds teleport into the over-smoothing string (combined with JK, per spec).
        osm = oversmooth
        if self.use_tele and "teleport" not in osm:
            osm = osm + "+teleport"
        self.oversmooth = osm
        self.teleport_alpha = float(teleport_alpha)

        # --- modules created in the v1 order when all knobs are off (faithful v1_baseline init stream) ---
        self.msg = nn.ModuleList([_BiasFreeMLP(d, d_hidden, d) for _ in range(self.K)])          # in-channel
        if self.use_dir:
            self.msg_out = nn.ModuleList([_BiasFreeMLP(d, d_hidden, d) for _ in range(self.K)])   # out-channel
            self.dir_alpha = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(alpha_init)))))
        self.upd = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.K)])
        self.gate = nn.Parameter(torch.ones(self.K))
        if self.use_edge:
            self.edge_gate = _BiasFreeMLP(edge_dim, d_hidden, d, zero_last=True)                  # a_e = w + this
        feat_dim = d * self.K if "jk" in osm else d
        self.feat_dim = feat_dim
        self.read_in = nn.Linear(feat_dim, d_hidden, bias=False)
        self.read_out = nn.Linear(d_hidden, 1, bias=False)
        if self.use_film:
            self.film = _BiasFreeMLP(ctx_dim, d_hidden, feat_dim, zero_last=True)                 # gamma=1+tanh(.)
        self.drop = nn.Dropout(dropout)

    def forward(self, seed_nodes, deltas, src, tgt, w, N, phi_e=None, c_g=None):
        """seed_nodes (B,) long; deltas (B,) float; src,tgt (E,) long; w (E,) float; phi_e (E,edge_dim) or None;
        c_g (N,ctx_dim) or None. Returns (B, N) predicted specific residual."""
        B = seed_nodes.shape[0]; d = self.d; dev = w.device
        x0 = torch.zeros(N, B, d, device=dev, dtype=w.dtype)
        valid = seed_nodes >= 0
        if valid.any():
            bidx = torch.arange(B, device=dev)[valid]
            x0[seed_nodes[valid], bidx, :] = deltas[valid].unsqueeze(-1)
        x = x0

        # per-edge message multiplier: v1 scalar w, or the V-EDGE d-dim gate a_e = w + edge_gate(phi_e)
        if self.use_edge and phi_e is not None and w.numel() > 0:
            a_e = w.view(-1, 1) + self.edge_gate(phi_e)     # (E, d)
            mw = a_e.unsqueeze(1)                            # (E, 1, d) broadcast over batch
        else:
            mw = w.view(-1, 1, 1)                            # (E, 1, 1)

        alpha = torch.sigmoid(self.dir_alpha) if self.use_dir else None
        hops = []
        for k in range(self.K):
            m_in = mw * self.msg[k](mw * x[src])                          # (E,B,d), zero-preserving
            a = torch.zeros(N, B, d, device=dev, dtype=w.dtype).index_add_(0, tgt, m_in)
            if self.use_dir:
                m_out = mw * self.msg_out[k](mw * x[tgt])                 # reverse direction, separate MLP
                a_out = torch.zeros(N, B, d, device=dev, dtype=w.dtype).index_add_(0, src, m_out)
                a = alpha * a + (1.0 - alpha) * a_out
            h = F.gelu(self.upd[k](a))
            h = F.layer_norm(h, (d,))                                     # affine=False ⇒ LN(0)=0
            if "teleport" in self.oversmooth:
                h = h + self.teleport_alpha * x0                          # x0 is 0 off the seed ⇒ ψ(0)=0 kept
            x = torch.clamp(self.gate[k], 0.0, 1.0) * h
            hops.append(x)

        f = torch.cat(hops, dim=-1) if "jk" in self.oversmooth else x     # (N,B,feat)
        if self.use_film and c_g is not None:
            gamma = 1.0 + torch.tanh(self.film(c_g))                       # (N, feat) ∈ [0,2]
            f = gamma.unsqueeze(1) * f                                     # γ⊙0 = 0 for unreached genes
        f = self.drop(f)
        out = self.read_out(F.gelu(self.read_in(f)))                      # bias-free ⇒ ψ(0)=0
        return out.squeeze(-1).transpose(0, 1).contiguous()               # (B,N)
