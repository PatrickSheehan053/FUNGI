"""
obj_009.2 — model_v3.py : DNPN-v3. The obj_009.1 backbone VERBATIM + Deep-K/teleport/GCNII/DropEdge, Cleaned
edge features, and Attention-over-incoming-edges knobs behind a `variant` switch (superset of obj_009.1's 6
variants). Every knob zero-inits to its predecessor and preserves ALL 5 hard invariants — above all ψ(0)=0.

New vs obj_009.1:
  v_edge_clean : V-EDGE but with the CLEAN φ_e profile (no raw degree/strength) -> the capacity-confound-removed
                 version of the v2 lead (edge_dim = 6).
  v_deepK      : teleport + optional GCNII initial-residual (gcnii_beta) + optional train-only DropEdge, for
                 K in {6,8,10,12}. The un-gameable long-range lever. ψ(0)=0 kept (x⁰ is 0 off the seed).
  v_attn       : V-EDGE + multi-head attention over incoming edges (attention logits from graph-derived φ_e
                 only; zero-preserving via attention.py). Anti-leak V9 in verify_v3.
  v_full3      : the Phase-1 winners fused (composed at run time; default clean+tele+attn).
  capacity_match (flag, not a variant) : v1_baseline + one extra bias-free MLP on the aggregated state that does
                 NOT read φ_e -> pure capacity, for the Phase-0 capacity-matched-baseline test.
"""
from __future__ import annotations
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention import attn_aggregate

# feats: edge, film, dir, tele, attn, deepK, ovsq ; profile selects φ_e ("full" 8 / "clean" 6 / "dashmotif" 8)
VARIANT_FEATURES_V3 = {
    "v1_baseline":  dict(edge=False, film=False, dir=False, tele=False, attn=False, deepK=False, profile="full"),
    "v_edge":       dict(edge=True,  film=False, dir=False, tele=False, attn=False, deepK=False, profile="full"),
    "v_film":       dict(edge=False, film=True,  dir=False, tele=False, attn=False, deepK=False, profile="full"),
    "v_dir":        dict(edge=False, film=False, dir=True,  tele=False, attn=False, deepK=False, profile="full"),
    "v_tele":       dict(edge=False, film=False, dir=False, tele=True,  attn=False, deepK=False, profile="full"),
    "v_edge_clean": dict(edge=True,  film=False, dir=False, tele=False, attn=False, deepK=False, profile="clean"),
    "v_deepK":      dict(edge=False, film=False, dir=False, tele=True,  attn=False, deepK=True,  profile="full"),
    "v_attn":       dict(edge=True,  film=False, dir=False, tele=False, attn=True,  deepK=False, profile="full"),
    "v_full3":      dict(edge=True,  film=False, dir=False, tele=True,  attn=True,  deepK=True,  profile="clean"),
    "v_fuse":       dict(edge=True,  film=False, dir=False, tele=True,  attn=False, deepK=True,  profile="clean"),
    # ── obj_009.2 STRENGTHEN arms — each a variant of the FROZEN carry config (pure v_edge_clean, K12,
    #    d32/dh64, α0.05, clean φ_e). Every one zero-inits to v_edge_clean so ΔGAP is attributable. ──
    # ARM 1 — DASH + motif edge features: clean φ_e with its two currently-zero-filled columns (FUNGI DASH
    #   sub-scores + cisTarget motif NES) WIRED FOR REAL. profile="dashmotif" -> edge_dim 8. Hypothesis: these
    #   are FUNGI-specific signals greedy structurally lacks, so completing them should WIDEN the gap. If the
    #   dash/motif caches are absent, the profile falls back to 2 ZERO columns (== v_edge_clean; gate ignores them).
    "v_edge_clean_dashmotif": dict(edge=True, film=False, dir=False, tele=False, attn=False, deepK=False, profile="dashmotif"),
    # ARM 2 — v_film / v_dir CLOSURE at the carry config (clean profile). Add each lever ON TOP of v_edge_clean.
    #   v_dir requires the reverse-collapse control (if reverse stops collapsing -> symmetrization-in-disguise).
    "v_edge_clean_film":      dict(edge=True, film=True,  dir=False, tele=False, attn=False, deepK=False, profile="clean"),
    "v_edge_clean_dir":       dict(edge=True, film=False, dir=True,  tele=False, attn=False, deepK=False, profile="clean"),
    # ARM 3 — over-squash-aware readout: weight the JK hop-states by inverse effective resistance (per-node,
    #   graph-derived, leakage-free). Multiplicative on a zero-preserving quantity -> ψ(0)=0 holds. `ovsq` on.
    "v_edge_clean_ovsq":      dict(edge=True, film=False, dir=False, tele=False, attn=False, deepK=False, profile="clean", ovsq=True),
    # ARM 4 — bigger-seed confirmation of the carry config itself (no new knob; seeds 0-9, both rungs). Alias of
    #   v_edge_clean so the runner/leaderboard label it distinctly.
    "v_edge_clean_seeds":     dict(edge=True, film=False, dir=False, tele=False, attn=False, deepK=False, profile="clean"),
}


class _BiasFreeMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, zero_last=False):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_out, bias=False)
        if zero_last:
            nn.init.zeros_(self.w2.weight)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class DNPNv3(nn.Module):
    def __init__(self, d=32, d_hidden=64, K=3, oversmooth="jk", dropout=0.1, teleport_alpha=0.15,
                 variant="v1_baseline", edge_dim=8, ctx_dim=6, alpha_init=0.5,
                 gcnii_beta=0.0, dropedge_p=0.0, attn_heads=4, capacity_match=False, feats=None):
        super().__init__()
        f = dict(feats) if feats is not None else dict(VARIANT_FEATURES_V3[variant])
        self.variant = variant
        self.use_edge = bool(f["edge"]); self.use_film = bool(f["film"]); self.use_dir = bool(f["dir"])
        self.use_tele = bool(f["tele"]); self.use_attn = bool(f["attn"]); self.use_deepK = bool(f["deepK"])
        self.use_ovsq = bool(f.get("ovsq", False))   # over-squash-aware readout (inverse-eff-res hop weighting)
        self.profile = f.get("profile", "full")
        self.d = d; self.K = int(K); self.heads = int(attn_heads)
        self.gcnii_beta = float(gcnii_beta); self.dropedge_p = float(dropedge_p)
        self.teleport_alpha = float(teleport_alpha); self.capacity_match = bool(capacity_match)
        assert d % self.heads == 0 or not self.use_attn, "d must be divisible by attn_heads"

        osm = oversmooth
        if (self.use_tele or self.use_deepK) and "teleport" not in osm:
            osm = osm + "+teleport"
        self.oversmooth = osm

        # v1-order modules first (so v1_baseline's init RNG stream is faithful)
        self.msg = nn.ModuleList([_BiasFreeMLP(d, d_hidden, d) for _ in range(self.K)])
        if self.use_dir:
            self.msg_out = nn.ModuleList([_BiasFreeMLP(d, d_hidden, d) for _ in range(self.K)])
            self.dir_alpha = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(alpha_init)))))
        self.upd = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.K)])
        self.gate = nn.Parameter(torch.ones(self.K))
        if self.use_edge:
            self.edge_gate = _BiasFreeMLP(edge_dim, d_hidden, d, zero_last=True)     # a_e = w + this
        if self.use_attn:
            self.attn_key = _BiasFreeMLP(edge_dim, d_hidden, self.heads)             # graph-derived logits
        if self.capacity_match:
            self.cap = _BiasFreeMLP(d, d_hidden, d, zero_last=True)                  # pure-capacity, no φ_e
        feat_dim = d * self.K if "jk" in osm else d
        self.feat_dim = feat_dim
        self.read_in = nn.Linear(feat_dim, d_hidden, bias=False)
        self.read_out = nn.Linear(d_hidden, 1, bias=False)
        if self.use_film:
            self.film = _BiasFreeMLP(ctx_dim, d_hidden, feat_dim, zero_last=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, seed_nodes, deltas, src, tgt, w, N, phi_e=None, c_g=None, hop_w=None):
        B = seed_nodes.shape[0]; d = self.d; dev = w.device
        x0 = torch.zeros(N, B, d, device=dev, dtype=w.dtype)
        valid = seed_nodes >= 0
        if valid.any():
            bidx = torch.arange(B, device=dev)[valid]
            x0[seed_nodes[valid], bidx, :] = deltas[valid].unsqueeze(-1)
        x = x0

        # train-only DropEdge (does NOT touch the eval graph)
        if self.training and self.use_deepK and self.dropedge_p > 0 and w.numel() > 0:
            keep = torch.rand(w.shape[0], device=dev) > self.dropedge_p
            src, tgt, w = src[keep], tgt[keep], w[keep]
            if phi_e is not None:
                phi_e = phi_e[keep]

        if self.use_edge and phi_e is not None and w.numel() > 0:
            a_e = w.view(-1, 1) + self.edge_gate(phi_e)     # (E,d)
            mw = a_e.unsqueeze(1)                            # (E,1,d)
        else:
            mw = w.view(-1, 1, 1)
        score_e = self.attn_key(phi_e) if (self.use_attn and phi_e is not None and w.numel() > 0) else None
        alpha = torch.sigmoid(self.dir_alpha) if self.use_dir else None

        hops = []
        for k in range(self.K):
            m_in = mw * self.msg[k](mw * x[src])                        # (E,B,d), zero-preserving
            if self.use_attn and score_e is not None:
                a = attn_aggregate(m_in, score_e, src, tgt, N, self.heads)
            else:
                a = torch.zeros(N, B, d, device=dev, dtype=w.dtype).index_add_(0, tgt, m_in)
            if self.use_dir:
                m_out = mw * self.msg_out[k](mw * x[tgt])
                a_out = torch.zeros(N, B, d, device=dev, dtype=w.dtype).index_add_(0, src, m_out)
                a = alpha * a + (1.0 - alpha) * a_out
            if self.capacity_match:
                a = a + self.cap(a)                                    # pure capacity, zero-preserving
            h = F.gelu(self.upd[k](a))
            h = F.layer_norm(h, (d,))
            if self.gcnii_beta > 0.0:
                h = (1.0 - self.gcnii_beta) * h + self.gcnii_beta * x0  # GCNII initial-residual (x0=0 off seed)
            elif "teleport" in self.oversmooth:
                h = h + self.teleport_alpha * x0
            x = torch.clamp(self.gate[k], 0.0, 1.0) * h
            hops.append(x)

        f = torch.cat(hops, dim=-1) if "jk" in self.oversmooth else x
        # Over-squash-aware readout: down-weight the JK feature of high-effective-resistance (over-squashed)
        # nodes. hop_w is a per-node graph-derived weight in (0,1]; multiplicative on f, and f is EXACTLY 0 for
        # every unreached/no-out-degree node -> 0·w = 0, so ψ(0)=0 is preserved for any finite hop_w.
        if self.use_ovsq and hop_w is not None:
            f = f * hop_w.view(-1, 1, 1)
        if self.use_film and c_g is not None:
            gamma = 1.0 + torch.tanh(self.film(c_g))
            f = gamma.unsqueeze(1) * f
        f = self.drop(f)
        out = self.read_out(F.gelu(self.read_in(f)))                   # bias-free ⇒ ψ(0)=0
        return out.squeeze(-1).transpose(0, 1).contiguous()
