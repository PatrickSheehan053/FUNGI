"""
obj_009.3 (RHIZO-final) — test_invariant_v3f.py : ψ(0)=0 / zero-preservation for the ASSEMBLED model and every
φ_e profile — the ship-critical GATE (must PASS on the 2070 before anything is scp'd to the HPC). Stress design
(harder than obj_009.2's): core weights fully random; the ADDED modules (edge_gate, film, ovsq path, cap,
msg_out, gcnii) randomized to NONZERO and fed random NONZERO φ_e / c_g / hop_w, so any additive or receiver-side
leak (a FiLM β, an ovsq bias, a teleport onto a non-seed) would light up an unreached NON-SEED gene and FAIL.

Covers: assembled (directed_topo AND clean profiles), the widener ablations (assembled_nofilm/noovsq/nofusion/
engine), and the inherited edge_clean/ovsq/film variants + capacity_match. FiLM and ovsq are the ones to watch.
CPU-only.
"""
import os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
from model_v3f import RHIZOFinal, VARIANT_FEATURES_V3F, make_feats
import edge_features_v3f as EF3F

N, d = 12, 8
HEADS, CTX_DIM = 2, 6

# (variant, profile) cases — the assembled model on BOTH profiles + widener ablations + inherited + capacity
CASES = [
    ("assembled", "directed_topo"), ("assembled", "clean"),
    ("assembled_nofilm", "directed_topo"), ("assembled_noovsq", "directed_topo"),
    ("assembled_nofusion", "directed_topo"), ("assembled_engine", "directed_topo"),
    ("v_edge_clean", "clean"), ("v_edge_clean_ovsq", "clean"), ("v_edge_clean_film", "clean"),
    ("v1_baseline", "full"), ("capacity_match", "full"),
]


def _edge_dim(profile):
    if profile in ("directed_topo", "regulatory"):
        return EF3F.profile_dim(profile, with_dash=True)
    if profile == "clean":
        return 6
    return 8


def _build(variant, profile, Kv):
    torch.manual_seed(0)
    edim = _edge_dim(profile)
    cap = (variant == "capacity_match")
    feats = make_feats("assembled", profile=profile) if variant == "assembled" else (
        None if cap else make_feats(variant, profile=profile))
    kw = dict(d=d, d_hidden=16, K=Kv, oversmooth="jk", dropout=0.0, teleport_alpha=0.2,
              edge_dim=edim, ctx_dim=CTX_DIM, attn_heads=HEADS)
    if cap:
        mdl = RHIZOFinal(variant="v1_baseline", capacity_match=True, **kw)
    else:
        gcnii = 0.3 if feats.get("deepK") else 0.0
        mdl = RHIZOFinal(variant="v1_baseline", feats=feats, gcnii_beta=gcnii, **kw)
    mdl.eval()
    with torch.no_grad():
        for name, pm in mdl.named_parameters():
            if name == "gate":
                pm.copy_(torch.rand_like(pm) * 0.5 + 0.5)
            elif name == "dir_alpha":
                pm.zero_()
            elif any(t in name for t in ["edge_gate.w2", "film.w2", "cap.w2", "attn_key.w2"]):
                pm.copy_(torch.randn_like(pm) * 0.5)     # NONZERO added-module output (stress)
            else:
                pm.copy_(torch.randn_like(pm))
    return mdl, edim


def _pred(mdl, edim, src, tgt, w, seeds, deltas):
    g = torch.Generator().manual_seed(1)
    E = len(src)
    phi = torch.randn(E, edim, generator=g) if E > 0 else torch.zeros(0, edim)
    c = torch.randn(N, CTX_DIM, generator=g)
    hop = torch.rand(N, generator=g) * 0.9 + 0.1        # NONZERO positive per-node ovsq weight (stress)
    st = torch.tensor(src, dtype=torch.long); tt = torch.tensor(tgt, dtype=torch.long)
    wt = torch.tensor(w, dtype=torch.float32)
    s = torch.tensor(seeds, dtype=torch.long); dl = torch.tensor(deltas, dtype=torch.float32)
    with torch.no_grad():
        return mdl(s, dl, st, tt, wt, N, phi_e=phi, c_g=c, hop_w=hop).numpy()


def _max_excl(out, seed):
    return float(np.max(np.abs(out[[g for g in range(len(out)) if g != seed]])))


def run_one(variant, profile):
    mdl, edim = _build(variant, profile, 3)
    r = {}
    r["empty_max_abs"] = _max_excl(_pred(mdl, edim, [], [], [], [3], [1.7])[0], 3)
    src = [5, 6, 6]; tgt = [6, 7, 8]; w = [0.9, 0.8, 0.7]
    r["no_deg_max_abs"] = _max_excl(_pred(mdl, edim, src, tgt, w, [3], [1.5])[0], 3)
    src = [0, 1, 2, 3]; tgt = [1, 2, 3, 4]; w = [0.9, 0.8, 0.7, 0.6]
    out = _pred(mdl, edim, src, tgt, w, [0], [1.0])[0]
    r["reached_max_abs"] = float(np.max([abs(out[g]) for g in [1, 2, 3]]))
    r["unreached_max_abs"] = float(np.max([abs(out[g]) for g in [4, 9, 10, 11]]))
    mdl2, _ = _build(variant, profile, 2)
    o2 = _pred(mdl2, edim, src, tgt, w, [0], [1.0])[0]
    r["K2_node3_abs"] = float(abs(o2[3])); r["K2_node2_abs"] = float(abs(o2[2]))
    ok = (r["empty_max_abs"] < 1e-6 and r["no_deg_max_abs"] < 1e-6 and r["unreached_max_abs"] < 1e-6
          and r["reached_max_abs"] > 1e-6 and r["K2_node3_abs"] < 1e-6 and r["K2_node2_abs"] > 1e-6)
    return ok, r


def run():
    results = {}; allok = True
    for v, prof in CASES:
        ok, r = run_one(v, prof)
        key = f"{v}[{prof}]"
        results[key] = dict(passed=bool(ok), **{k: round(x, 3) for k, x in r.items()})
        allok = allok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {key:28s} unreached={r['unreached_max_abs']:.2e} "
              f"empty={r['empty_max_abs']:.2e} K2_node3={r['K2_node3_abs']:.2e} reached={r['reached_max_abs']:.2e}")
    assert allok, "ψ(0)=0 invariant FAILED for at least one RHIZO-final case"
    return results


if __name__ == "__main__":
    run()
