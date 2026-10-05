"""
obj_009.1 — test_invariant_v2.py : the ψ(0)=0 / zero-preservation invariant for ALL SIX variants
(verification #1, the load-bearing gate — run BEFORE any training). A gene gets a nonzero prediction ONLY if
a directed walk of length ≤K reaches it from the perturbed node (for V-DIR: a walk in either aggregation
channel). Empty graph / no-out-degree perturbed node / unreached gene ⇒ EXACTLY 0.

Stress design: the CORE weights (msg/upd/readout) are fully randomized as in v1; the ADDED modules
(edge_gate, film, msg_out) are randomized to NONZERO values and fed random NONZERO phi_e / c_g, so any additive
term or receiver-side leak in a knob would produce a nonzero prediction on an unreached gene and FAIL. The
dir mixing weight is pinned to alpha=0.5 (both channels live) so the propagation sanity stays meaningful.
CPU-only.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model_v2 import DNPNv2, VARIANT_FEATURES

N, d, K = 12, 8, 3
EDGE_DIM, CTX_DIM = 8, 6


def _build(variant, Kv):
    torch.manual_seed(0)
    mdl = DNPNv2(d=d, d_hidden=16, K=Kv, oversmooth="jk", dropout=0.0, teleport_alpha=0.2,
                 variant=variant, edge_dim=EDGE_DIM, ctx_dim=CTX_DIM).eval()
    with torch.no_grad():
        for name, pm in mdl.named_parameters():
            if name == "gate":
                pm.copy_(torch.rand_like(pm) * 0.5 + 0.5)           # gates in [0.5,1.0] (no legit hop-zeroing)
            elif name == "dir_alpha":
                pm.zero_()                                          # sigmoid(0)=0.5 -> both channels live
            elif "edge_gate.w2" in name or "film.w2" in name:
                pm.copy_(torch.randn_like(pm) * 0.5)                # NONZERO added-module output (stress test)
            else:
                pm.copy_(torch.randn_like(pm))
    return mdl


def _rand_feats(E, seed=1):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(E, EDGE_DIM, generator=g) if E > 0 else torch.zeros(0, EDGE_DIM)
    c = torch.randn(N, CTX_DIM, generator=g)
    return phi, c


def _pred(mdl, src, tgt, w, seeds, deltas):
    src = torch.tensor(src, dtype=torch.long); tgt = torch.tensor(tgt, dtype=torch.long)
    w = torch.tensor(w, dtype=torch.float32)
    s = torch.tensor(seeds, dtype=torch.long); dl = torch.tensor(deltas, dtype=torch.float32)
    phi, c = _rand_feats(len(src))
    with torch.no_grad():
        return mdl(s, dl, src, tgt, w, N, phi_e=phi, c_g=c).numpy()


def _max_excl(out, seed):
    """max |prediction| over all NON-SEEDED genes (the metric excludes the perturbed gene; V-TELE legitimately
    re-injects the seed via the initial residual, and that column is never scored)."""
    idx = [g for g in range(len(out)) if g != seed]
    return float(np.max(np.abs(out[idx])))


def run_one(variant):
    mdl = _build(variant, K)
    r = {}
    # (1) EMPTY graph -> every NON-SEEDED gene exactly 0
    r["empty_max_abs"] = _max_excl(_pred(mdl, [], [], [], [3], [1.7])[0], 3)
    # (2) perturbed node with NO in/out edges (edges elsewhere) -> all non-seeded genes 0
    src = [5, 6, 6]; tgt = [6, 7, 8]; w = [0.9, 0.8, 0.7]
    r["no_deg_max_abs"] = _max_excl(_pred(mdl, src, tgt, w, [3], [1.5])[0], 3)
    # (3) reachability chain 0->1->2->3->4 ; 9,10,11 isolated
    src = [0, 1, 2, 3]; tgt = [1, 2, 3, 4]; w = [0.9, 0.8, 0.7, 0.6]
    out = _pred(mdl, src, tgt, w, [0], [1.0])[0]
    reached = [1, 2, 3]                                    # within K=3 forward
    unreached = [4, 9, 10, 11]                             # 4 is 4-hops; 9/10/11 fully isolated
    r["reached_max_abs"] = float(np.max([abs(out[g]) for g in reached]))     # SOME reached node fires
    r["unreached_max_abs"] = float(np.max([abs(out[g]) for g in unreached]))
    # (4) K-hop horizon with K=2 -> node 3 (3 hops) exactly 0
    mdl2 = _build(variant, 2)
    o2 = _pred(mdl2, src, tgt, w, [0], [1.0])[0]
    r["K2_node3_abs"] = float(abs(o2[3]))
    r["K2_node2_abs"] = float(abs(o2[2]))
    ok = (r["empty_max_abs"] < 1e-6 and r["no_deg_max_abs"] < 1e-6 and r["unreached_max_abs"] < 1e-6
          and r["reached_max_abs"] > 1e-6 and r["K2_node3_abs"] < 1e-6 and r["K2_node2_abs"] > 1e-6)
    return ok, r


def run():
    results = {}; allok = True
    for v in VARIANT_FEATURES:
        ok, r = run_one(v)
        results[v] = dict(passed=bool(ok), **{k: round(x, 3) for k, x in r.items()})
        allok = allok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {v:12s} unreached={r['unreached_max_abs']:.2e} "
              f"empty={r['empty_max_abs']:.2e} K2_node3={r['K2_node3_abs']:.2e} reached={r['reached_max_abs']:.2e}")
    assert allok, "ψ(0)=0 invariant FAILED for at least one variant — an additive/leak term broke zero-preservation"
    return results


if __name__ == "__main__":
    run()
