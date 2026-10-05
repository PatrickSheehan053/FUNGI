"""
obj_009 — test_invariant.py : the ψ(0)=0 / zero-preservation invariant (verification #1, run BEFORE anything).
A gene g gets a nonzero prediction ONLY if a directed walk of length ≤K reaches it from the perturbed node.
Empty graph / no-out-degree perturbed node ⇒ exactly 0 for every non-reached gene. CPU-only.
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model import DNPN

torch.manual_seed(0)


def run():
    N, d, K = 12, 8, 3
    mdl = DNPN(d=d, d_hidden=16, K=K, oversmooth="jk", dropout=0.0).eval()
    # randomize weights away from 0 so a bias/leak would show up; keep the per-hop gate POSITIVE
    # (a negative gate clamps to 0 and legitimately zeros that hop — intended soft-truncation, but it
    #  would hide propagation in this test, so we set gates to a positive value here).
    for pm in mdl.parameters():
        with torch.no_grad():
            pm.copy_(torch.randn_like(pm))
    with torch.no_grad():
        mdl.gate.copy_(torch.rand_like(mdl.gate) * 0.5 + 0.5)   # gates in [0.5, 1.0]

    def pred(src, tgt, w, seeds, deltas):
        src = torch.tensor(src, dtype=torch.long); tgt = torch.tensor(tgt, dtype=torch.long)
        w = torch.tensor(w, dtype=torch.float32)
        s = torch.tensor(seeds, dtype=torch.long); dl = torch.tensor(deltas, dtype=torch.float32)
        with torch.no_grad():
            return mdl(s, dl, src, tgt, w, N).numpy()

    results = {}

    # (1) EMPTY graph -> every gene exactly 0
    out = pred([], [], [], [3], [1.7])
    results["empty_graph_max_abs"] = float(np.max(np.abs(out)))

    # (2) perturbed node with NO out-edges (edges exist elsewhere) -> all genes 0
    src = [5, 6, 6]; tgt = [6, 7, 8]; w = [0.9, 0.8, 0.7]     # component 5->6->{7,8}; node 3 isolated
    out = pred(src, tgt, w, [3], [1.5])                        # seed node 3 (no out-edges)
    results["no_out_degree_max_abs"] = float(np.max(np.abs(out)))

    # (3) reachability chain p=0 -> 1 -> 2 -> 3 (-> 4 is 4 hops, unreachable at K=3); 9,10,11 isolated
    src = [0, 1, 2, 3]; tgt = [1, 2, 3, 4]; w = [0.9, 0.8, 0.7, 0.6]
    out = pred(src, tgt, w, [0], [1.0])[0]                     # (N,)
    reached = {1, 2, 3}                                        # within K=3 from node 0
    unreached = [4, 5, 6, 7, 8, 9, 10, 11]                    # 4 is 4-hops; rest isolated
    results["reached_min_abs"] = float(np.min([abs(out[g]) for g in reached]))
    results["unreached_max_abs"] = float(np.max([abs(out[g]) for g in unreached]))

    # (4) K-hop horizon: with K=2 node 3 (3 hops away) must be exactly 0
    mdl2 = DNPN(d=d, d_hidden=16, K=2, oversmooth="jk", dropout=0.0).eval()
    for pm in mdl2.parameters():
        with torch.no_grad():
            pm.copy_(torch.randn_like(pm))
    with torch.no_grad():
        mdl2.gate.copy_(torch.rand_like(mdl2.gate) * 0.5 + 0.5)
    s = torch.tensor([0], dtype=torch.long); dl = torch.tensor([1.0], dtype=torch.float32)
    st = torch.tensor(src, dtype=torch.long); tt = torch.tensor(tgt, dtype=torch.long)
    wt = torch.tensor(w, dtype=torch.float32)
    with torch.no_grad():
        o2 = mdl2(s, dl, st, tt, wt, N).numpy()[0]
    results["K2_node3_abs"] = float(abs(o2[3]))               # 3 hops away, K=2 -> must be 0
    results["K2_node2_abs"] = float(abs(o2[2]))               # 2 hops away -> nonzero

    ok = (results["empty_graph_max_abs"] < 1e-6 and results["no_out_degree_max_abs"] < 1e-6
          and results["unreached_max_abs"] < 1e-6 and results["reached_min_abs"] > 1e-6
          and results["K2_node3_abs"] < 1e-6 and results["K2_node2_abs"] > 1e-6)
    print("psi(0)=0 / zero-preservation invariant:")
    for k, v in results.items():
        print(f"  {k}: {v:.3e}")
    print(f"PASS: {ok}")
    assert ok, "ψ(0)=0 invariant FAILED — a bias/leak broke zero-preservation"
    return results


if __name__ == "__main__":
    run()
