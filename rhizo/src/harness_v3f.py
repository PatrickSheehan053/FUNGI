"""
obj_009.3 (RHIZO-final) — harness_v3f.py : the trainer. Re-exports the obj_009.2 GPU harness (train_one /
make_delta / _gpu_rsc / _predict) UNCHANGED — it already threads BOTH the FiLM node-context `c_g` and the
over-squash per-node weight `hop_w` end-to-end — and adds ONE shared loader, `arm_inputs`, that assembles the
(φ_e, c_g, hop_w) triple an ASSEMBLED model needs for a given arm.

WHY arm_inputs exists: in the obj_009.2 strengthen study the runner hardcoded `c_g=None`, so the FiLM arm's
readout never actually fired (its "widening" is therefore NOT attributable to FiLM). RHIZO-final wires FiLM for
real — `arm_inputs` loads the graph-derived node context whenever `feats['film']` is on, and the over-squash hop
weight whenever `feats['ovsq']` is on — so every runner (HPO, gauntlet, zero-shot) gets the wiring right by
construction and the confirmed wideners are genuinely exercised. All inputs are GRAPH-DERIVED + train-safe
(no perturbation responses); ψ(0)=0 is preserved because c_g enters multiplicatively and hop_w multiplies a
zero-preserving quantity.
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clone"))
from harness_v3 import train_one, make_delta, _gpu_rsc, _predict   # noqa: F401  (verbatim; threads c_g + hop_w)
import graphs_v3f as G


def arm_inputs(arm, feats, D, profile=None, with_dash=True):
    """Return (phi_e, c_g, hop_w) for `arm` given the model feature dict. Loads only what the model consumes:
      φ_e   if feats['edge'] or feats['attn']  (profile = feats['profile'] unless overridden)
      c_g   if feats['film']                    (graph-derived node context — the FiLM condition)
      hop_w if feats['ovsq']                     (per-node inverse-eff-resistance over-squash weight)."""
    prof = profile if profile is not None else feats.get("profile", "full")
    phi = G.load_edge_feat_profile(arm, prof, D, with_dash=with_dash) if (feats.get("edge") or feats.get("attn")) else None
    c_g = G.load_node_ctx(arm) if feats.get("film") else None
    hop_w = G.load_oversquash_w(arm, D["N"]) if feats.get("ovsq") else None
    return phi, c_g, hop_w
