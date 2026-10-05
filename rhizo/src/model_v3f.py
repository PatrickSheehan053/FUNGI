"""
obj_009.3 (RHIZO-final) — model_v3f.py : the ASSEMBLED RHIZO instrument.

RHIZOFinal == the obj_009.2 DNPN-v3 backbone (imported VERBATIM from clone/model_v3.py — the strengthen-study
version with the ovsq/hop_w readout) composed into the single assembled model the spec calls for:

    v_edge_clean ENGINE  (clean edge gate, degree-blind — the confirmed source of the FUNGI win)
  + FiLM readout         (multiplicative γ = 1+tanh(MLP(c_g)) from graph-derived node context; ψ(0)=0 safe)   [confirmed widener]
  + over-squash readout  (JK hop-states × per-node inverse-eff-resistance weight; multiplicative on a           [confirmed widener]
                          zero-preserving quantity)
  + fusion depth         (teleport + optional GCNII initial-residual + deep K)                                  [swept, not assumed]

`dir` and `attn` are RETAINED in the backbone but DISABLED (dir: no HPC result / symmetrization risk; attn: dead
on the sparse subset). The φ_e profile is `directed_topo` (a.k.a. `regulatory`) by default; the HPO A/B toggles
it against `clean`. Every one of the 5 hard invariants is inherited unchanged — above all ψ(0)=0 ABSOLUTE
(FiLM multiplicative-only, ovsq multiplicative on a zero-preserving quantity, teleport/gcnii add α·x⁰ which is 0
off-seed, the edge gate/attn/film/cap all zero-init their last layer). test_invariant_v3f.py gates every profile.
"""
from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clone"))
from model_v3 import DNPNv3, VARIANT_FEATURES_V3, _BiasFreeMLP   # noqa: F401  (backbone verbatim from the clone)

# RHIZOFinal is the backbone itself — the assembly is a FEATURE COMPOSITION, not a new class, so ψ(0)=0 and the
# other invariants are inherited exactly (no new forward path to re-verify beyond the profile check).
RHIZOFinal = DNPNv3

# The assembled feature set: edge_clean engine + FiLM + over-squash + fusion machinery (teleport/deepK; gcnii/
# teleport_alpha/K swept via cfg). dir/attn OFF. profile defaults to directed_topo (overridable per run).
_ASSEMBLED = dict(edge=True, film=True, dir=False, tele=True, attn=False, deepK=True, ovsq=True,
                  profile="directed_topo")

VARIANT_FEATURES_V3F = dict(VARIANT_FEATURES_V3)
VARIANT_FEATURES_V3F.update({
    "assembled":          dict(_ASSEMBLED),                                   # the RHIZO-final model
    # one-off ABLATIONS of the assembled model (each drops exactly one widener) — for the ablate-if-budget arm
    "assembled_nofilm":   dict(_ASSEMBLED, film=False),
    "assembled_noovsq":   dict(_ASSEMBLED, ovsq=False),
    "assembled_nofusion": dict(_ASSEMBLED, tele=False, deepK=False),          # engine + readouts, shallow
    "assembled_engine":   dict(edge=True, film=False, dir=False, tele=False, attn=False, deepK=False,
                               ovsq=False, profile="directed_topo"),          # pure v_edge_clean on directed_topo
})


def make_feats(variant="assembled", profile=None, film=None, ovsq=None):
    """Resolve the feature dict for a variant, optionally overriding the φ_e profile and toggling FiLM/ovsq
    (used by the HPO A/B `edge_profile` axis and the widener ablations). Everything else stays as the variant's."""
    f = dict(VARIANT_FEATURES_V3F[variant])
    if profile is not None:
        f["profile"] = profile
    if film is not None:
        f["film"] = bool(film)
    if ovsq is not None:
        f["ovsq"] = bool(ovsq)
    return f


if __name__ == "__main__":
    import torch
    # smoke: assembled model builds on CPU; profile-driven edge_dim; ψ(0)=0 checked properly in test_invariant_v3f
    for prof, edim in [("directed_topo", 9), ("clean", 6)]:
        f = make_feats("assembled", profile=prof)
        m = RHIZOFinal(d=16, d_hidden=16, K=4, variant="assembled", feats=f, edge_dim=edim, ctx_dim=6).eval()
        n_params = sum(p.numel() for p in m.parameters())
        print(f"assembled[{prof}] edge_dim={edim} params={n_params} "
              f"use_edge={m.use_edge} use_film={m.use_film} use_ovsq={m.use_ovsq} "
              f"use_tele={m.use_tele} use_deepK={m.use_deepK} use_dir={m.use_dir} use_attn={m.use_attn}")
