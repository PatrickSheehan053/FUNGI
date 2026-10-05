"""
obj_009.2 — metrics_v3.py : the obj_009.1 scoreboard re-exported UNCHANGED (P1 RSC/DNSA/wr2/bootstrap + the P2
topology-isolating metrics dsRSC_ge2/ge3, reach_at_K, eff_resistance, DNSA_ge2hop). obj_009.2 adds no new
metric — the confirmation battery and the A/B gauntlet reuse these exactly so v2↔v3 comparisons are apples-to-apples.
"""
from __future__ import annotations
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "clone"))
from metrics_v2 import (  # noqa: F401
    rsc_per_pert, dnsa, weighted_r2_per_pert, topk_sign_acc, auprc_per_pert,
    paired_bootstrap, one_sample_bootstrap, wilcoxon_paired, fisher_mean, _pearson,
    dsrsc_per_pert, reach_at_k_per_pert, eff_resistance_per_pert, dnsa_ge2hop_per_pert, true_top_degs,
    UNREACH,
)
