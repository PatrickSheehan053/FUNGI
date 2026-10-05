"""
obj_009.1 — spill_probe.py : per-variant VRAM + timing probe on the 2070 (8.59 GB). For each variant, trains a
few epochs on the fungi_bio graph at the base batch, measures peak reserved VRAM and per-epoch wall time, and
if it SPILLS (peak reserved above the usable ceiling) drops the batch until spill-free — writing the resulting
variant_batch_cap to intermediate/spill_caps.json (which ablation_v2 then honors). Also doubles as the GPU
smoke test for the model+harness+feature path. SPILL-FREE rule: never run a spilling combo in the real grid.
"""
from __future__ import annotations
import os, sys, json, time, argparse
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "clone"))
import graphs_v2 as G
import harness_v2 as H
import graph_caches as GC
from model_v2 import VARIANT_FEATURES
import yaml

INTER = os.path.join(HERE, "..", "intermediate")
CFG = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "obj_009_1.yaml")))
FROZEN = dict(CFG["frozen_hp"]); FROZEN["teleport_alpha"] = CFG["teleport"]["alpha"]
USABLE_GB = float(CFG["spill_guard"]["probe_reserved_gb"])   # 8.0 GB usable ceiling (spill above this)


def log(m): print(f"[probe {time.strftime('%H:%M:%S')}] {m}", flush=True)


def probe_once(variant, K, batch, D, eval_ix, epochs=4):
    feats = dict(VARIANT_FEATURES[variant])
    phi = GC.load_edge_feat("fungi_bio") if feats["edge"] else None
    ctx = GC.load_node_ctx("fungi_bio") if feats["film"] else None
    graph = G.build_arm("fungi_bio", D["g2i"], D["N"])[:3]
    cfg = dict(FROZEN, K=K, batch_perts=batch, epochs=epochs, patience=epochs + 1)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    H.train_one(graph, D["pidx"][eval_ix], D["s_full"][eval_ix], D["pidx"][eval_ix][:32], D["s_full"][eval_ix][:32],
                D["signal_mask"], D["mu_ctrl"], cfg, seed=0, variant=variant, feats=feats, phi_e=phi, c_g=ctx)
    secs = time.time() - t0
    peak = torch.cuda.max_memory_reserved() / 1e9
    per_ep = secs / epochs
    return peak, per_ep


def find_safe(variant, K, start, D, eval_ix, report):
    """Ladder DOWN from `start` (never above -> never triggers a known spill). Returns first spill-free batch."""
    ladder = [b for b in [16, 14, 12, 10, 8, 6, 4] if b <= start]
    for cand in ladder:
        peak, per_ep = probe_once(variant, K, cand, D, eval_ix)
        spill = peak > USABLE_GB
        log(f"{variant:12s} K{K} batch={cand:2d} peak={peak:.2f}GB per_ep={per_ep:.1f}s {'SPILL' if spill else 'ok'}")
        report.setdefault(variant, {})[f"K{K}_b{cand}_gb"] = round(peak, 2)
        report[variant][f"K{K}_b{cand}_per_ep_s"] = round(per_ep, 1)
        if not spill:
            return cand
    return 0  # 5090-only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(CFG["variants"]))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        log("CUDA not available — abort probe"); sys.exit(1)
    log(f"device={torch.cuda.get_device_name(0)} total={torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB usable={USABLE_GB}GB")
    D = G.load_data()
    eval_ix = np.where(D["pidx"] >= 0)[0]
    prov = CFG["spill_guard"]["variant_batch_cap"]
    base = int(CFG["spill_guard"]["base_batch"])
    b6base = int(CFG["spill_guard"]["batch_by_K"].get(6, CFG["spill_guard"]["batch_by_K"].get("6", 8)))
    caps4 = {}; caps6 = {}; report = {}
    for variant in args.variants.split(","):
        start4 = min(base, prov.get(variant, base))
        c4 = find_safe(variant, 4, start4, D, eval_ix, report)
        caps4[variant] = c4
        start6 = min(b6base, prov.get(variant, base), c4 if c4 else b6base)
        c6 = find_safe(variant, 6, start6, D, eval_ix, report)
        caps6[variant] = c6
        log(f"{variant:12s} -> cap4={c4} cap6={c6}")
    os.makedirs(INTER, exist_ok=True)
    json.dump(dict(variant_batch_cap=caps4, variant_batch_cap_k6=caps6, report=report, usable_gb=USABLE_GB),
              open(os.path.join(INTER, "spill_caps.json"), "w"), indent=2)
    log(f"wrote spill_caps.json  cap4={caps4} cap6={caps6}")


if __name__ == "__main__":
    main()
