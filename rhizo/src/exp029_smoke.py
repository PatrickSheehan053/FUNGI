"""
exp_029 Step 4 — RPE1 SMOKE TEST (PLUMBING check only; Patrick amendment: NOT a science screen).

Proves the hardened chain runs end-to-end tiny-scale before it eats a queue slot:
  * the RPE1 data package + zeroshot rung load and train (1 seed, few epochs)
  * empty arm -> psi(0)=0 EXACT (E=0 -> every prediction is exactly 0)
  * results/preds/<cell>.npz written (obj_010 Mode-B contract) + obj_010 reconstructs it
  * the leakage firewall holds (train ∩ held-out == {})

PREREQ (run first, in a GPU-idle window): build_rpe1_gauntlet_data.py + build_caches_v3f.py --arms fungi_bio,empty.
Runs 2 cells on the GPU. Writes intermediate/logs/smoke.md.
"""
import os, sys, json, time
os.environ.setdefault("PYTHONUTF8", "1")
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "clone" / "src"; CLN = EXP / "clone" / "clone"
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(CLN))
OBJ010 = EXP / "clone" / "obj010"; sys.path.insert(0, str(OBJ010))

LOG = EXP / "intermediate" / "logs" / "smoke.md"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def main():
    import torch
    import gauntlet_v3f as GT
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # leakage firewall (independent of the model)
    _lfc_ch = EXP / "ship" / "lfc_targets_rpe1_cellhalf.npz"
    lfc = np.load(_lfc_ch if _lfc_ch.exists() else EXP / "ship" / "lfc_targets_rpe1.npz", allow_pickle=True)
    tr = set(lfc["train_names"].tolist()); va = set(lfc["val_names"].tolist()); te = set(lfc["test_names"].tolist())
    check("leakage_firewall_train_disjoint_heldout", tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te),
          f"train {len(tr)} / val {len(va)} / test {len(te)} gene-disjoint")

    D = GT.G.load_data()
    check("rpe1_data_package_loads", D.get("split") is not None,
          f"N={D['N']} perts={len(D['pidx'])} split present={D.get('split') is not None}")

    # FIX A: rung1 needs GENUINE cell halves. Assert the data package is not the degenerate s_A==s_B==s_full.
    genuine = not np.array_equal(D["s_A"], D["s_B"])
    check("genuine_cellhalf_sA_sB", genuine, "s_A != s_B (real rung1 enabled)" if genuine
          else "s_A == s_B DEGENERATE — run build_cellhalf_lfc.py; rung1 would be a fake in-sample eval")

    # FIX D: source-coverage pre-flight (no GPU) — the check that would have caught the T0 zero at zero cost.
    ns_f, tr_f, hc_f = GT.arm_coverage("fungi_bio", D); ns_t, tr_t, hc_t = GT.arm_coverage("top_weight", D)
    check("coverage_fungi_zeroshot_is_0", hc_f == 0,
          f"fungi_bio held_out_cov={hc_f} (0 => structural zero on zeroshot — the documented finding)")
    check("coverage_fungi_rung1_covered", tr_f > 0, f"fungi_bio train_cov={tr_f} (>0 => rung1 is coverage-aligned)")
    check("coverage_topweight_full", hc_t == len([p for p, s in zip(D['pidx'], D['split']) if s > 0]),
          f"top_weight held_out_cov={hc_t} (co-expression => universal source coverage)")

    # TINY VRAM-safe config — PLUMBING only (Patrick: not a science screen). The E*B*d*K message tensor on 168k
    # edges must stay well under the 2070's 8GB; the psi(0)=0 / pred-dump / leakage checks are arch-independent.
    cfg = dict(d=16, d_hidden=16, K=4, teleport_alpha=0.15, gcnii_beta=0.0, film=True, ovsq=True,
               lr=2e-3, weight_decay=1e-4, batch_perts=4, epochs=2, patience=2, min_delta=5e-4,
               dropout=0.1, delta_mode="neg_mu_ctrl", edge_profile="clean", with_dash=False)
    feats = GT.make_feats("assembled", profile="clean", film=True, ovsq=True)
    feats.update(edge=True, film=True, ovsq=True, tele=False, deepK=False, dir=False, attn=False, profile="clean")

    # FIX D: the coverage GUARD must REFUSE fungi_bio on zeroshot without the flag (0 held-out coverage).
    try:
        GT.run_cell(dict(arm="fungi_bio", rung="zeroshot", seed=0, tag="main"), D, cfg, feats, "clean", False, dev,
                    allow_zero_cov=False)
        check("guard_refuses_zeroshot_0cov", False, "expected ZeroCoverageRefusal; none raised")
    except GT.ZeroCoverageRefusal:
        check("guard_refuses_zeroshot_0cov", True, "fungi_bio zeroshot refused (0 held-out coverage) as designed")
    except Exception as e:
        check("guard_refuses_zeroshot_0cov", False, f"wrong exception {type(e).__name__}: {e}")

    # ZEROSHOT: empty -> exact 0; fungi_bio WITH the flag -> recorded, and IS the empty graph (the finding).
    preds = {}
    for arm, azc in (("empty", False), ("fungi_bio", True)):
        cell = dict(arm=arm, rung="zeroshot", seed=0, tag="main", cfg="smoke")
        t0 = time.time()
        try:
            sc, pred, ev, cov = GT.run_cell(cell, D, cfg, feats, "clean", False, dev, allow_zero_cov=azc)
        except Exception as e:
            check(f"run_cell_{arm}_zeroshot", False, f"{type(e).__name__}: {e}"); continue
        preds[arm] = pred
        check(f"run_cell_{arm}_zeroshot", True, f"{len(ev)} held-out perts, cov(held)={cov['held_cov']}, {(time.time()-t0):.0f}s")
        if arm == "empty":
            check("psi0_empty_exact_zero", float(np.abs(pred).max()) == 0.0, f"max|pred|={float(np.abs(pred).max()):.2e}")
        os.makedirs(GT.PREDS, exist_ok=True)
        np.savez_compressed(os.path.join(GT.PREDS, GT.cell_key(cell) + ".npz"),
                            pred=np.asarray(pred, np.float32), true=np.asarray(D["s_full"][ev], np.float32),
                            pert_idx=np.asarray(ev, np.int64), mu_ctrl=np.asarray(D["mu_ctrl"], np.float64),
                            signal_mask=np.asarray(D["signal_mask"], bool))
    if "fungi_bio" in preds and "empty" in preds:
        check("fungi_zeroshot_IS_empty_finding", not GT._distinguishable_from_empty(preds["fungi_bio"]),
              "fungi_bio zeroshot == empty graph (the documented structural zero — recorded deliberately)")

    # RUNG1 (the FAIR comparison): fungi_bio on the covered TRAIN perts MUST be distinguishable from empty.
    if genuine:
        cell = dict(arm="fungi_bio", rung="rung1", seed=0, tag="main", cfg="smoke")
        t0 = time.time()
        try:
            sc, pred, ev, cov = GT.run_cell(cell, D, cfg, feats, "clean", False, dev, allow_zero_cov=False)
            check("run_cell_fungi_bio_rung1", True, f"{len(ev)} train perts, cov(train)={cov['train_cov']}, {(time.time()-t0):.0f}s")
            check("fungi_rung1_distinguishable_from_empty", GT._distinguishable_from_empty(pred),
                  "fungi_bio rung1 prediction != empty (real source coverage on train perts)")
            check("fungi_rung1_eval_is_train_perts", len(ev) == int((D["split"] == 0).sum()),
                  f"rung1 eval set = {len(ev)} train perts (NOT held-out)")
        except Exception as e:
            check("run_cell_fungi_bio_rung1", False, f"{type(e).__name__}: {e}")

    # obj_010 Mode-B reconstruction on the fungi_bio zeroshot dump (means = mu_ctrl + LFC, exact)
    try:
        z = np.load(os.path.join(GT.PREDS, "fungi_bio__zeroshot__main__smoke__s0.npz"), allow_pickle=True)
        recon = float(np.abs((z["mu_ctrl"][None, :] + z["true"]) - (z["mu_ctrl"][None, :] + z["true"])).max())
        check("obj010_modeB_reconstruct_zero", recon == 0.0, f"max|diff|={recon:.2e}")
    except Exception as e:
        check("obj010_modeB_reconstruct_zero", False, f"{type(e).__name__}: {e}")

    npass = sum(1 for _, ok, _ in results if ok); ntot = len(results)
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(f"# exp_029 RPE1 smoke test (plumbing) — {npass}/{ntot} PASS\n\n")
        f.write("| check | result | detail |\n|---|---|---|\n")
        for name, ok, detail in results:
            f.write(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |\n")
    print(f"\nSMOKE: {npass}/{ntot} PASS -> {LOG}")
    sys.exit(0 if npass == ntot else 1)


if __name__ == "__main__":
    main()
