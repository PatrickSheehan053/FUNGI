#!/usr/bin/env bash
# K562 coverage-fixed SPORE — ONE-STOP run (exp_009.1 rebuild).
#
# Everything (coverage force-carry + control-preserving MBK k=2 substrate + train-hybrid firewall subset +
# coverage/firewall verification gate) is now folded INTO SPORE. A single invocation emits the HYPHAE-ready
# substrate. No manual patch scripts, no re-runs, no separate build/verify steps.
#
# Set the 2 vars below, then:  bash run_k562_pipeline.sh
set -euo pipefail

# ===== SET THESE (K562 raw data lives on the HPC) =====================================================
K562_RAW="${K562_RAW:-/PATH/ON/HPC/k562_raw_singlecell.h5ad}"   # the FULL Replogle K562 raw h5ad
PY="${PY:-python}"                                              # your K562 venv's python (deps: requirements.txt)
# =====================================================================================================
PKG="$(cd "$(dirname "$0")" && pwd)"
cd "$PKG"
CFG="configs/spore_light_config_k562_complete.yaml"
OUT="spore_out_k562_complete"            # FRESH dir (never reuse an old spore_out -> stale-checkpoint bug)

echo "[0] project_root=$PKG ; raw=$K562_RAW"
# fill the SET-ME paths in the config in place
sed -i "s#^spore_plus_path:.*#spore_plus_path: $PKG#" "$CFG"
sed -i "s#  project_root:.*#  project_root: $PKG#" "$CFG"
sed -i "s#  raw_h5ad:.*#  raw_h5ad: $K562_RAW#" "$CFG"

# >>> Before running, scale test_n/val_n in the config to the K562 perturbation count (~20% test / ~10% val
#     of SURVIVING perts). RPE1 essential (~1516 perts) used 306/153; genome-wide K562 (~9-10k perts) -> ~2000/~1000.
#     SPORE hard-fails at second zero if test_n+val_n leaves no train perts, so a wrong value stops instantly. <<<

echo "[1] ONE-STOP SPORE run (coverage force-carry -> MBK k=2 substrate -> verification gate)"
$PY spore.py --config "$CFG"

echo ""
echo "DONE. HYPHAE-ready deliverables (in $PKG/$OUT/processed/):"
echo "   K562_essential_allsplits_metacell_ctrlpreserved.h5ad  (all splits, MBK k=2 hybrid — HYPHAE input)"
echo "   K562_essential_train_hybrid.h5ad                       (firewall-safe graph substrate: train+control)"
echo "   COVERAGE_REPORT.md / COVERAGE_REPORT.json              (coverage % + firewall PASS/FAIL)"
echo ""
echo "HYPHAE-ready iff the run exits 0 (the internal gate exits non-zero if coverage < measurable ceiling"
echo "or either firewall fails)."
