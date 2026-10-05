# obj_009.3 — RHIZO-final (V100 SLURM package, for the HPC operator)

A **self-contained, resumable** SLURM package that runs the exhaustive final RHIZO analysis on **2–3 V100s**.
Built + ψ(0)=0-gated + 1-epoch-CPU-smoked on an RTX 2070 (all it can do); everything else runs here. All data +
graph caches are **BUNDLED** (`data/`, `intermediate/graph_caches/`) — no anndata / parquet / substrate rebuild.

Judge everything by the **GAP = FUNGI(fungi_bio) − arm**, never the absolute level. The primary objective is
**gap vs `top_weight`** (aggregate RSC); the honest boss is **kNN** (a first-class arm), adjudicated on the
axes where causal structure should win: **residualized RSC**, **dnsa_ge2hop** (directed long-range), and the
**zero-shot/HYPHAE transfer** (the decisive test, when it lands).

---

## 0. THE ONE INVARIANT (graph-swap fairness) — do not violate
The SAME assembled model + the SAME feature extractor run on every arm, so any difference is caused by the graph.
- **Allowed** features: anything recomputed uniformly from each arm's OWN directed graph — reciprocity,
  community role, effective resistance, FFL/directed-motif participation, 2-hop reciprocity, recomputed-DASH.
- **Forbidden:** (1) external/identity priors (cisTarget/TF-motif NES, GO, gene embeddings — the LINGER
  approach, refused); (2) FUNGI's *stored* pruning scores (privileged — only FUNGI has them). Every NEW feature
  is credited only if it passes **V7** (φ_e-permutation collapses its gain). A feature whose gain SURVIVES
  permutation is reported as an **artifact, not a win** (the obj_009.2 inert-DASH V7 failure is the cautionary tale).

---

## 1. Environment (READ FIRST — the cu128 trap)
V100 = compute capability **sm_70**. Install a torch build matching the cluster's CUDA:
- CUDA 12.1: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- CUDA 11.8: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- **NEVER cu128** (that targets sm_120 / the 5090; it will NOT run on a V100).

Then `pip install -r requirements.txt` for the rest (numpy/scipy/pandas/pyyaml/scikit-learn/networkx/python-louvain/joblib).
If the cluster provides torch via `module load`, use that and pip-install only the rest.

Operator style (from the standing rules): **paste commands as SINGLE LINES — no heredocs, no backslash
continuations.** Agent-authored files do not exist here until you `scp` this whole folder over.

---

## 2. Submit / monitor / resume
Set the footprint in `sbatch_rhizo_final.sh` (ASK Patrick — do NOT change nodes/mem/walltime on your own): the
`#SBATCH --gres`, `--time`, `--partition`, `--account`, `--constraint` lines are marked **SET ME**. Default is
`--gres=gpu:2 --time=14:00:00 --requeue`.

- Dry-run the plan (no GPU, no SLURM): `bash sbatch_rhizo_final.sh --dry_run`
- Submit: `sbatch sbatch_rhizo_final.sh`   (or `sbatch --gres=gpu:3 sbatch_rhizo_final.sh` for 3 V100s)
- Monitor: `tail -f results/slurm_*.log`   (per-GPU: `tail -f results/gauntlet_worker_0.log`, `results/trial_*.log`)
- **Resume after a wall-kill / requeue:** just `sbatch` again — the HPO skips finished `results/trials/*.json`
  and the gauntlet skips finished `results/cells/*.json`. Nothing is recomputed. `trap summarize EXIT` banks a
  leaderboard + gauntlet table + verification even on preempt/timeout.

The job runs a hard `psi(0)=0` gate FIRST (`src/test_invariant_v3f.py`); if it fails it aborts before training.
It then builds caches idempotently (the O(N^3) Laplacian pinv is the one-time CPU spike — most caches are
already bundled), then Stage 1 self-HPO → Stage 2 gauntlet → Stage 3 zero-shot dry-run → Stage 4 verification.

---

## 3. What runs
- **Stage 1 — self-HPO (`hpo_v3f.py`, config `configs/hpo_v3f.yaml`):** ~40 random trials of the ASSEMBLED
  model (v_edge_clean engine + FiLM + over-squash + swept teleport/deepK/GCNII), SEEDED from the known-good
  region (v_fuse t0001, 5090 t0013) but **RE-SEARCHED** (seed 200; new directed-topology features move the
  optimum). The A/B axis `edge_profile: [clean, directed_topo]` asks whether the real regulatory features widen
  the gap. Objective = gap vs top_weight; also logs gap-vs-kNN, residualized-RSC, dnsa. A VRAM-aware stacker
  packs each V100 (usable 30 GB, safety 1.15). Output: `results/hpo_leaderboard.csv` + `results/trials/*.json`.
- **Stage 2 — gauntlet (`gauntlet_v3f.py`) at the HPO winner:** FUNGI vs {kNN, top_weight, mst} + nulls
  {shuffle, reverse, labelperm, empty}, BOTH rungs, seeds 0–4, all metrics incl. residualized RSC; PLUS
  per-new-feature **attribution** (which directed-topology column carries signal) and **V7** (φ_e-permutation
  per feature). Sharded one worker per GPU. Output: `results/gauntlet.csv` + `results/reg_feature_attribution.json`.
- **Stage 3 — zero-shot dry-run (`zeroshot_v3f.py --dry_run`):** the gene-held-out harness on the current
  substrate — proves the pipeline runs end-to-end, the split is leakage-clean, empty→0, reverse/labelperm→~0.
  Output: `results/zeroshot_dryrun.json`. **When a HYPHAE parquet lands** (RPE1 `fungi_hyphae.parquet` etc. in a
  dir), fire the real crown-jewel test: `python src/zeroshot_v3f.py --hyphae_dir <dir> --device cuda` → `results/zeroshot_hyphae.json`.
- **Stage 4 — verification (`verify_v3f.py`):** ψ(0)=0 + mean-baseline≤0 + cache sha256 + nulls-collapse + V7
  per feature → `results/verification_v3f.json`.

---

## 4. Files to send back to Patrick
`results/hpo_leaderboard.csv`, `results/gauntlet.csv`, `results/reg_feature_attribution.json`,
`results/verification_v3f.json`, `results/zeroshot_dryrun.json` (+ `results/zeroshot_hyphae.json` if HYPHAE fired),
`results/trials/*.json`, `results/*.log`. The whole `results/` tarball is fine.

---

## 5. Success criteria (the final-run gate)
- Verification passes (ψ(0)=0 absolute; shuffle/reverse/labelperm collapse; V7 collapses per feature).
- FUNGI beats top_weight decisively, both rungs, at the self-HPO winner.
- The directed-topology features **earn their place IFF** they widen the gap (ΔGAP>0 vs the clean-profile carry)
  AND pass V7 — else a clean negative (kNN's aggregate co-expression advantage stands; the honest RHIZO case vs
  kNN is direction + transfer + residualized RSC).
- kNN adjudicated on ALL axes: aggregate (kNN's turf — a flip is a bonus, not required), residualized RSC +
  dnsa_ge2hop (FUNGI's expected win), and the zero-shot/HYPHAE transfer (the decisive test, when it lands).

---

## 6. Notes / caveats (honest)
- **FiLM is genuinely wired here.** In the obj_009.2 strengthen study the runner passed `c_g=None`, so the FiLM
  arm's readout never fired — its reported "widening" is therefore NOT attributable to FiLM. RHIZO-final wires
  the node context for real (`harness_v3f.arm_inputs`), so the HPO/gauntlet re-measure FiLM honestly.
- **`dash_recomp`** is a DASH-kernel-style component RECOMPUTED per-arm (effective-resistance × weight × bridge),
  NOT FUNGI's stored scores. It is flagged as a likely-redundant candidate; attribution + V7 decide its fate.
- The whole pipeline is clone-only — it touches no production SPORE/SHROOM/FUNGI/obj_003/004. Sequential by
  default; do not overlap CPU-heavy cache builds with GPU training.
