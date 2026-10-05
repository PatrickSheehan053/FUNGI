# obj_009.3 — BUILD SPEC: RHIZO-final, the exhaustive final run (real regulatory features + confirmed wideners + its own HPO, built to ship to the HPC)

> **What this is.** The build-ready blueprint for **obj_009.3 (RHIZO-final)** — the last, exhaustive
> analysis+development run for RHIZO before the thesis. It is BUILT on the local 2070 by a Claude Code agent
> and SHIPPED to the Human Technopole SLURM cluster to run on **2–3 V100s via `sbatch`**; it is NOT fully
> testable locally (only the ψ(0)=0 CPU gate + a tiny 1-epoch smoke run). Produced from the curated evidence
> of three arms — the RunPod **5090** results, the HPC **obj_009.1 HPO sweep**, and the HPC **obj_009.2
> strengthen study** — plus a literature pass (LINGER, GTAT-GRN, CausalBench/geneRNIB, TxPert, GNN-FiLM,
> over-squashing/effective-resistance). Build to THIS document.
>
> **Mission.** RHIZO's core claim is won and confirmed: a clean-edge-reading GNN lets a **FUNGI** graph beat a
> greedy `top_weight` prune on aggregate RSC, un-confounded (5090 +0.029→HPO +0.058; strengthen carry
> +0.038/+0.053). The **remaining bosses** are (1) **kNN**, an undirected co-expression graph that beats FUNGI
> on aggregate in-distribution RSC (a documented, field-wide co-expression-baseline phenomenon), and (2)
> **generalization** (a 2nd cell line + the zero-shot/HYPHAE crown jewel). obj_009.3 assembles the confirmed
> wideners, implements — **for real, and purely graph-derived** — the directed higher-order topology features
> (FFLs, directed motifs) that an undirected co-expression graph structurally lacks (NO external/identity
> priors — that would break RHIZO's graph-agnosticism), runs its **own** HPO (because new features move the
> optimal hyperparameters), and settles the kNN question on the axes where causal structure should win
> (direction + transfer), all with mandatory un-confounding controls.
>
> **Evidence base this design is built on (curated, verified):**
> - **5090:** `v_edge_clean` (clean φ_e, degree-blind) is the engine; HPO winner t0023 (fusion d64/dh256/K8)
>   +0.058, tied by pure t0013 (d32/dh64/K12/α0.05) — the portable carry. Attribution: **reciprocity (is_recip)
>   #1, cross_community #2** drive the win; hub2hub/recip_frac are noise. Gauntlet: **kNN beats FUNGI on
>   aggregate RSC** (0.192 vs 0.096) but the gap collapses 5× in transfer and FUNGI wins direction (dnsa).
> - **HPC obj_009.1 HPO (2/48 trials done):** t0001 `v_fuse` (edge_clean+teleport+deepK, d64/dh128/K6/α0.15/
>   gcnii0.1) → **gap +0.047, both metrics lifted (compounding)** — bigger than 5090, one unreplicated draw.
> - **HPC obj_009.2 strengthen (226 cells):** **FiLM readout WIDENS** (+0.014/+0.035, nulls collapse);
>   **over-squash readout WIDENS** (+0.019/+0.011, nulls collapse); **DASH/motif "widens" but is an ARTIFACT**
>   (columns shipped ZERO-FILLED/inert AND V7 φ_e-permutation FAILS −0.38/+0.57) → the real regulatory feature
>   is **UNTESTED**; **`dir` produced no result** (failed/symmetrization-disqualified — drop it); seeds10
>   confirms the carry gap is seed-robust (~0.05+).
>
> **Standing rules (non-negotiable).** Clone-only (never edit production or the HPC's SPORE/SHROOM/FUNGI/
> obj_003/004 in place — copy into `clone/`). Disable-don't-delete. ASK Patrick before deleting ANY file.
> Leakage firewall: only gene IDENTITIES cross; train-only features. Judge by the **GAP**, never the level.
> **Every claimed win must pass the shuffle-collapse AND the V7 φ_e-permutation control** — the strengthen
> study's inert-DASH V7 failure is the cautionary tale. **HPC working style:** the operator pastes commands
> himself — single-line commands, NO heredocs, NO backslash continuations; agent-"created" files don't exist on
> the HPC until pasted/scp'd; NEVER change the SLURM resource footprint (nodes/mem/walltime) without asking.
> Everything obj_009.3 lives under `OBJECTS/obj_009.3_rhizo_final/`.

---

## Purpose and Scope

obj_009.3 builds **RHIZO-final**: (A) the assembled RHIZO model = the confirmed `v_edge_clean` engine + the two
confirmed wideners (**FiLM topology-conditioned readout**, **over-squash-aware readout**) + the optional fusion
depth machinery (teleport/deepK/GCNII), plus (B) **real directed regulatory edge features** — DASH sub-scores,
cisTarget motif NES, and **feed-forward-loop (FFL)/directed-motif participation** — the LINGER-style regulatory
signal an undirected co-expression graph (kNN) structurally cannot have; (C) a **self-contained HPC `sbatch`
package** that first verifies invariants, then runs the assembled model's **own HPO sweep** (seeded from the
best known region but re-searched), then the **full graph-type gauntlet** (FUNGI vs kNN, top_weight, mst, +
nulls) at the HPO winner, with mandatory un-confounding controls and a co-expression-residualized metric; and
(D) a pre-built **zero-shot / held-out-gene harness** ready to fire on the VCC/K562 HYPHAE graphs when they land.

**In scope (v3):** the assembled model + the three new feature families (real, not inert); the self-HPO; the
kNN-inclusive gauntlet; the co-expression-residualized RSC; the zero-shot harness; the SLURM package (2–3 V100s,
overnight); verification (ψ(0)=0 + shuffle + V7 per new feature/variant). **Out of scope:** touching production
SPORE/SHROOM/FUNGI; the HYPHAE dense-graph build itself (separate HPC job); anything requiring >32 GB/GPU.

---

## Interface Design

**Entry points (all under `src/`, cloned/extended from obj_009.2):**
- `model_v3f.py` — `RHIZOFinal`: the obj_009.2 backbone + a `variant`/`feats` switch that composes {edge_clean,
  film, ovsq, deepK/teleport/gcnii} and selects the φ_e profile. All zero-init to the previous variant; all 5
  invariants preserved (ψ(0)=0 absolute; FiLM multiplicative-only; ovsq multiplicative on a zero-preserving
  quantity; teleport/gcnii add α·x⁰ which is 0 off-seed). `dir` retained but DISABLED by default.
- `edge_features_v3f.py` — the φ_e builder with a **`directed_topo` profile** = the confirmed clean columns
  (is_recip, w_recip, cross_community) **+ real, purely graph-derived** FFL/directed-motif participation +
  2-hop reciprocity + community-role columns (NOT zero-filled, NO external priors). Drops the
  attribution-confirmed noise columns (hub2hub, recip_frac_src/tgt).
- `compute_topo_features.py` — computes the directed higher-order topology features **from each arm's own
  directed adjacency only** (pure graph, no external data, no gene identity): FFL/feed-forward + directed
  3-node motif + directed-cycle participation, 2-hop reciprocity, community participation-coefficient. Writes
  per-arm caches. (NO DASH-kernel or motif-database calls — those are forbidden, see Architecture D.)
- `metrics_v3f.py` — obj_009.2 metrics (RSC, dnsa_ge2hop, dsrsc_ge2) + a new **co-expression-residualized RSC**
  (`rsc_coexpr_resid`): correlate prediction with the response residual after removing the component explained
  by a Pearson co-expression baseline; the axis where FUNGI should beat kNN.
- `harness_v3f.py`, `graphs_v3f.py` (arms incl. **kNN as first-class**), `verify_v3f.py`, `hpo_v3f.py`
  (the self-HPO driver + VRAM-packer), `gauntlet_v3f.py`, `zeroshot_v3f.py`, `test_invariant_v3f.py`.
- `configs/obj_009_3.yaml` (base + variant list), `configs/hpo_v3f.yaml` (the self-HPO search space),
  `sbatch_rhizo_final.sh` (the SLURM job), `README_RHIZO_FINAL.md` (for the HPC operator).

**Config (`configs/hpo_v3f.yaml`) — the self-HPO, SEEDED from the best region but RE-SEARCHED:**
```yaml
base:   {arms: [fungi_bio, knn, top_weight, shuffle, reverse, labelperm], rung: [rung2, rung1], seeds: [0,1,2],
         epochs: 130, patience: 12, delta_mode: neg_mu_ctrl}
model:  {variant: assembled}          # edge_clean + film + ovsq ; deepK/teleport swept
search: # centered on the known-good region (v_fuse t0001 + 5090 t0013/t0023) but re-tuned for the new features
  edge_profile: [clean, regulatory]   # A/B: does the REAL regulatory feature set widen the gap vs kNN?
  d: [32, 64] ; d_hidden: [64, 128, 256]
  K: [6, 8, 12] ; teleport_alpha: [0.05, 0.15] ; gcnii_beta: [0.0, 0.1]
  film: [true] ; ovsq: [true]         # confirmed wideners ON; ablate one-off if budget allows
  lr: [1.0e-3, 2.0e-3] ; weight_decay: [1.0e-4]
  batch_perts: [16]
sampler: {method: random, n_trials: 40, seed: 200}   # seed 200 ≠ any prior draw (no overlap)
objective: gap_vs_topweight            # primary; ALSO report gap_vs_knn + rsc_coexpr_resid + dnsa_ge2hop
hardware: {gpus: [0,1,2], usable_vram_gb: 30, safety_factor: 1.15}   # 2–3 V100s, packed
```

**Outputs:** `results/hpo_leaderboard.csv` + `trials/*.json`; `results/gauntlet.csv` (winner vs all graph types,
both rungs, incl. `rsc_coexpr_resid`); `results/reg_feature_attribution.json` (which new feature carries signal);
`results/verification_v3f.json` (ψ(0)=0 + shuffle + V7 per variant/feature); `results/zeroshot_dryrun.json`;
`results/rhizo_final_report.md`.

---

## Architecture (RHIZO-final)

Keeps the RHIZO backbone + 5 invariants verbatim. The assembled model composes only **confirmed or
real-and-controlled** pieces:

**A. The engine — `v_edge_clean` (unchanged).** Clean φ_e edge gate, degree-blind. The confirmed source of the
FUNGI win (attribution: reciprocity + community-bridge).

**B. Confirmed wideners (bundled ON) —**
- **FiLM readout** (multiplicative-only γ from node graph-context; ψ(0)=0 safe). Strengthen ΔvsCarry
  +0.014/+0.035, nulls collapse.
- **Over-squash-aware readout** (weight JK hop-states by inverse effective resistance; multiplicative on a
  zero-preserving quantity). Strengthen ΔvsCarry +0.019/+0.011, nulls collapse.

**C. Fusion depth machinery (swept) —** teleport + optional GCNII initial-residual + deep K. From the HPC HPO's
`v_fuse` t0001 compounding (+0.047, both lifted). Swept, not assumed.

**D. NEW real directed higher-order TOPOLOGY edge features (the anti-kNN weapon) — PURELY GRAPH-DERIVED, no
external/identity knowledge.** Added to a `directed_topo` φ_e profile, computed **identically from each arm's
own directed graph** (so the graph-swap stays fair):
- **Feed-forward-loop (FFL) participation** — per-edge count of FFLs the edge is in. The canonical directed GRN
  motif; an undirected co-expression graph (kNN) **structurally cannot represent it**.
- **Directed 3-node motif + directed-cycle participation** — other directed-motif memberships kNN lacks.
- **2-hop (higher-order) reciprocity** — extends the attribution-confirmed #1 driver (`is_recip`).
- **Community role** — participation coefficient / bridge role (extends the #2 driver `cross_community`).
These REPLACE the strengthen study's inert zero-filled DASH/motif columns. **CRITICAL INVARIANT NOTE: NO
external/identity priors.** No cisTarget/TF-motif NES, no DASH-as-FUNGI-kernel-output — those break invariant #2
(graph-derived only) and #4 (identical extractor per arm) AND self-defeat the graph-swap argument (a win with
external priors is not a win for topology). **LINGER-style motif priors are deliberately REFUSED** — RHIZO
beats co-expression on directed topology alone. Every feature is credited only if it **passes V7** (permuting
φ_e across edges collapses the gain) — the strengthen DASH arm's V7 failure (−0.38/+0.57) is the failure mode to avoid.

**E. Dropped/disabled —** `dir` (no result / symmetrization risk); `attn` (dead on the sparse subset); the
noise φ_e columns hub2hub/recip_frac_src/tgt (attribution-confirmed inert).

**The metric upgrade —** alongside aggregate RSC, report **co-expression-residualized RSC**: because aggregate
RSC rewards co-expression (why kNN wins; CausalBench/geneRNIB), the residualized metric isolates the causal
component where FUNGI's directed/regulatory structure should beat kNN. This is an ADDED honest axis, not a
replacement.

---

## Implementation Plan (numbered, ordered; built on the 2070, shipped to the HPC)

1. **Scaffold.** Clone obj_009.2 `src/*` + `clone/*` into obj_009.3 `clone/`; reuse the obj_009.2/5090 data +
   graph caches; reuse the offloaded HPC results in `DATA/rhizo_hpc_20260709_1159/` as the design evidence.
   Build `configs/obj_009_3.yaml` + `configs/hpo_v3f.yaml`.
2. **`compute_topo_features.py` — build the REAL directed-topology features** (the load-bearing new code),
   from each arm's own directed adjacency ONLY: FFL/directed-motif + directed-cycle participation, 2-hop
   reciprocity, community participation-coefficient. Pure graph — no external data, no DASH kernel, no motif
   database (those break the invariants). Cache per arm.
3. **`edge_features_v3f.py`** — the `directed_topo` profile (clean columns + the real directed-topology
   families; noise columns dropped). `metrics_v3f.py` — add `rsc_coexpr_resid`.
4. **`model_v3f.py`** — the assembled variant (edge_clean+film+ovsq, deepK/teleport swept). `test_invariant_v3f.py`
   — ψ(0)=0 for the assembled model + every feature profile; **gate everything on it** (CPU, runs locally).
5. **`harness_v3f.py`/`graphs_v3f.py`** — kNN a first-class arm; the gauntlet arm set; leakage asserts.
6. **`verify_v3f.py`** — ψ(0)=0 + empty-collapse + shuffle-collapse + **V7 φ_e-permutation per new feature**
   (the gate that the inert DASH arm failed) + reverse/labelperm collapse.
7. **`hpo_v3f.py`** — the self-HPO (VRAM-packer for 2–3 V100s), objective = gap-vs-top_weight, also logging
   gap-vs-kNN + rsc_coexpr_resid + dnsa. **Seeded from the known-good region, re-searched** (Patrick's rule:
   new features move the optimum, so the prior winners are a starting point, not an inheritance).
8. **`gauntlet_v3f.py`** — at the HPO winner: FUNGI vs {kNN, top_weight, mst, knn-variants} + nulls, both rungs,
   seeds 0–4, all metrics incl. residualized. `reg_feature_attribution` (which new feature carries signal).
9. **`zeroshot_v3f.py`** — the gene-held-out harness (per the obj_009.2 ZERO_SHOT_HARNESS_SPEC): dry-run on the
   current substrate now; wired to fire on the VCC/K562 HYPHAE parquet the instant it lands.
10. **`sbatch_rhizo_final.sh`** — SLURM job for **2–3 V100s** (ask Patrick for the exact count/partition;
    default `--gres=gpu:2 --time=14:00:00 --requeue`), resumable (skip completed trials/cells), `trap summarize
    EXIT`, single-line-friendly launch. **`README_RHIZO_FINAL.md`** for the HPC operator (env: V100 sm_70 →
    torch cu118/cu121, **NEVER cu128**; single-line commands; how to submit/monitor/resume; what to send back).
11. **Local pre-flight (all the 2070 can do):** `ast.parse` every script; `test_invariant_v3f.py` PASS; a
    1-epoch CPU smoke of the assembled model on 2 arms. Then STOP — report the package is ready to scp to the HPC.

---

## Data and Dependencies

- **Python:** torch (HPC V100 = sm_70 → cu118/cu121; the 2070 build/test uses its local torch), numpy, scipy
  (`sparse.csgraph`, effective resistance), pandas, pyarrow, anndata, scikit-learn, networkx + python-louvain,
  pyyaml, joblib.
- **Data (reused):** the 5090/obj_009.2 data + graph caches; the exp_024 File-A prunes + FUNGI champion + the
  kNN/mst/other pruned graphs. The new directed-topology features need NO external data — they are computed
  from each arm's graph. HYPHAE parquets (VCC/K562) when they land → the zero-shot arm.
- **Production scripts CLONED (never edited in place):** obj_009.2 src; obj_008 graph_io/metrics. (NO FUNGI
  DASH kernel or obj_003.1 motif tooling — those would inject non-graph-agnostic signal; deliberately excluded.)
- **Evidence inputs (read-only):** `DATA/rhizo_hpc_20260709_1159/` (HPC HPO + strengthen results),
  `OBJECTS/obj_009.1_topology_aware_gnn_v2/RUNPOD_5090/pulled_results/` (5090 results + journal).

**External resources centered (per design-object):**
- **LINGER** (Nature Biotechnology 2024, `Durenlab/LINGER`) — TF-motif priors as regularization give 4–7× over
  co-expression. Cited as the **ANTI-EXAMPLE RHIZO defines itself against:** LINGER buys accuracy with external
  identity priors; RHIZO refuses them and must win on directed graph topology alone. This contrast is a thesis
  strength, not a method to copy.
- **Network-motif / feed-forward-loop tooling** (`networkx` directed-triad census; PMC8687426 review) — the
  basis for the purely-graph-derived FFL/directed-motif features that kNN structurally cannot represent.

---

## Testing Plan (verification — gate everything)

1. **ψ(0)=0 per variant + feature profile** (`test_invariant_v3f.py`, CPU, runs on the 2070) — EXACTLY 0 for
   empty/unreached; the assembled model + regulatory profile must pass. FiLM/ovsq are the ones to watch.
2. **Empty-collapse; mean-baseline ≤0** on RSC/residualized/dnsa.
3. **Leakage asserts** — gene-disjoint; train-only features (DASH/motif/FFL computed on the train graph only).
4. **Shuffle/reverse/labelperm collapse** per variant.
5. **V7 φ_e-permutation per NEW regulatory feature** — permuting φ_e across edges MUST collapse the gain toward
   the scalar baseline. **This is the gate the inert DASH arm failed; any regulatory feature that fails V7 is
   reported as an artifact, not a win.**
6. **kNN sanity** — kNN is a real arm and must reproduce the 5090 gauntlet's kNN>FUNGI-on-aggregate result
   before any residualized/transfer claim is trusted.

---

## Success Criteria (the final-run gate)

- **Verification passes** per graduating variant (ψ(0)=0 absolute; shuffle + V7 collapse).
- **FUNGI beats top_weight** decisively, both rungs, un-confounded (confirm the standing result at the self-HPO
  winner).
- **The regulatory features earn their place IFF** they widen the gap (ΔGAP>0 vs the assembled carry) AND pass
  V7 — otherwise reported as a clean negative (kNN's co-expression advantage on aggregate stands, and the
  honest RHIZO case vs kNN is direction + transfer + residualized RSC).
- **kNN adjudicated on all axes:** aggregate (kNN's turf — flip is a bonus, not required), **residualized RSC +
  dnsa_ge2hop (FUNGI's expected win)**, and the **zero-shot/HYPHAE transfer** (the decisive test, when it lands).
- **Deliverable:** the self-HPO winner config + the assembled model + the gauntlet table + the honest kNN read,
  packaged as the thesis's final RHIZO result.

---

## Promotion Path

obj_009.3 is the terminal RHIZO evaluation instrument for the thesis; no production pipeline change. On success,
the self-HPO winner + assembled model + the gauntlet/zero-shot tables become the Results-chapter RHIZO evidence,
and the winning config is the frozen instrument for the HYPHAE zero-shot crown jewel (VCC first, K562 if it
lands). Lives at `OBJECTS/obj_009.3_rhizo_final/`.

---

## Compute + shipping (2–3 V100 sbatch; built on the 2070, run on the HPC, results tomorrow)

Built + ψ(0)=0-gated + 1-epoch-smoked on the 2070 (all it can do), then scp'd to the HPC and run as a single
`sbatch` on **2–3 V100s** (Patrick sets the exact count — do NOT hardcode a footprint change). The self-HPO
(~40 trials, packed 2–3-up) + gauntlet + zero-shot dry-run fit an overnight wall; resumable/requeue-safe so a
wall-kill loses nothing. HYPHAE-abort: when a VCC/K562 HYPHAE parquet lands, the zero-shot arm fires as the
headline.

---

## Rejected / disabled
- **External/identity edge priors — FORBIDDEN (invariant #2 + #4):** cisTarget/TF-motif NES, LINGER-style
  regulatory priors, DASH-as-FUNGI-specific-kernel-output, any GO/ontology/gene-embedding channel. They break
  RHIZO's graph-agnosticism AND self-defeat the graph-swap argument (a win with external biology is not a win
  for topology). The anti-kNN signal comes ONLY from directed graph topology (FFLs) an undirected co-expression
  graph lacks.
- **`dir`** (no result, symmetrization risk), **`attn`** (dead on sparse subset), **noise φ_e columns**
  (hub2hub/recip_frac — attribution-inert). **Inert/zero-filled features** — forbidden; a feature ships real or
  is logged inert, never credited. **Inheriting the prior HPO winner without re-search** — forbidden (new
  features move the optimum). **cu128 on V100** — wrong arch (sm_120); use cu118/cu121.

---

## Critical References
Internal: `DATA/rhizo_hpc_20260709_1159/` (HPC obj_009.1 HPO + obj_009.2 strengthen results);
`OBJECTS/obj_009.1_topology_aware_gnn_v2/RUNPOD_5090/pulled_results/FINDINGS_JOURNAL.md` + leaderboard +
`attribution.json`; `markdowns/objects/obj_009.2_topology_reader_endgame.md`; `.../ZERO_SHOT_HARNESS_SPEC.md`;
`markdowns/handoffs/Handoff - 09JULY2026.md`; obj_009.2 `src/{model_v3,edge_features_v3,attention}.py`.
External: LINGER (Nat Biotech 2024) — TF-motif priors 4–7× over co-expression; CausalBench / geneRNIB (2025) —
co-expression baselines hard to beat in-distribution; GTAT-GRN (Front. Genet. 2025) + signed-directed-graph GCN
(2025) — directed multi-source feature fusion; TxPert (Nat Biotech 2026, arXiv:2505.14919) — causal graphs win
OOD/transfer; GNN-FiLM (Brockschmidt 2020, arXiv:1906.12192) — feature-wise modulation (the FiLM readout);
Di Giovanni 2023 + over-squashing/effective-resistance surveys (2024–25) — the over-squash readout; network-motif
/ feed-forward-loop reviews (PMC8687426) — the directed FFL feature; Ahlmann-Eltze/Huber/Anders (Nat Methods
2025) — the honesty guardrail (structure, not capacity).
