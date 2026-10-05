"""
Verifies the session-3 refinement.py basin-diversity fixes:
  1. min_loss_improvement gates the primary drill's patience reset (was a bare
     1e-8 tolerance -- noise-level wiggles used to fully reset patience).
  2. alt-start candidates are drawn from the untouched diverse_pool (Phase 3 Sobol
     + pre-rounds) instead of the contaminated, progressively-narrowing
     df_all_viable.
  3. best_source tracking + the new "[Multi-start]"/"differs in:"/final-summary
     printouts wire correctly end-to-end with no crash.

Three layers, increasing scope:
  A. Math-level: replay the EXACT round sequence from the real RPE1 log (session 3
     conversation) through old (1e-8) vs new (0.001) threshold logic.
  B. Pool-selection: reproduce the contaminated-pool-crowds-out-diverse-candidate
     bug directly (no drilling dynamics needed) and confirm the fix.
  C. Integration smoke test: run the real run_ml_gmm_refinement with a scripted
     mock evaluator + synthetic df_phase3, confirm it reaches alt-start, correctly
     identifies the diverse candidate, and the new prints/best_source appear.
"""
import io
import sys
import contextlib
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import refinement as R

PARAM_COLS = R.PARAM_COLS  # ["beta","delta","kappa","k_core","lambda","psi","nu"]

n_fail = 0


def check(desc, ok):
    global n_fail
    print(f"{'PASS' if ok else 'FAIL'}: {desc}")
    if not ok:
        n_fail += 1


# ===========================================================================
# A. Math-level: real round sequence, old vs new threshold
# ===========================================================================
print("=== A. Patience-reset threshold: real RPE1 round sequence ===")

# (round_loss) exactly as pasted from the real run in the session-3 conversation
real_losses = [6.669751, 6.67638, 6.65973, 6.64830, 6.64826, 6.64602, 6.64792, 6.64635]
patience = 3


def simulate(losses, threshold, patience, max_rounds=None):
    best = float("inf")
    no_improve = 0
    history = []
    for i, rb in enumerate(losses):
        if max_rounds is not None and i >= max_rounds:
            break
        if rb < best - threshold:
            best = rb
            no_improve = 0
            history.append((i + 1, rb, "improve", no_improve))
        else:
            no_improve += 1
            history.append((i + 1, rb, "no-improve", no_improve))
        if no_improve >= patience:
            history.append((i + 1, rb, "PATIENCE_EXHAUSTED", no_improve))
            break
    return history


old_hist = simulate(real_losses, 1e-8, patience)
new_hist = simulate(real_losses, 0.001, patience)

print("  round  loss       OLD(1e-8)         NEW(0.001)")
for i in range(max(len(old_hist), len(new_hist))):
    o = old_hist[i] if i < len(old_hist) else ("", "", "", "")
    n = new_hist[i] if i < len(new_hist) else ("", "", "", "")
    print(f"  {i+1:2d}     {real_losses[i] if i < len(real_losses) else '':>9}  "
          f"{o[2]:>10} no_imp={o[3]}      {n[2]:>10} no_imp={n[3]}")

# Round 5 (6.64826, delta=0.00004 vs prior best 6.64830) is the noise event.
old_r5 = old_hist[4]
new_r5 = new_hist[4]
check("round 5 (delta=0.00004): OLD treats it as improvement",
      old_r5[2] == "improve")
check("round 5 (delta=0.00004): NEW correctly rejects it as noise",
      new_r5[2] == "no-improve")
check("NEW threshold does not exhaust patience by round 8 in this exact "
      "sequence (round 6's real 0.00224 improvement absorbs it -- confirms "
      "the real log's specific sequence happens to mask the effect, an "
      "honest finding, not the test failing)",
      not any(h[2] == "PATIENCE_EXHAUSTED" for h in new_hist))

# Constructed worst-case: repeated noise-level "improvements" in a row, which
# the real log did NOT happen to contain consecutively -- this is what the
# old threshold actually fails on unboundedly.
print("\n  Constructed repeated-noise sequence (worst case for the old bug):")
noise_losses = [6.80, 6.7999, 6.7998, 6.7997, 6.7996, 6.7995, 6.7994, 6.7993, 6.7992, 6.7991]
old_noise = simulate(noise_losses, 1e-8, patience, max_rounds=10)
new_noise = simulate(noise_losses, 0.001, patience, max_rounds=10)
check("OLD threshold: patience never exhausts in 10 rounds of pure noise "
      f"(ran all {len(old_noise)} rounds, last={old_noise[-1][2]})",
      not any(h[2] == "PATIENCE_EXHAUSTED" for h in old_noise))
check("NEW threshold: patience exhausts by round 4 on the same noise",
      any(h[2] == "PATIENCE_EXHAUSTED" for h in new_noise)
      and new_noise[-1][0] == 4)


# ===========================================================================
# B. Pool-selection: contaminated df_all_viable vs untouched diverse_pool
# ===========================================================================
print("\n=== B. Alt-start candidate pool: contamination crowd-out ===")

rng = np.random.default_rng(0)
lower = np.array([0.5, 0.0, 0.02, 8.0, 0.006, 0.0, 0.0])
upper = np.array([4.0, 3.0, 0.35, 28.0, 0.010, 3.0, 2.0])
hp_range = upper - lower

p_main = lower + 0.5 * hp_range          # "primary basin" anchor
p_alt = p_main.copy()
p_alt[3] = upper[3] - 0.05 * hp_range[3]  # k_core: far from p_main (>20% range)
p_alt[4] = lower[4] + 0.05 * hp_range[4]  # lambda: far from p_main (>20% range)

# The genuinely diverse candidate as it exists in the original Sobol pool
diverse_pool = pd.DataFrame([dict(zip(PARAM_COLS, p_main), utopia_loss=6.90),
                             dict(zip(PARAM_COLS, p_alt), utopia_loss=6.95)])
# Pad with filler so nsmallest's top_fraction has a realistic denominator
filler = pd.DataFrame(
    [dict(zip(PARAM_COLS, lower + rng.random(7) * hp_range), utopia_loss=9.0 + rng.random())
     for _ in range(40)])
diverse_pool = pd.concat([diverse_pool, filler], ignore_index=True)

# Simulate primary-drill contamination: ~50 near-duplicates of p_main, each
# BETTER than p_alt's loss (6.95), as a real drill converging on one basin
# would produce -- this is what df_all_viable looks like after several rounds.
contaminated = pd.concat([diverse_pool, pd.DataFrame(
    [dict(zip(PARAM_COLS, p_main + rng.normal(0, 0.01, 7) * hp_range), utopia_loss=6.85)
     for _ in range(50)])], ignore_index=True)


def find_alt_candidates(pool, best_params, top_fraction=0.05):
    elites = pool.nsmallest(max(int(len(pool) * top_fraction), 20), "utopia_loss")
    cands = []
    for _, row in elites.iterrows():
        p = row[PARAM_COLS].values.astype(np.float64)
        frac_diff = np.abs(p - best_params) / np.maximum(hp_range, 1e-12)
        if np.any(frac_diff > 0.20):
            cands.append(float(row["utopia_loss"]))
    return cands


old_cands = find_alt_candidates(contaminated, p_main)   # OLD: from df_all_viable-style pool
new_cands = find_alt_candidates(diverse_pool, p_main)    # NEW: from the snapshot

check("OLD behavior (selecting from the contaminated pool): the diverse "
      f"candidate (loss=6.95) is crowded out of the elite set -- found "
      f"{len(old_cands)} diverse candidates (expected 0)",
      len(old_cands) == 0)
check("NEW behavior (selecting from diverse_pool snapshot): the diverse "
      f"candidate (loss=6.95) is present among {len(new_cands)} candidate(s) "
      "(some random uniform filler also qualifies as 'diverse' in this tiny "
      "42-point synthetic pool -- a realism gap in the test's filler data, "
      "not the production nsmallest+frac_diff logic being checked here; the "
      "point is presence, not exact count)",
      any(abs(c - 6.95) < 1e-9 for c in new_cands))


# ===========================================================================
# C. Integration smoke test: real run_ml_gmm_refinement, scripted evaluator
# ===========================================================================
print("\n=== C. Integration smoke test (real run_ml_gmm_refinement) ===")


class ScriptedEvaluator:
    """Mock evaluator: each .evaluate() call consumes the next scripted loss
    set. Param values come straight from the caller's param_list (real HPs),
    only the loss/shattered columns are scripted -- so alt-start's "differs
    in:" reporting and best_params re-centering exercise real code paths."""

    def __init__(self, mode="organic"):
        self.mode = mode
        self.n_genes = 100
        self._calls = 0
        self.script = []  # list of {"losses": [...]} per call, by order

    def evaluate(self, param_list, chunk_size=None, desc="", show_progress=False):
        param_list = np.asarray(param_list, dtype=np.float64)
        n = len(param_list)
        spec = self.script[self._calls] if self._calls < len(self.script) else {"losses": [9.0]}
        self._calls += 1
        losses = np.full(n, 9.0)
        given = np.asarray(spec["losses"], dtype=np.float64)
        losses[:min(n, len(given))] = given[:min(n, len(given))]
        df = pd.DataFrame(param_list, columns=PARAM_COLS)
        df["utopia_loss"] = losses
        df["is_shattered"] = 0
        df["n_edges"] = 150_000
        return df


df_phase3 = pd.concat([
    pd.DataFrame([dict(zip(PARAM_COLS, p_main), utopia_loss=6.90, is_shattered=0, n_edges=150_000)]),
    pd.DataFrame([dict(zip(PARAM_COLS, p_alt), utopia_loss=6.95, is_shattered=0, n_edges=150_000)]),
    pd.DataFrame([dict(zip(PARAM_COLS, lower + rng.random(7) * hp_range),
                       utopia_loss=9.0 + rng.random(), is_shattered=0, n_edges=150_000)
                  for _ in range(20)]),
], ignore_index=True)

ev = ScriptedEvaluator(mode="organic")
ev.script = [
    {"losses": [9.5, 9.5, 9.5, 9.5]},          # call 0: boundary pre-rounds (n_boundary_probes=4)
    {"losses": [6.80]},                         # call 1: primary round 1 -- real improvement
    {"losses": [6.7999]},                       # call 2: round 2 -- noise (delta 0.0001 < 0.001)
    {"losses": [6.7998]},                       # call 3: round 3 -- noise
    {"losses": [6.7997]},                       # call 4: round 4 -- noise -> patience exhausts here
    {"losses": [6.90]},                         # call 5: alt-start round 1
    {"losses": [6.92]},                         # call 6: alt-start round 2
]

refinement_cfg = {
    "convergence_threshold": 0.0, "density_mode_epsilon": 1e-6,
    "min_zero_for_density": 30, "basin_expansion_rounds": 0,
    "top_fraction": 0.05, "n_gmm_components": 2, "n_samples_per_round": 1,
    "chunk_size": 50, "good_loss_quantile": 0.15, "bad_loss_quantile": 0.70,
    "n_boundary_probes": 4, "run_delta_kcore_grid": False,
    "exhaustion_mode": True, "n_rounds": 3, "patience": 3,
    "min_loss_improvement": 0.001, "n_alt_starts": 2, "alt_patience": 2,
    "max_basins": 2, "cluster_pool_cap": 100, "frontier_shortlist_k": 3,
    "global_improve_patience": 1, "min_fitness_improvement": 0.005,
    "redetection_passes": 0, "region_samples_per_round": 8,
    "trust_region": {}, "surrogate": {"enabled": False},
}

buf = io.StringIO()
crashed = None
try:
    with contextlib.redirect_stdout(buf):
        result_df, _ = R.run_ml_gmm_refinement(
            df_phase3, lower, upper, ev, refinement_cfg, verbose=True)
except Exception as e:
    crashed = e
out = buf.getvalue()

check("run_ml_gmm_refinement completes without raising", crashed is None)
if crashed:
    print(f"  EXCEPTION: {crashed!r}")
    print(out[-2000:])

check("'[Multi-start]' summary printed", "[Multi-start]" in out)
check("alt-start correctly identifies which HPs differ ('differs in:')",
      "differs in:" in out and ("k_core" in out or "lambda" in out))
check("final summary includes best-loss source", "source:" in out)
check("alt-start used the diverse candidate (loss=6.95...), not a "
      "contaminated near-duplicate", "loss=6.95" in out)
check("primary patience-exhausted message mentions trying alternative basins",
      "alternative basin" in out)

print("\n" + ("ALL PASS" if n_fail == 0 else f"{n_fail} CHECK(S) FAILED"))
if n_fail:
    raise SystemExit(1)
