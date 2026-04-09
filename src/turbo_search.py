"""
FUNGI v6 -- TuRBO Refinement (Part D: Trust Region Bayesian Optimization)

Replaces the v5 GMM + Random Forest surrogate with independent, localized
Trust Region Bayesian Optimization (TuRBO) centered on the anchor coordinates
extracted by the spatial niching module.

Each trust region maintains its own local Gaussian Process surrogate and
adaptively expands or contracts based on consecutive successes or failures.

Dependencies:
    - botorch (Meta's Bayesian Optimization library built on GPyTorch)
    - torch
    - gpytorch

If botorch is not available, this module falls back to a simplified local
search using Gaussian perturbation with adaptive step sizes.
"""

import numpy as np
import pandas as pd


def _try_import_botorch():
    """Attempts to import botorch and its dependencies."""
    try:
        import torch
        import botorch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import ExpectedImprovement
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
        return True
    except ImportError:
        return False


BOTORCH_AVAILABLE = _try_import_botorch()


# =========================================================================
# TuRBO State (Per Trust Region)
# =========================================================================

class TrustRegionState:
    """Tracks the state of a single TuRBO trust region.

    Attributes
    ----------
    center : np.ndarray, shape (6,)
        Current best coordinate in this region.
    best_loss : float
        Best utopian loss observed in this region.
    length : float
        Current side-length of the trust region (fraction of total range).
    successes : int
        Consecutive improvements.
    failures : int
        Consecutive non-improvements.
    converged : bool
        Whether the region has been permanently destroyed.
    history_X : list of np.ndarray
        Evaluated coordinates within this region.
    history_y : list of float
        Corresponding loss values.
    """

    def __init__(self, center, best_loss, initial_length=0.05):
        self.center = center.copy()
        self.best_loss = best_loss
        self.length = initial_length
        self.successes = 0
        self.failures = 0
        self.converged = False
        self.history_X = [center.copy()]
        self.history_y = [best_loss]

    def update(self, new_X, new_y, success_threshold=3,
               failure_threshold=5, min_length=0.001):
        """Updates the trust region state after a batch evaluation.

        Parameters
        ----------
        new_X : np.ndarray, shape (batch, 6)
            Evaluated coordinates.
        new_y : np.ndarray, shape (batch,)
            Corresponding loss values.
        """
        for x, y in zip(new_X, new_y):
            self.history_X.append(x.copy())
            self.history_y.append(y)

            if y < self.best_loss:
                self.best_loss = y
                self.center = x.copy()
                self.successes += 1
                self.failures = 0
            else:
                self.failures += 1
                self.successes = 0

            # Expand on consecutive successes
            if self.successes >= success_threshold:
                self.length = min(self.length * 2.0, 1.0)
                self.successes = 0

            # Contract on consecutive failures
            if self.failures >= failure_threshold:
                self.length /= 2.0
                self.failures = 0

            # Check convergence
            if self.length < min_length:
                self.converged = True
                return


# =========================================================================
# Candidate Generation (Local GP or Fallback)
# =========================================================================

def _generate_candidates_gp(state, lower, upper, batch_size=20):
    """Generates candidate coordinates using a local GP surrogate (botorch).

    Fits a SingleTaskGP on the trust region's history, then optimizes
    Expected Improvement within the current bounding box.
    """
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    X_history = np.array(state.history_X)
    y_history = np.array(state.history_y)

    # Normalize to [0, 1] within the trust region bounds
    tr_lower = np.maximum(state.center - state.length * (upper - lower) / 2, lower)
    tr_upper = np.minimum(state.center + state.length * (upper - lower) / 2, upper)
    tr_range = tr_upper - tr_lower
    tr_range = np.where(tr_range < 1e-10, 1e-10, tr_range)

    X_norm = (X_history - tr_lower) / tr_range
    X_norm = np.clip(X_norm, 0.0, 1.0)

    train_X = torch.tensor(X_norm, dtype=torch.float64)
    # Negate loss because botorch maximizes
    train_Y = torch.tensor(-y_history, dtype=torch.float64).unsqueeze(-1)

    if len(train_X) < 3:
        # Not enough data for GP: fall back to random perturbation
        return _generate_candidates_fallback(state, lower, upper, batch_size)

    try:
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        ei = ExpectedImprovement(model=model, best_f=train_Y.max())

        bounds_01 = torch.stack([
            torch.zeros(6, dtype=torch.float64),
            torch.ones(6, dtype=torch.float64)
        ])

        candidates_norm, _ = optimize_acqf(
            acq_function=ei,
            bounds=bounds_01,
            q=batch_size,
            num_restarts=5,
            raw_samples=256,
        )

        candidates = candidates_norm.detach().numpy() * tr_range + tr_lower
        candidates = np.clip(candidates, lower, upper)
        return candidates

    except Exception:
        return _generate_candidates_fallback(state, lower, upper, batch_size)


def _generate_candidates_fallback(state, lower, upper, batch_size=20):
    """Generates candidates via Gaussian perturbation (no botorch required).

    Each candidate is drawn from a normal distribution centered on the
    current trust region center with standard deviation proportional to
    the trust region length.
    """
    total_range = upper - lower
    sigma = state.length * total_range / 4.0

    candidates = np.array([
        state.center + np.random.normal(0, sigma)
        for _ in range(batch_size)
    ])

    candidates = np.clip(candidates, lower, upper)
    return candidates


# =========================================================================
# TuRBO Orchestrator
# =========================================================================

def run_turbo_refinement(anchor_coords, anchor_losses, lower, upper,
                         evaluate_fn, turbo_cfg):
    """Runs the full TuRBO refinement phase across all anchor trust regions.

    Parameters
    ----------
    anchor_coords : np.ndarray, shape (n_anchors, 6)
        Starting coordinates from spatial niching.
    anchor_losses : np.ndarray, shape (n_anchors,)
        Starting loss values.
    lower, upper : np.ndarray, shape (6,)
        Static hyperparameter bounds.
    evaluate_fn : callable
        Function that takes an np.ndarray of shape (batch, 6) and returns
        a list of result dicts (same as run_dash_and_score output).
    turbo_cfg : dict
        The 'turbo' section of the YAML config.

    Returns
    -------
    df_results : pd.DataFrame
        All evaluated coordinates and their scores.
    best_result : dict
        The single best coordinate found across all trust regions.
    """
    max_evals = turbo_cfg["max_evaluations"]
    batch_size = turbo_cfg["batch_size"]
    initial_length = turbo_cfg["initial_length"]
    min_length = turbo_cfg["min_length"]
    success_thresh = turbo_cfg["success_threshold"]
    failure_thresh = turbo_cfg["failure_threshold"]

    # Initialize trust regions
    regions = []
    for i in range(len(anchor_coords)):
        state = TrustRegionState(
            center=anchor_coords[i],
            best_loss=anchor_losses[i],
            initial_length=initial_length
        )
        regions.append(state)

    all_results = []
    total_evals = 0
    round_num = 0

    print(f"TuRBO Refinement: {len(regions)} trust regions, "
          f"budget = {max_evals:,} evaluations.")

    while total_evals < max_evals:
        # Collect active (non-converged) regions
        active = [r for r in regions if not r.converged]
        if not active:
            print("  All trust regions converged.")
            break

        round_num += 1
        round_candidates = []
        region_indices = []

        # Generate candidates from each active region
        for idx, state in enumerate(regions):
            if state.converged:
                continue

            if BOTORCH_AVAILABLE:
                candidates = _generate_candidates_gp(
                    state, lower, upper, batch_size
                )
            else:
                candidates = _generate_candidates_fallback(
                    state, lower, upper, batch_size
                )

            round_candidates.append(candidates)
            region_indices.extend([idx] * len(candidates))

        if not round_candidates:
            break

        batch_params = np.vstack(round_candidates)

        # Evaluate all candidates
        batch_results = evaluate_fn(batch_params)
        all_results.extend(batch_results)
        total_evals += len(batch_results)

        # Update trust region states
        losses_by_region = {}
        coords_by_region = {}
        for j, result in enumerate(batch_results):
            r_idx = region_indices[j]
            if r_idx not in losses_by_region:
                losses_by_region[r_idx] = []
                coords_by_region[r_idx] = []
            losses_by_region[r_idx].append(result["utopia_loss"])
            coords_by_region[r_idx].append(batch_params[j])

        for r_idx in losses_by_region:
            new_X = np.array(coords_by_region[r_idx])
            new_y = np.array(losses_by_region[r_idx])
            regions[r_idx].update(
                new_X, new_y,
                success_threshold=success_thresh,
                failure_threshold=failure_thresh,
                min_length=min_length
            )

        n_active = sum(1 for r in regions if not r.converged)
        best_global = min(r.best_loss for r in regions)

        if round_num % 5 == 0 or n_active <= 5:
            print(f"  Round {round_num}: {total_evals:,}/{max_evals:,} evals | "
                  f"{n_active} active regions | best loss = {best_global:.4f}")

    # Compile final results
    df_results = pd.DataFrame(all_results)

    # Find global best
    best_region = min(regions, key=lambda r: r.best_loss)
    best_result = {
        "center": best_region.center,
        "loss": best_region.best_loss,
    }

    print(f"\nTuRBO complete: {total_evals:,} evaluations across "
          f"{round_num} rounds.")
    print(f"  Best loss: {best_region.best_loss:.6f}")
    print(f"  Converged regions: {sum(1 for r in regions if r.converged)}/{len(regions)}")

    return df_results, best_result
