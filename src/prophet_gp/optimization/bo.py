from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from typing import Optional

from prophet_gp.models.gp_surrogate import GPSurrogate


class BayesianOptimizer:
    def __init__(
        self,
        objective: str = "maximize",
        n_restarts: int = 10,
        raw_samples: int = 128,
        target_value: Optional[float] = None,
        target_search_size: int = 4096,
    ):
        if objective not in {"maximize", "minimize", "target"}:
            raise ValueError("objective must be 'maximize', 'minimize', or 'target'")
        if objective == "target" and target_value is None:
            raise ValueError("target_value must be set when objective='target'")
        self.objective = objective
        self.target_value = target_value
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples
        self.target_search_size = target_search_size

    def suggest(
        self,
        surrogate: GPSurrogate,
        bounds: np.ndarray,
        n_candidates: int = 1,
    ) -> np.ndarray:
        if surrogate.model is None or surrogate.train_y is None:
            raise RuntimeError("Surrogate model must be trained before suggestion.")
        if self.objective == "target":
            return self._suggest_by_target_matching(
                surrogate=surrogate,
                bounds=bounds,
                n_candidates=n_candidates,
            )

        best_f = surrogate.train_y.max().item()
        if self.objective == "minimize":
            best_f = -best_f

        acquisition = qExpectedImprovement(model=surrogate.model, best_f=best_f)
        bounds_tensor = torch.tensor(bounds, dtype=torch.float64)
        candidates, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=bounds_tensor,
            q=n_candidates,
            num_restarts=self.n_restarts,
            raw_samples=self.raw_samples,
        )
        return candidates.detach().cpu().numpy()

    def _suggest_by_target_matching(
        self,
        surrogate: GPSurrogate,
        bounds: np.ndarray,
        n_candidates: int,
    ) -> np.ndarray:
        if self.target_value is None:
            raise RuntimeError("target_value is required for target-matching suggestion.")

        lower = bounds[0]
        upper = bounds[1]
        if lower.shape != upper.shape:
            raise ValueError("Invalid bounds shape.")

        rng = np.random.default_rng(seed=42)
        candidate_pool = rng.uniform(
            low=lower,
            high=upper,
            size=(self.target_search_size, lower.shape[0]),
        )
        mean, _ = surrogate.predict(candidate_pool)
        gap = np.abs(mean - self.target_value)
        best_idx = np.argsort(gap)[:n_candidates]
        return candidate_pool[best_idx]
