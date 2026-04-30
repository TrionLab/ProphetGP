from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

from prophet_gp.models.gp_surrogate import GPSurrogate


class BayesianOptimizer:
    def __init__(self, objective: str = "maximize", n_restarts: int = 10, raw_samples: int = 128):
        if objective not in {"maximize", "minimize"}:
            raise ValueError("objective must be 'maximize' or 'minimize'")
        self.objective = objective
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples

    def suggest(
        self,
        surrogate: GPSurrogate,
        bounds: np.ndarray,
        n_candidates: int = 1,
    ) -> np.ndarray:
        if surrogate.model is None or surrogate.train_y is None:
            raise RuntimeError("Surrogate model must be trained before suggestion.")

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
