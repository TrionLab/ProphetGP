from __future__ import annotations

import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from typing import Dict, Optional

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
        strategy: str = "best_output",
        target_objectives: Optional[Dict[str, dict]] = None,
        target_names: Optional[list[str]] = None,
    ) -> np.ndarray:
        if surrogate.model is None or surrogate.train_y is None:
            raise RuntimeError("Surrogate model must be trained before suggestion.")
        if strategy not in {"best_output", "best_information"}:
            raise ValueError("strategy must be 'best_output' or 'best_information'")
        if strategy == "best_information":
            return self._suggest_by_information_gain(
                surrogate=surrogate,
                bounds=bounds,
                n_candidates=n_candidates,
            )
        if surrogate.train_y.shape[1] > 1:
            return self._suggest_multiobjective_output(
                surrogate=surrogate,
                bounds=bounds,
                n_candidates=n_candidates,
                target_objectives=target_objectives or {},
                target_names=target_names or [],
            )
        if self.objective == "target":
            return self._suggest_by_target_matching(
                surrogate=surrogate,
                bounds=bounds,
                n_candidates=n_candidates,
            )

        acquisition_objective = None
        if self.objective == "maximize":
            best_f = surrogate.train_y.max().item()
        elif self.objective == "minimize":
            # EI는 최대화 형태이므로, -f(x)를 최대화하는 objective를 사용한다.
            best_f = (-surrogate.train_y).max().item()
            acquisition_objective = GenericMCObjective(lambda Y, X=None: -Y.squeeze(-1))
        else:
            raise ValueError(f"Unsupported objective in EI path: {self.objective}")

        acquisition = qExpectedImprovement(
            model=surrogate.model,
            best_f=best_f,
            objective=acquisition_objective,
        )
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
        # single-output 경로이므로 1차원으로 평탄화해 정렬한다.
        gap = np.abs(np.asarray(mean, dtype=np.float64).reshape(-1) - float(self.target_value))
        best_idx = np.argsort(gap)[:n_candidates]
        return candidate_pool[best_idx]

    def _suggest_by_information_gain(
        self,
        surrogate: GPSurrogate,
        bounds: np.ndarray,
        n_candidates: int,
    ) -> np.ndarray:
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
        _, var = surrogate.predict(candidate_pool)
        info_score = np.asarray(var, dtype=np.float64)
        if info_score.ndim == 2:
            info_score = info_score.sum(axis=1)
        best_idx = np.argsort(-info_score)[:n_candidates]
        return candidate_pool[best_idx]

    def _suggest_multiobjective_output(
        self,
        surrogate: GPSurrogate,
        bounds: np.ndarray,
        n_candidates: int,
        target_objectives: Dict[str, dict],
        target_names: list[str],
    ) -> np.ndarray:
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
        scores = np.zeros(candidate_pool.shape[0], dtype=np.float64)
        for idx, target_name in enumerate(target_names):
            obj_cfg = target_objectives.get(target_name, {})
            objective = obj_cfg.get("objective", self.objective)
            target_value = obj_cfg.get("target_value", self.target_value)
            weight = float(obj_cfg.get("weight", 1.0))
            target_mean = mean[:, idx]
            if objective == "maximize":
                scores += weight * target_mean
            elif objective == "minimize":
                scores += -weight * target_mean
            elif objective == "target":
                if target_value is None:
                    raise ValueError(f"target_value is required for target objective: {target_name}")
                scores += -weight * np.abs(target_mean - float(target_value))
            else:
                raise ValueError(f"Unknown objective: {objective}")
        best_idx = np.argsort(-scores)[:n_candidates]
        return candidate_pool[best_idx]

