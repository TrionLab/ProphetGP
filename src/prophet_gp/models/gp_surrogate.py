from __future__ import annotations

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood


class GPSurrogate:
    def __init__(self):
        self.model: SingleTaskGP | None = None
        self.train_x: torch.Tensor | None = None
        self.train_y: torch.Tensor | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        train_x = torch.tensor(x, dtype=torch.float64)
        train_y = torch.tensor(y.reshape(-1, 1), dtype=torch.float64)
        model = SingleTaskGP(train_X=train_x, train_Y=train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        self.model = model
        self.train_x = train_x
        self.train_y = train_y

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(torch.tensor(x, dtype=torch.float64))
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            var = posterior.variance.squeeze(-1).cpu().numpy()
        return mean, var
