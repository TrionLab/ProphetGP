from __future__ import annotations

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood


class GPSurrogate:
    def __init__(self):
        self.model: SingleTaskGP | ModelListGP | None = None
        self.train_x: torch.Tensor | None = None
        self.train_y: torch.Tensor | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        train_x = torch.tensor(x, dtype=torch.float64)
        train_y = torch.tensor(y, dtype=torch.float64)
        if train_y.shape[1] == 1:
            model = SingleTaskGP(train_X=train_x, train_Y=train_y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
        else:
            models = [
                SingleTaskGP(train_X=train_x, train_Y=train_y[:, i : i + 1])
                for i in range(train_y.shape[1])
            ]
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        self.model = model
        self.train_x = train_x
        self.train_y = train_y

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float64)
            if isinstance(self.model, ModelListGP):
                means = []
                variances = []
                for model_i in self.model.models:
                    posterior_i = model_i.posterior(x_tensor)
                    means.append(posterior_i.mean.squeeze(-1).cpu().numpy())
                    variances.append(posterior_i.variance.squeeze(-1).cpu().numpy())
                mean = np.stack(means, axis=1)
                var = np.stack(variances, axis=1)
            else:
                posterior = self.model.posterior(x_tensor)
                mean = posterior.mean.squeeze(-1).cpu().numpy().reshape(-1, 1)
                var = posterior.variance.squeeze(-1).cpu().numpy().reshape(-1, 1)
        return mean, var
