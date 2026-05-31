from __future__ import annotations

import numpy as np
import torch
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from sklearn.preprocessing import StandardScaler


class GPSurrogate:
    def __init__(self, standardize_targets: bool = False):
        self.standardize_targets = standardize_targets
        self.model: SingleTaskGP | ModelListGP | None = None
        self.train_x: torch.Tensor | None = None
        self.train_y: torch.Tensor | None = None
        self._y_scaler: StandardScaler | None = None
        # sklearn은 모표준편차(ddof=0) 기준 정규화; BoTorch 검증은 torch.std(unbiased=True·ddof=1) 기준.
        self._y_torch_std_adjust: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self._y_scaler = None
        self._y_torch_std_adjust = 1.0
        y_fit = y_arr
        if self.standardize_targets:
            self._y_scaler = StandardScaler()
            y_fit = self._y_scaler.fit_transform(y_arr)
            n = y_fit.shape[0]
            if n > 1:
                self._y_torch_std_adjust = float(np.sqrt((n - 1) / n))
                y_fit = y_fit * self._y_torch_std_adjust

        train_x = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)
        train_y = torch.tensor(y_fit, dtype=torch.float64)
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
        try:
            fit_gpytorch_mll(mll, max_attempts=10)
        except ModelFittingError as exc:
            raise RuntimeError(
                "GP model fitting failed. Set optimization.standardize_gp_inputs and "
                "optimization.standardize_gp_targets to true in config (strongly "
                "recommended for topo_physchem / high-dimensional inputs)."
            ) from exc
        self.model = model
        self.train_x = train_x
        self.train_y = train_y

    def _inverse_scale_posterior(self, mean_z: np.ndarray, var_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._y_scaler is None:
            return mean_z, var_z
        adj = self._y_torch_std_adjust
        mean_sklearn = mean_z / adj
        var_sklearn = var_z / (adj**2)
        mu = np.asarray(self._y_scaler.mean_, dtype=np.float64)
        scale = np.asarray(self._y_scaler.scale_, dtype=np.float64)
        mean_y = mean_sklearn * scale + mu
        var_y = var_sklearn * (scale**2)
        return mean_y, var_y

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)
            if isinstance(self.model, ModelListGP):
                means = []
                variances = []
                for model_i in self.model.models:
                    posterior_i = model_i.posterior(x_tensor)
                    means.append(posterior_i.mean.squeeze(-1).cpu().numpy())
                    variances.append(posterior_i.variance.squeeze(-1).cpu().numpy())
                mean_z = np.stack(means, axis=1)
                var_z = np.stack(variances, axis=1)
            else:
                posterior = self.model.posterior(x_tensor)
                mean_z = posterior.mean.squeeze(-1).cpu().numpy().reshape(-1, 1)
                var_z = posterior.variance.squeeze(-1).cpu().numpy().reshape(-1, 1)
        var_z = np.maximum(var_z, 1e-10)
        return self._inverse_scale_posterior(mean_z, var_z)
