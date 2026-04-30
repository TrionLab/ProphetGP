from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from prophet_gp.config import AppConfig
from prophet_gp.data.dataset import PreparedDataset, ReactionDatasetService
from prophet_gp.features.gauche_adapter import GaucheFeaturizerRegistry
from prophet_gp.models.gp_surrogate import GPSurrogate
from prophet_gp.optimization.bo import BayesianOptimizer


@dataclass
class TrainingArtifacts:
    x_train: np.ndarray
    y_train: np.ndarray
    feature_names: List[str]


class ProphetGPPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.dataset_service = ReactionDatasetService(config.data)
        self.featurizers = GaucheFeaturizerRegistry()
        self.surrogate = GPSurrogate()
        self.optimizer = BayesianOptimizer(
            objective=config.optimization.objective,
            n_restarts=config.optimization.n_restarts,
            raw_samples=config.optimization.raw_samples,
            target_value=config.optimization.target_value,
            target_search_size=config.optimization.target_search_size,
        )

    def prepare_features(self, prepared: PreparedDataset) -> TrainingArtifacts:
        # 반응물 개수는 가변이므로, 반응물 분자 피처를 평균 pooling해 고정 길이 벡터로 만든다.
        reactant_vectors = []
        for reactant_smiles in prepared.resolved_smiles:
            per_molecule = self.featurizers.featurize(
                reactant_smiles, name=self.config.featurization.featuriser
            )
            pooled = per_molecule.mean(axis=0)
            reactant_vectors.append(pooled)
        x_mol = np.vstack(reactant_vectors)

        transformer = self.dataset_service.build_condition_transformer(prepared.schema)
        cond_df = prepared.frame[prepared.schema.condition_columns]
        if prepared.schema.condition_columns:
            x_cond = transformer.fit_transform(cond_df)
            if hasattr(x_cond, "toarray"):
                x_cond = x_cond.toarray()
            x = np.concatenate([x_mol, np.asarray(x_cond, dtype=np.float32)], axis=1)
        else:
            x = x_mol

        y = prepared.y.to_numpy(dtype=np.float32)
        feature_names = [f"x{i}" for i in range(x.shape[1])]
        return TrainingArtifacts(x_train=x, y_train=y, feature_names=feature_names)

    def train_from_csv(self, data_path: str) -> TrainingArtifacts:
        prepared = self.dataset_service.load_csv(data_path)
        artifacts = self.prepare_features(prepared)
        self.surrogate.fit(artifacts.x_train, artifacts.y_train)
        return artifacts

    def suggest_next_experiments(self, artifacts: TrainingArtifacts, n_candidates: int) -> np.ndarray:
        x = artifacts.x_train
        bounds = np.vstack([x.min(axis=0), x.max(axis=0)])
        return self.optimizer.suggest(self.surrogate, bounds=bounds, n_candidates=n_candidates)
