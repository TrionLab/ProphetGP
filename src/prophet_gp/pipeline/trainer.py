from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from prophet_gp.config import AppConfig
from prophet_gp.data.dataset import PreparedDataset, ReactionDatasetService
from prophet_gp.features.gauche_adapter import GaucheFeaturizerRegistry
from prophet_gp.models.gp_surrogate import GPSurrogate
from prophet_gp.optimization.bo import BayesianOptimizer


@dataclass
class TrainingArtifacts:
    x_train: np.ndarray
    y_train: np.ndarray
    target_columns: List[str]
    feature_names: List[str]
    mol_feature_dim: int
    reactant_inputs: List[str]
    reactant_smiles: List[List[str]]
    condition_columns: List[str]
    condition_types: Dict[str, str]
    condition_transformer: Optional[ColumnTransformer]
    allowed_reactant_vectors: Optional[np.ndarray] = None
    allowed_reactant_inputs: Optional[List[str]] = None
    allowed_reactant_smiles: Optional[List[List[str]]] = None
    gp_input_scaler: Optional[MinMaxScaler] = None
    # GP 표준화 전(분자+조건 결합) 특징; standardize_gp_inputs=True일 때만 설정.
    x_train_pre_gp_scale: Optional[np.ndarray] = None


@dataclass
class SuggestionResult:
    raw_candidates: np.ndarray
    decoded_candidates: List[Dict[str, Any]]


@dataclass
class PredictionResult:
    raw_features: np.ndarray
    predictions: List[Dict[str, Any]]


class ProphetGPPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.dataset_service = ReactionDatasetService(config.data)
        self.featurizers = GaucheFeaturizerRegistry()
        self.surrogate = GPSurrogate(standardize_targets=config.optimization.standardize_gp_targets)
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

        x_pre_gp = np.asarray(x, dtype=np.float32).copy()
        gp_scaler: Optional[MinMaxScaler] = None
        if self.config.optimization.standardize_gp_inputs:
            gp_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
            x = gp_scaler.fit_transform(np.asarray(x_pre_gp, dtype=np.float64)).astype(np.float32)

        y = prepared.y.to_numpy(dtype=np.float32)
        feature_names = [f"x{i}" for i in range(x.shape[1])]
        return TrainingArtifacts(
            x_train=x,
            y_train=y,
            target_columns=prepared.schema.target_columns,
            feature_names=feature_names,
            mol_feature_dim=x_mol.shape[1],
            reactant_inputs=prepared.frame[prepared.schema.reactant_column].astype(str).tolist(),
            reactant_smiles=prepared.resolved_smiles,
            condition_columns=prepared.schema.condition_columns,
            condition_types=prepared.schema.condition_types,
            condition_transformer=transformer if prepared.schema.condition_columns else None,
            gp_input_scaler=gp_scaler,
            x_train_pre_gp_scale=x_pre_gp if gp_scaler is not None else None,
        )

    def train_from_csv(self, data_path: str) -> TrainingArtifacts:
        prepared = self.dataset_service.load_csv(data_path)
        prepared = self.dataset_service.drop_na_training_rows(prepared)
        # print("Dropped NA rows:", prepared.frame)
        artifacts = self.prepare_features(prepared)
        self.surrogate.fit(artifacts.x_train, artifacts.y_train)
        return artifacts

    def suggest_next_experiments(
        self,
        artifacts: TrainingArtifacts,
        n_candidates: int,
        strategy: Optional[str] = None,
    ) -> SuggestionResult:
        x = artifacts.x_train
        bounds = np.vstack([x.min(axis=0), x.max(axis=0)])
        chosen_strategy = strategy or self.config.optimization.suggestion_strategy
        raw_candidates_scaled = self.optimizer.suggest(
            self.surrogate,
            bounds=bounds,
            n_candidates=n_candidates,
            strategy=chosen_strategy,
            target_objectives=self._build_target_objective_map(artifacts),
            target_names=artifacts.target_columns,
        )
        if artifacts.gp_input_scaler is not None:
            cand_pre_gp = artifacts.gp_input_scaler.inverse_transform(
                np.asarray(raw_candidates_scaled, dtype=np.float64)
            ).astype(np.float32)
        else:
            cand_pre_gp = np.asarray(raw_candidates_scaled, dtype=np.float32).copy()
        cand_pre_gp = self._apply_input_constraints(cand_pre_gp, artifacts)
        if artifacts.gp_input_scaler is not None:
            raw_candidates = artifacts.gp_input_scaler.transform(
                np.asarray(cand_pre_gp, dtype=np.float64)
            ).astype(np.float32)
        else:
            raw_candidates = cand_pre_gp
        decoded = self._decode_candidates(raw_candidates, artifacts, chosen_strategy)
        return SuggestionResult(raw_candidates=raw_candidates, decoded_candidates=decoded)

    def predict_targets(
        self,
        artifacts: TrainingArtifacts,
        inputs: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> PredictionResult:
        """지정한 반응물·조건 입력에 대해 학습된 GP의 posterior mean/std를 반환한다."""
        if self.surrogate.model is None:
            raise RuntimeError("GP model is not fitted. Call train_from_csv first.")
        rows = [inputs] if isinstance(inputs, dict) else list(inputs)
        if not rows:
            raise ValueError("At least one input row is required.")
        x_gp, meta_rows = self._encode_query_inputs(artifacts, rows)
        pred_mean, pred_var = self.surrogate.predict(np.asarray(x_gp, dtype=np.float64))
        predictions: List[Dict[str, Any]] = []
        for i, meta in enumerate(meta_rows):
            stats = self._target_prediction_stats(pred_mean[i], pred_var[i], artifacts)
            predictions.append({**meta, **stats})
        return PredictionResult(raw_features=x_gp, predictions=predictions)

    def _encode_query_inputs(
        self,
        artifacts: TrainingArtifacts,
        inputs: List[Dict[str, Any]],
    ) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        react_col = self.config.data.reactant_column
        delim = self.config.data.reactant_delimiter
        feature_rows: List[np.ndarray] = []
        meta_rows: List[Dict[str, Any]] = []

        for inp in inputs:
            raw_reactants = inp.get("reactants", inp.get(react_col))
            if raw_reactants is None:
                raise ValueError(
                    f"Each input must include 'reactants' or '{react_col}'."
                )
            if isinstance(raw_reactants, list):
                tokens = [str(t).strip() for t in raw_reactants if str(t).strip()]
                raw_str = delim.join(tokens)
            else:
                raw_str = str(raw_reactants).strip()
                tokens = [x.strip() for x in raw_str.split(delim) if x.strip()]
            if not tokens:
                raise ValueError("At least one reactant token is required.")

            smiles_list = [
                self.dataset_service.resolver.to_canonical_smiles(tok) for tok in tokens
            ]
            per_molecule = self.featurizers.featurize(
                smiles_list, name=self.config.featurization.featuriser
            )
            mol_vec = per_molecule.mean(axis=0)

            cond_values: Dict[str, Any] = {}
            nested = inp.get("conditions")
            if isinstance(nested, dict):
                cond_values.update(nested)
            for col in artifacts.condition_columns:
                if col in inp:
                    cond_values[col] = inp[col]

            missing = [c for c in artifacts.condition_columns if c not in cond_values]
            if missing:
                raise ValueError(f"Missing condition values for: {missing}")

            if artifacts.condition_columns:
                if artifacts.condition_transformer is None:
                    raise RuntimeError("condition_transformer is missing on artifacts.")
                cond_df = pd.DataFrame(
                    [{col: cond_values[col] for col in artifacts.condition_columns}]
                )
                x_cond = artifacts.condition_transformer.transform(cond_df)
                if hasattr(x_cond, "toarray"):
                    x_cond = x_cond.toarray()
                x_row = np.concatenate(
                    [mol_vec, np.asarray(x_cond, dtype=np.float32).reshape(-1)],
                    axis=0,
                )
            else:
                x_row = mol_vec

            feature_rows.append(np.asarray(x_row, dtype=np.float32))
            meta_rows.append(
                {
                    "reactants_input": raw_str,
                    "reactants_smiles": smiles_list,
                    "conditions": cond_values,
                }
            )

        x_pre_gp = np.vstack(feature_rows)
        if artifacts.gp_input_scaler is not None:
            x_gp = artifacts.gp_input_scaler.transform(
                np.asarray(x_pre_gp, dtype=np.float64)
            ).astype(np.float32)
        else:
            x_gp = x_pre_gp
        return x_gp, meta_rows

    def _target_prediction_stats(
        self,
        mean_row: np.ndarray,
        var_row: np.ndarray,
        artifacts: TrainingArtifacts,
    ) -> Dict[str, Any]:
        objective_map = self._build_target_objective_map(artifacts)
        target_predictions = {
            t: float(mean_row[t_idx]) for t_idx, t in enumerate(artifacts.target_columns)
        }
        target_uncertainty = {
            t: float(np.sqrt(max(float(var_row[t_idx]), 0.0)))
            for t_idx, t in enumerate(artifacts.target_columns)
        }
        target_gap: Dict[str, Optional[float]] = {}
        for t in artifacts.target_columns:
            t_cfg = objective_map.get(t, {})
            t_target = t_cfg.get("target_value")
            target_gap[t] = (
                abs(target_predictions[t] - float(t_target)) if t_target is not None else None
            )
        return {
            "predicted_target_mean": target_predictions,
            "predicted_target_std": target_uncertainty,
            "target_gap": target_gap,
        }

    def _apply_input_constraints(
        self, raw_candidates: np.ndarray, artifacts: TrainingArtifacts
    ) -> np.ndarray:
        adjusted = np.array(raw_candidates, copy=True)
        self._apply_reactant_constraints(adjusted, artifacts)
        self._apply_condition_constraints(adjusted, artifacts)
        return adjusted

    def _apply_reactant_constraints(self, candidates: np.ndarray, artifacts: TrainingArtifacts) -> None:
        allowed = self.config.data.reactant_allowed_values
        if not allowed:
            return
        allowed_vectors = []
        allowed_inputs: List[str] = []
        allowed_smiles_list: List[List[str]] = []
        for raw in allowed:
            tokens = [x.strip() for x in raw.split(self.config.data.reactant_delimiter) if x.strip()]
            if not tokens:
                continue
            smiles_list = [self.dataset_service.resolver.to_canonical_smiles(tok) for tok in tokens]
            per_molecule = self.featurizers.featurize(
                smiles_list, name=self.config.featurization.featuriser
            )
            allowed_vectors.append(per_molecule.mean(axis=0))
            allowed_inputs.append(raw)
            allowed_smiles_list.append(smiles_list)
        if not allowed_vectors:
            return
        allowed_matrix = np.vstack(allowed_vectors)
        artifacts.allowed_reactant_vectors = allowed_matrix
        artifacts.allowed_reactant_inputs = allowed_inputs
        artifacts.allowed_reactant_smiles = allowed_smiles_list
        for i in range(candidates.shape[0]):
            cand_mol = candidates[i, : artifacts.mol_feature_dim]
            distances = np.linalg.norm(allowed_matrix - cand_mol, axis=1)
            nearest_idx = int(np.argmin(distances))
            candidates[i, : artifacts.mol_feature_dim] = allowed_matrix[nearest_idx]

    def _apply_condition_constraints(self, candidates: np.ndarray, artifacts: TrainingArtifacts) -> None:
        if artifacts.condition_transformer is None or not artifacts.condition_columns:
            return
        ranges = self.config.data.condition_ranges
        if not ranges:
            return
        transformer = artifacts.condition_transformer
        categorical_cols = [c for c in artifacts.condition_columns if artifacts.condition_types.get(c) == "categorical"]
        numeric_cols = [c for c in artifacts.condition_columns if c not in categorical_cols]
        scaler = None
        if numeric_cols and "numeric" in transformer.named_transformers_:
            num_pipe = transformer.named_transformers_["numeric"]
            if isinstance(num_pipe, Pipeline):
                scaler = num_pipe.named_steps.get("scaler")

        categorical_width = 0
        if categorical_cols and "categorical" in transformer.named_transformers_:
            cat_pipe = transformer.named_transformers_["categorical"]
            ohe = cat_pipe.named_steps["ohe"]
            categorical_width = len(ohe.get_feature_names_out(categorical_cols))
            # categorical allowed_values 제약 적용
            start = artifacts.mol_feature_dim
            offset = 0
            for col in categorical_cols:
                cfg = ranges.get(col)
                cat_vals = list(ohe.categories_[offset])
                col_width = len(cat_vals)
                slice_start = start + offset
                slice_end = slice_start + col_width
                if cfg is not None and cfg.allowed_values:
                    allowed_set = {str(v) for v in cfg.allowed_values}
                    allowed_idx = [i for i, v in enumerate(cat_vals) if str(v) in allowed_set]
                    if allowed_idx:
                        for r in range(candidates.shape[0]):
                            row_slice = candidates[r, slice_start:slice_end]
                            best_local = max(allowed_idx, key=lambda j: row_slice[j])
                            row_slice[:] = 0.0
                            row_slice[best_local] = 1.0
                            candidates[r, slice_start:slice_end] = row_slice
                offset += col_width

        for n_idx, col in enumerate(numeric_cols):
            if scaler is None:
                continue
            cfg = ranges.get(col)
            if cfg is None:
                continue
            feat_idx = artifacts.mol_feature_dim + categorical_width + n_idx
            if cfg.min is not None:
                min_scaled = (float(cfg.min) - float(scaler.mean_[n_idx])) / float(scaler.scale_[n_idx])
                candidates[:, feat_idx] = np.maximum(candidates[:, feat_idx], min_scaled)
            if cfg.max is not None:
                max_scaled = (float(cfg.max) - float(scaler.mean_[n_idx])) / float(scaler.scale_[n_idx])
                candidates[:, feat_idx] = np.minimum(candidates[:, feat_idx], max_scaled)

    def _decode_candidates(
        self,
        raw_candidates: np.ndarray,
        artifacts: TrainingArtifacts,
        chosen_strategy: str,
    ) -> List[Dict[str, Any]]:
        decoded_rows: List[Dict[str, Any]] = []
        raw_f = np.asarray(raw_candidates, dtype=np.float64)
        train_basis = (
            artifacts.x_train_pre_gp_scale
            if artifacts.x_train_pre_gp_scale is not None
            else artifacts.x_train
        )
        train_mol = train_basis[:, : artifacts.mol_feature_dim]
        pred_mean, pred_var = self.surrogate.predict(raw_f)
        if artifacts.gp_input_scaler is not None:
            candidates_unscaled = artifacts.gp_input_scaler.inverse_transform(raw_f).astype(np.float32)
        else:
            candidates_unscaled = raw_f.astype(np.float32)
        objective_map = self._build_target_objective_map(artifacts)

        for i, candidate in enumerate(raw_f):
            cand_unscaled_row = candidates_unscaled[i]
            cand_mol = cand_unscaled_row[: artifacts.mol_feature_dim]
            if (
                artifacts.allowed_reactant_vectors is not None
                and artifacts.allowed_reactant_inputs is not None
                and artifacts.allowed_reactant_smiles is not None
            ):
                distances = np.linalg.norm(artifacts.allowed_reactant_vectors - cand_mol, axis=1)
                nearest_idx = int(np.argmin(distances))
                nearest_input = artifacts.allowed_reactant_inputs[nearest_idx]
                nearest_smiles = artifacts.allowed_reactant_smiles[nearest_idx]
            else:
                distances = np.linalg.norm(train_mol - cand_mol, axis=1)
                nearest_idx = int(np.argmin(distances))
                nearest_input = artifacts.reactant_inputs[nearest_idx]
                nearest_smiles = artifacts.reactant_smiles[nearest_idx]
            stats = self._target_prediction_stats(pred_mean[i], pred_var[i], artifacts)
            target_predictions = stats["predicted_target_mean"]
            target_uncertainty = stats["predicted_target_std"]
            target_gap = stats["target_gap"]

            objective_score = 0.0
            information_score = 0.0
            for t in artifacts.target_columns:
                t_cfg = objective_map.get(t, {})
                objective = t_cfg.get("objective", "maximize")
                t_target = t_cfg.get("target_value")
                weight = float(t_cfg.get("weight", 1.0))
                mean_val = target_predictions[t]
                std_val = target_uncertainty[t]
                information_score += weight * std_val
                if objective == "maximize":
                    objective_score += weight * mean_val
                elif objective == "minimize":
                    objective_score += -weight * mean_val
                elif objective == "target":
                    if t_target is None:
                        raise ValueError(f"target_value is required for target objective: {t}")
                    objective_score += -weight * abs(mean_val - float(t_target))
                else:
                    raise ValueError(f"Unknown objective: {objective}")
            total_score = information_score if chosen_strategy == "best_information" else objective_score

            row: Dict[str, Any] = {
                "predicted_target_mean": target_predictions,
                "predicted_target_std": target_uncertainty,
                "target_gap": target_gap,
                "objective_score": objective_score,
                "information_score": information_score,
                "total_score": total_score,
                "ranking_strategy": chosen_strategy,
                "mapped_reactants_input": nearest_input,
                "mapped_reactants_smiles": nearest_smiles,
                "nearest_known_reactants_input": nearest_input,
                "nearest_known_reactants_smiles": nearest_smiles,
                "nearest_reactant_distance": float(distances[nearest_idx]),
            }
            if self.config.data.reactant_allowed_values:
                row["reactant_candidates_scope"] = self.config.data.reactant_allowed_values

            if artifacts.condition_columns:
                cand_cond = cand_unscaled_row[artifacts.mol_feature_dim :].reshape(1, -1)
                cond_values = self._inverse_condition_values(cand_cond, artifacts)
                row.update(cond_values)
            decoded_rows.append(row)
        return decoded_rows

    def _inverse_condition_values(
        self, cond_vector: np.ndarray, artifacts: TrainingArtifacts
    ) -> Dict[str, Any]:
        if artifacts.condition_transformer is None:
            return {}
        try:
            restored = artifacts.condition_transformer.inverse_transform(cond_vector)
            values = restored[0]
            return {
                col: (float(val) if isinstance(val, (np.floating, float, int)) else val)
                for col, val in zip(artifacts.condition_columns, values)
            }
        except Exception:
            restored: Dict[str, Any] = {}
            transformer = artifacts.condition_transformer
            categorical_cols = [
                c for c in artifacts.condition_columns if artifacts.condition_types.get(c) == "categorical"
            ]
            numeric_cols = [c for c in artifacts.condition_columns if c not in categorical_cols]

            idx = 0
            if categorical_cols and "categorical" in transformer.named_transformers_:
                cat_pipe = transformer.named_transformers_["categorical"]
                ohe = cat_pipe.named_steps["ohe"]
                cat_width = len(ohe.get_feature_names_out(categorical_cols))
                cat_slice = cond_vector[:, idx : idx + cat_width]
                decoded_cat = ohe.inverse_transform(cat_slice)[0]
                for col, value in zip(categorical_cols, decoded_cat):
                    restored[col] = value
                idx += cat_width

            if numeric_cols and "numeric" in transformer.named_transformers_:
                num_pipe = transformer.named_transformers_["numeric"]
                scaler = num_pipe.named_steps["scaler"]
                num_width = len(numeric_cols)
                num_slice = cond_vector[:, idx : idx + num_width]
                decoded_num = scaler.inverse_transform(num_slice)[0]
                for col, value in zip(numeric_cols, decoded_num):
                    restored[col] = float(value)

            for col in artifacts.condition_columns:
                restored.setdefault(col, None)
            return restored

    def _build_target_objective_map(self, artifacts: TrainingArtifacts) -> Dict[str, Dict[str, Any]]:
        explicit_target_objectives = self.config.optimization.target_objectives
        objective_map: Dict[str, Dict[str, Any]] = {}
        for t in artifacts.target_columns:
            # 우선순위:
            # 1) target_objectives.<target_name> 개별 설정
            # 2) optimization 전역 설정(objective/target_value)
            if t in explicit_target_objectives:
                obj = explicit_target_objectives[t]
                objective_map[t] = {
                    "objective": obj.objective,
                    "target_value": obj.target_value,
                    "weight": obj.weight,
                }
            else:
                objective_map[t] = {
                    "objective": self.config.optimization.objective,
                    "target_value": self.config.optimization.target_value,
                    "weight": 1.0,
                }
        return objective_map
