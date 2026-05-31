from unittest.mock import MagicMock

import numpy as np

from prophet_gp.config import AppConfig, DataConfig, OptimizationConfig
from prophet_gp.pipeline.trainer import ProphetGPPipeline, TrainingArtifacts


def _artifacts_with_conditions() -> TrainingArtifacts:
    return TrainingArtifacts(
        x_train=np.zeros((4, 3), dtype=np.float32),
        y_train=np.zeros((4, 1), dtype=np.float32),
        target_columns=["target"],
        feature_names=["x0", "x1", "x2"],
        mol_feature_dim=2,
        reactant_inputs=["A", "B", "A", "B"],
        reactant_smiles=[["C"], ["CC"], ["C"], ["CC"]],
        condition_columns=["Temperature"],
        condition_types={"Temperature": "continuous"},
        condition_transformer=None,
        condition_train_values={"Temperature": [100.0, 200.0, 300.0]},
        reactant_scope=["A", "B"],
        training_query_inputs=[
            {"reactants": "A", "Temperature": 100.0},
            {"reactants": "B", "Temperature": 200.0},
        ],
    )


def test_condition_grid_uses_config_only_not_training_temps() -> None:
    pipeline = ProphetGPPipeline.__new__(ProphetGPPipeline)
    pipeline.config = AppConfig(
        data=DataConfig(
            condition_ranges={"Temperature": {"min": 0.0, "max": 600.0, "grid_points": 7}},
        ),
        optimization=OptimizationConfig(condition_grid_points=25),
    )
    artifacts = _artifacts_with_conditions()
    temps = pipeline._condition_value_grid("Temperature", artifacts)
    assert 80.0 not in temps
    assert 150.0 not in temps
    assert temps == [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]


def test_discrete_inputs_training_only_without_condition_ranges() -> None:
    pipeline = ProphetGPPipeline.__new__(ProphetGPPipeline)
    pipeline.config = AppConfig(data=DataConfig(condition_ranges={}))
    artifacts = _artifacts_with_conditions()
    rows = pipeline._build_discrete_query_inputs(artifacts)
    assert len(rows) == 2
    assert rows == artifacts.training_query_inputs


def test_build_discrete_query_inputs_cartesian() -> None:
    pipeline = ProphetGPPipeline.__new__(ProphetGPPipeline)
    pipeline.config = AppConfig(
        data=DataConfig(
            reactant_allowed_values=["A", "B"],
            condition_ranges={"Temperature": {"min": 100.0, "max": 200.0, "grid_points": 3}},
        ),
        optimization=OptimizationConfig(condition_grid_points=25),
    )
    artifacts = _artifacts_with_conditions()
    rows = pipeline._build_discrete_query_inputs(artifacts)
    assert len(rows) == 6  # 2 reactants * 3 grid temps (no extra training temps)
    temps = {row["Temperature"] for row in rows}
    assert temps == {100.0, 150.0, 200.0}
    assert 300.0 not in temps


def test_score_candidate_pool_target_and_minimize() -> None:
    pipeline = ProphetGPPipeline.__new__(ProphetGPPipeline)
    pipeline.config = AppConfig(
        optimization=OptimizationConfig(
            target_objectives={
                "Emission Peak": {"objective": "target", "target_value": 490.0, "weight": 2.0},
                "FWHM": {"objective": "minimize", "weight": 1.0},
            }
        )
    )
    artifacts = TrainingArtifacts(
        x_train=np.zeros((1, 1), dtype=np.float32),
        y_train=np.zeros((1, 2), dtype=np.float32),
        target_columns=["Emission Peak", "FWHM"],
        feature_names=["x0"],
        mol_feature_dim=1,
        reactant_inputs=["A"],
        reactant_smiles=[["C"]],
        condition_columns=[],
        condition_types={},
        condition_transformer=None,
    )
    pred_mean = np.array([[512.0, 95.0], [515.0, 98.0]])
    pred_var = np.ones_like(pred_mean)
    objective_map = pipeline._build_target_objective_map(artifacts)
    scores = pipeline._score_candidate_pool(
        pred_mean, pred_var, artifacts, objective_map, "best_output"
    )
    assert scores[0] > scores[1]


def test_suggest_uses_discrete_pool(monkeypatch) -> None:
    pipeline = ProphetGPPipeline.__new__(ProphetGPPipeline)
    pipeline.config = AppConfig(
        data=DataConfig(reactant_allowed_values=["A"]),
        optimization=OptimizationConfig(n_candidates=1, target_search_size=100),
    )
    artifacts = _artifacts_with_conditions()

    query_rows = [{"reactants": "A", "Temperature": 150.0}]
    encoded = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    meta = [{"reactants_input": "A", "reactants_smiles": ["C"], "conditions": {"Temperature": 150.0}}]

    monkeypatch.setattr(pipeline, "_build_discrete_query_inputs", lambda _a: query_rows)
    monkeypatch.setattr(pipeline, "_encode_query_inputs", lambda _a, _r: (encoded, meta))
    monkeypatch.setattr(
        pipeline,
        "_decode_candidates",
        lambda raw, _a, strategy, meta_rows=None: [
            {
                "mapped_reactants_input": meta_rows[0]["reactants_input"],
                "nearest_reactant_distance": 0.0,
                "Temperature": 150.0,
                "ranking_strategy": strategy,
            }
        ],
    )

    pipeline.surrogate = MagicMock()
    pipeline.surrogate.predict.return_value = (np.array([[500.0]]), np.array([[1.0]]))

    result = pipeline.suggest_next_experiments(artifacts, n_candidates=1, strategy="best_output")
    assert result.decoded_candidates[0]["nearest_reactant_distance"] == 0.0
    assert result.decoded_candidates[0]["Temperature"] == 150.0
