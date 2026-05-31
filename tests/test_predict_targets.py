from unittest.mock import MagicMock

import numpy as np

from prophet_gp.pipeline.trainer import ProphetGPPipeline, TrainingArtifacts


def _minimal_artifacts() -> TrainingArtifacts:
    return TrainingArtifacts(
        x_train=np.zeros((3, 4), dtype=np.float32),
        y_train=np.zeros((3, 2), dtype=np.float32),
        target_columns=["Emission Peak", "FWHM"],
        feature_names=["x0", "x1", "x2", "x3"],
        mol_feature_dim=2,
        reactant_inputs=["a", "b", "c"],
        reactant_smiles=[["C"], ["CC"], ["CCC"]],
        condition_columns=["Temperature"],
        condition_types={"Temperature": "continuous"},
        condition_transformer=None,
        gp_input_scaler=None,
        reactant_scope=["a", "b", "c"],
    )


def test_predict_targets_calls_surrogate(monkeypatch) -> None:
    pipeline = ProphetGPPipeline.__new__(ProphetGPPipeline)
    pipeline.config = MagicMock()
    pipeline.config.data.reactant_column = "Mol.1"
    pipeline.config.data.reactant_delimiter = "|"
    pipeline.config.featurization.featuriser = "morgan_fp"
    pipeline.config.optimization = MagicMock()
    pipeline.config.optimization.target_objectives = {}
    pipeline.config.optimization.objective = "maximize"
    pipeline.config.optimization.target_value = None

    pipeline.dataset_service = MagicMock()
    pipeline.dataset_service.resolver.to_canonical_smiles.side_effect = lambda t: t
    pipeline.featurizers = MagicMock()
    pipeline.featurizers.featurize.return_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    artifacts = _minimal_artifacts()
    encoded = np.array([[1.5, 3.5, 0.2]], dtype=np.float32)
    monkeypatch.setattr(
        pipeline,
        "_encode_query_inputs",
        lambda _artifacts, _rows: (encoded, [{"reactants_input": "a|b", "reactants_smiles": ["a", "b"], "conditions": {"Temperature": 300.0}}]),
    )

    pipeline.surrogate = MagicMock()
    pipeline.surrogate.model = MagicMock()
    pipeline.surrogate.predict.return_value = (
        np.array([[500.0, 90.0]]),
        np.array([[9.0, 4.0]]),
    )

    result = pipeline.predict_targets(
        artifacts,
        {"reactants": "a|b", "Temperature": 300.0},
    )

    pipeline.surrogate.predict.assert_called_once()
    row = result.predictions[0]
    assert row["predicted_target_mean"]["Emission Peak"] == 500.0
    assert row["predicted_target_std"]["FWHM"] == 2.0
