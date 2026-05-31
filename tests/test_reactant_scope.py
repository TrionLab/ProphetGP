import pandas as pd

from prophet_gp.config import AppConfig, DataConfig
from prophet_gp.data.dataset import PreparedDataset
from prophet_gp.data.schema import DatasetSchema
from prophet_gp.pipeline.trainer import ProphetGPPipeline


def _prepared_frame() -> PreparedDataset:
    frame = pd.DataFrame(
        {
            "Mol.1": ["B", "A", "B", "C"],
            "Temperature": [100, 200, 300, 400],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    schema = DatasetSchema(
        reactant_column="Mol.1",
        target_columns=["target"],
        condition_columns=["Temperature"],
        condition_types={"Temperature": "continuous"},
    )
    return PreparedDataset(
        frame=frame,
        resolved_smiles=[["C"], ["CC"], ["C"], ["CCC"]],
        schema=schema,
        y=frame[["target"]],
    )


def test_reactant_scope_from_training_when_config_empty() -> None:
    pipeline = ProphetGPPipeline(AppConfig(data=DataConfig(reactant_allowed_values=[])))
    scope = pipeline._resolve_reactant_scope(_prepared_frame())
    assert scope == ["A", "B", "C"]


def test_reactant_scope_uses_config_when_provided() -> None:
    pipeline = ProphetGPPipeline(
        AppConfig(data=DataConfig(reactant_allowed_values=["A", "Z"]))
    )
    scope = pipeline._resolve_reactant_scope(_prepared_frame())
    assert scope == ["A", "Z"]
