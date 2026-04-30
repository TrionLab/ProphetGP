from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

ConditionType = Literal["categorical", "continuous", "discrete"]


class DataConfig(BaseModel):
    reactant_column: str = "reactants"
    target_column: str = "target"
    reactant_delimiter: str = "|"
    ignore_columns: List[str] = Field(
        default_factory=list,
        description="파일에 있어도 조건 피처에서 제외할 컬럼(보조 측정값 등).",
    )
    explicit_condition_types: Dict[str, ConditionType] = Field(default_factory=dict)


class FeaturizationConfig(BaseModel):
    featuriser: str = "ecfp_fingerprints"
    combine_strategy: Literal["concat"] = "concat"


class OptimizationConfig(BaseModel):
    objective: Literal["maximize", "minimize", "target"] = "maximize"
    suggestion_strategy: Literal["best_output", "best_information"] = "best_output"
    target_value: Optional[float] = None
    n_restarts: int = 10
    raw_samples: int = 128
    target_search_size: int = 4096
    n_candidates: int = 5


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    featurization: FeaturizationConfig = Field(default_factory=FeaturizationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)
