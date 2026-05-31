from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field

ConditionType = Literal["categorical", "continuous", "discrete"]


class InputRangeConfig(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    allowed_values: Optional[List[Union[str, float, int]]] = None
    grid_points: Optional[int] = Field(
        default=None,
        description="연속/이산 조건 suggestion 그리드 점 개수(컬럼별 override).",
    )


class TargetObjectiveConfig(BaseModel):
    objective: Literal["maximize", "minimize", "target"] = "maximize"
    target_value: Optional[float] = None
    weight: float = 1.0


class DataConfig(BaseModel):
    reactant_column: str = "reactants"
    target_column: Union[str, List[str]] = "target"
    reactant_delimiter: str = "|"
    ignore_columns: List[str] = Field(
        default_factory=list,
        description="파일에 있어도 조건 피처에서 제외할 컬럼(보조 측정값 등).",
    )
    reactant_allowed_values: List[str] = Field(
        default_factory=list,
        description=(
            "추천 시 허용할 반응물 입력값 목록. "
            "비어 있거나 YAML에서 생략하면 학습 CSV의 고유 반응물로 자동 설정된다."
        ),
    )
    condition_ranges: Dict[str, InputRangeConfig] = Field(
        default_factory=dict,
        description=(
            "조건 컬럼별 suggestion 그리드(min/max/grid_points 또는 allowed_values). "
            "비어 있으면 suggestion은 학습 CSV에 등장한 조건값 조합만 사용한다."
        ),
    )
    explicit_condition_types: Dict[str, ConditionType] = Field(default_factory=dict)


class FeaturizationConfig(BaseModel):
    featuriser: str = "ecfp_fingerprints"
    combine_strategy: Literal["concat"] = "concat"


class OptimizationConfig(BaseModel):
    objective: Literal["maximize", "minimize", "target"] = "maximize"
    suggestion_strategy: Literal["best_output", "best_information"] = "best_output"
    standardize_gp_inputs: bool = Field(
        default=False,
        description=(
            "True면 결합 입력 특징을 학습 데이터 min-max로 [0,1]에 넣는다(BoTorch 단위입방·수치 안정). "
            "Z-score 표준화와 다르다."
        ),
    )
    standardize_gp_targets: bool = Field(
        default=False,
        description="True면 타깃 Y에 StandardScaler(타깃별)를 적용해 GP를 학습한다; predict는 원 스케일로 역변환한다.",
    )
    target_value: Optional[float] = None
    target_objectives: Dict[str, TargetObjectiveConfig] = Field(default_factory=dict)
    n_restarts: int = 10
    raw_samples: int = 128
    target_search_size: int = 4096
    condition_grid_points: int = Field(
        default=25,
        description="연속 조건 suggestion 시 min~max 균등 그리드 점 개수(컬럼별 grid_points 없을 때).",
    )
    n_candidates: int = 5


class AppConfig(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    featurization: FeaturizationConfig = Field(default_factory=FeaturizationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AppConfig.model_validate(raw)
