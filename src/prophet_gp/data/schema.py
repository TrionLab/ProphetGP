from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

import pandas as pd

ConditionType = Literal["categorical", "continuous", "discrete"]


@dataclass(frozen=True)
class DatasetSchema:
    reactant_column: str
    target_column: str
    condition_columns: List[str]
    condition_types: Dict[str, ConditionType]


def infer_condition_type(series: pd.Series) -> ConditionType:
    non_null = series.dropna()
    if non_null.empty:
        return "continuous"

    if pd.api.types.is_object_dtype(non_null) or pd.api.types.is_string_dtype(non_null):
        return "categorical"

    if pd.api.types.is_integer_dtype(non_null):
        unique_count = non_null.nunique()
        if unique_count <= min(20, max(3, len(non_null) // 5)):
            return "discrete"
        return "continuous"

    return "continuous"
