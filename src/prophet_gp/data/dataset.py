from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prophet_gp.chem.resolver import MoleculeResolver
from prophet_gp.config import DataConfig
from prophet_gp.data.schema import DatasetSchema, infer_condition_type


@dataclass
class PreparedDataset:
    frame: pd.DataFrame
    resolved_smiles: List[List[str]]
    schema: DatasetSchema
    y: pd.Series


class ReactionDatasetService:
    def __init__(self, data_config: DataConfig):
        self.config = data_config
        self.resolver = MoleculeResolver()

    def load_csv(self, path: str | Path) -> PreparedDataset:
        df = pd.read_csv(path)
        return self._prepare(df)

    def append_csv(self, base_path: str | Path, new_path: str | Path, out_path: str | Path) -> pd.DataFrame:
        base = pd.read_csv(base_path)
        new = pd.read_csv(new_path)
        merged = pd.concat([base, new], ignore_index=True).drop_duplicates()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        return merged

    def build_condition_transformer(self, schema: DatasetSchema) -> ColumnTransformer:
        categorical_cols = [k for k, t in schema.condition_types.items() if t == "categorical"]
        numeric_cols = [k for k, t in schema.condition_types.items() if t in {"continuous", "discrete"}]

        transformers = []
        if categorical_cols:
            cat_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("categorical", cat_pipe, categorical_cols))

        if numeric_cols:
            num_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", num_pipe, numeric_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _prepare(self, df: pd.DataFrame) -> PreparedDataset:
        react_col = self.config.reactant_column
        target_col = self.config.target_column
        if react_col not in df.columns:
            raise ValueError(f"Missing reactant column: {react_col}")
        if target_col not in df.columns:
            raise ValueError(f"Missing target column: {target_col}")

        df = df.copy()
        reactants = df[react_col].astype(str).tolist()
        resolved: List[List[str]] = []
        for row in reactants:
            tokens = [x.strip() for x in row.split(self.config.reactant_delimiter) if x.strip()]
            if not tokens:
                raise ValueError("At least one reactant is required per row.")
            resolved_row = [self.resolver.to_canonical_smiles(token) for token in tokens]
            resolved.append(resolved_row)

        condition_cols = [c for c in df.columns if c not in {react_col, target_col}]
        inferred = {col: infer_condition_type(df[col]) for col in condition_cols}
        inferred.update(self.config.explicit_condition_types)

        schema = DatasetSchema(
            reactant_column=react_col,
            target_column=target_col,
            condition_columns=condition_cols,
            condition_types=inferred,
        )
        return PreparedDataset(frame=df, resolved_smiles=resolved, schema=schema, y=df[target_col])
