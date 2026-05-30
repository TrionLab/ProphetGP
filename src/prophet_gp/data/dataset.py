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
    y: pd.DataFrame


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
        target_cols = (
            [self.config.target_column]
            if isinstance(self.config.target_column, str)
            else list(self.config.target_column)
        )
        if react_col not in df.columns:
            raise ValueError(f"Missing reactant column: {react_col}")
        for target_col in target_cols:
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

        ignore = set(self.config.ignore_columns)
        condition_cols = [
            c for c in df.columns if c not in ({react_col} | set(target_cols)) and c not in ignore
        ]
        inferred = {col: infer_condition_type(df[col]) for col in condition_cols}
        inferred.update(self.config.explicit_condition_types)

        schema = DatasetSchema(
            reactant_column=react_col,
            target_columns=target_cols,
            condition_columns=condition_cols,
            condition_types=inferred,
        )
        return PreparedDataset(frame=df, resolved_smiles=resolved, schema=schema, y=df[target_cols])

    def drop_na_training_rows(self, prepared: PreparedDataset) -> PreparedDataset:
        """반응물·타깃·조건 열 중 하나라도 NA인 행을 제거한다."""
        cols = [
            prepared.schema.reactant_column,
            *prepared.schema.target_columns,
            *prepared.schema.condition_columns,
        ]
        mask = prepared.frame[cols].notna().all(axis=1)
        if bool(mask.all()):
            return prepared
        frame = prepared.frame.loc[mask].reset_index(drop=True)
        resolved = [prepared.resolved_smiles[i] for i, m in enumerate(mask) if m]
        y = prepared.y.loc[mask].reset_index(drop=True)
        return PreparedDataset(
            frame=frame,
            resolved_smiles=resolved,
            schema=prepared.schema,
            y=y,
        )
