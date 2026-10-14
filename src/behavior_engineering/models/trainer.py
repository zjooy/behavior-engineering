"""Montagem, treino e persistência do pipeline de classificação."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


class Trainer:
    """Treina um Pipeline (pré-processador + XGBoost) e o salva em disco."""

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        model_params: dict[str, Any],
        seed: int,
        device: str = "cpu",
    ) -> None:
        classifier = XGBClassifier(**model_params, random_state=seed, device=device)
        self.pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Pipeline:
        """Ajusta o pipeline e o retorna."""
        return self.pipeline.fit(features, target)

    def save(self, path: Path) -> None:
        """Serializa o pipeline com joblib, criando o diretório se necessário."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
