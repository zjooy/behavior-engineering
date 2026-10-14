"""Cálculo das métricas de classificação."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


class Evaluator:
    """Avalia um classificador binário no conjunto de teste."""

    def evaluate(
        self, pipeline: Pipeline, features: pd.DataFrame, target: pd.Series
    ) -> dict[str, float]:
        """Retorna acurácia, F1, precisão, recall e PR-AUC."""
        predictions = pipeline.predict(features)
        probabilities = pipeline.predict_proba(features)[:, 1]
        return {
            "test_accuracy": float(accuracy_score(target, predictions)),
            "test_f1": float(f1_score(target, predictions)),
            "test_precision": float(precision_score(target, predictions)),
            "test_recall": float(recall_score(target, predictions)),
            "test_pr_auc": float(average_precision_score(target, probabilities)),
        }

    def save(self, metrics: dict[str, float], path: Path) -> None:
        """Grava as métricas em JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
