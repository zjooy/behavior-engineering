"""Leitura e divisão do dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """Carrega um CSV e separa features e alvo."""

    def __init__(self, target: str) -> None:
        self.target = target

    def load(self, path: Path) -> pd.DataFrame:
        """Lê o CSV indicado."""
        return pd.read_csv(path)

    def split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Retorna (X, y), com o alvo convertido para inteiro."""
        return df.drop(columns=[self.target]), df[self.target].astype(int)


def split_train_test(
    df: pd.DataFrame, target: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o dataset em treino e teste, estratificando pelo alvo."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[target]
    )
    return train_df, test_df
