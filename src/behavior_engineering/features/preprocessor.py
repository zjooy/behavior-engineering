"""Pré-processamento das features."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from behavior_engineering.config import FeatureParams


def build_preprocessor(params: FeatureParams) -> ColumnTransformer:
    """Descarta colunas de ruído e aplica one-hot nas categóricas; o resto passa direto."""
    encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=params.ohe_min_frequency
    )
    return ColumnTransformer(
        transformers=[
            ("drop_noise", "drop", params.drop),
            ("one_hot", encoder, params.categorical),
        ],
        remainder="passthrough",
    )
