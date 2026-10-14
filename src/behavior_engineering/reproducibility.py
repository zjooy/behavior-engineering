"""Utilitários de reprodutibilidade."""

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Fixa as seeds globais do Python e do NumPy."""
    random.seed(seed)
    np.random.seed(seed)
