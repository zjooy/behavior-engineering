"""Estágio `preprocess`: divide o dataset bruto em treino e teste."""

from behavior_engineering.config import Settings, load_params
from behavior_engineering.data.loader import DataLoader, split_train_test
from behavior_engineering.reproducibility import set_global_seed


def main() -> None:
    """Lê o CSV bruto e grava `train.csv` e `test.csv` em `processed_dir`."""
    params = load_params(Settings().params_path)
    set_global_seed(params.seed)
    data = params.data

    raw_df = DataLoader(data.target).load(data.raw_path)
    train_df, test_df = split_train_test(raw_df, data.target, data.test_size, params.seed)

    data.processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(data.processed_dir / "train.csv", index=False)
    test_df.to_csv(data.processed_dir / "test.csv", index=False)
    print(f"Treino: {len(train_df)} linhas | Teste: {len(test_df)} linhas")


if __name__ == "__main__":
    main()
