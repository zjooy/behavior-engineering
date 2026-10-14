"""Estágio `train`: treina o pipeline, avalia no teste e salva modelo e métricas."""

from behavior_engineering.config import Settings, load_params
from behavior_engineering.data.loader import DataLoader
from behavior_engineering.features.preprocessor import build_preprocessor
from behavior_engineering.models.evaluator import Evaluator
from behavior_engineering.models.trainer import Trainer
from behavior_engineering.reproducibility import set_global_seed


def main() -> None:
    """Executa treino e avaliação a partir dos splits gerados pelo `preprocess`."""
    settings = Settings()
    params = load_params(settings.params_path)
    set_global_seed(params.seed)

    loader = DataLoader(params.data.target)
    x_train, y_train = loader.split_features_target(
        loader.load(params.data.processed_dir / "train.csv")
    )
    x_test, y_test = loader.split_features_target(
        loader.load(params.data.processed_dir / "test.csv")
    )

    preprocessor = build_preprocessor(params.features)
    trainer = Trainer(preprocessor, params.model, params.seed, settings.device)
    pipeline = trainer.fit(x_train, y_train)
    trainer.save(params.train.model_path)

    evaluator = Evaluator()
    metrics = evaluator.evaluate(pipeline, x_test, y_test)
    evaluator.save(metrics, params.train.metrics_path)
    print(metrics)


if __name__ == "__main__":
    main()
