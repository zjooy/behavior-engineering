from pathlib import Path

from behavior_engineering.config import load_params

PARAMS_PATH = Path(__file__).parents[1] / "configs" / "params.yaml"


def test_load_params_reads_project_file() -> None:
    params = load_params(PARAMS_PATH)

    assert params.seed == 42
    assert params.data.target == "Revenue"
    assert "Month" in params.features.categorical
