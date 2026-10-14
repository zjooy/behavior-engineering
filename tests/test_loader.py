import pandas as pd

from behavior_engineering.data.loader import DataLoader, split_train_test


def make_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PageValues": range(100),
            "Revenue": [True] * 20 + [False] * 80,
        }
    )


def test_split_preserves_target_proportion_and_size() -> None:
    train_df, test_df = split_train_test(make_sample(), "Revenue", test_size=0.2, seed=42)

    assert len(test_df) == 20
    assert train_df["Revenue"].mean() == test_df["Revenue"].mean() == 0.2


def test_split_is_deterministic_for_same_seed() -> None:
    first, _ = split_train_test(make_sample(), "Revenue", test_size=0.2, seed=42)
    second, _ = split_train_test(make_sample(), "Revenue", test_size=0.2, seed=42)

    assert first.index.equals(second.index)


def test_split_features_target_removes_target_and_casts_to_int() -> None:
    features, target = DataLoader("Revenue").split_features_target(make_sample())

    assert "Revenue" not in features.columns
    assert target.dtype.kind == "i"
