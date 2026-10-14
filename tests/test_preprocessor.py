import pandas as pd

from behavior_engineering.config import FeatureParams
from behavior_engineering.features.preprocessor import build_preprocessor


def make_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PageValues": [0.0, 10.5, 3.2, 0.0],
            "Month": ["Feb", "Mar", "Feb", "Nov"],
            "VisitorType": ["Returning_Visitor", "New_Visitor", "Other", "Returning_Visitor"],
            "Weekend": [False, True, False, True],
        }
    )


def test_preprocessor_drops_noise_and_encodes_categoricals() -> None:
    params = FeatureParams(
        drop=["Weekend"], categorical=["Month", "VisitorType"], ohe_min_frequency=0.01
    )
    transformed = build_preprocessor(params).fit_transform(make_sample())

    # 3 meses + 3 tipos de visitante (one-hot) + PageValues (passthrough)
    assert transformed.shape == (4, 7)


def test_preprocessor_ignores_unseen_category() -> None:
    params = FeatureParams(drop=["Weekend"], categorical=["Month"], ohe_min_frequency=0.01)
    preprocessor = build_preprocessor(params)
    preprocessor.fit(make_sample().drop(columns=["VisitorType"]))

    unseen = make_sample().drop(columns=["VisitorType"]).assign(Month="Dec")
    assert preprocessor.transform(unseen).shape[0] == 4
