import pandas as pd

from prophet_gp.data.schema import infer_condition_type


def test_infer_condition_type_for_categorical():
    series = pd.Series(["A", "B", "A"])
    assert infer_condition_type(series) == "categorical"


def test_infer_condition_type_for_discrete():
    series = pd.Series([0, 1, 0, 1, 2])
    assert infer_condition_type(series) == "discrete"
