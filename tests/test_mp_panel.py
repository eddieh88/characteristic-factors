"""The holiday-file filter in data/mp_panel.py."""
import numpy as np
import pandas as pd

from conftest import load_script

mp_panel = load_script("data/mp_panel.py")


def test_holiday_file_with_one_symbol_is_dropped():
    days = pd.bdate_range("2020-01-01", periods=60)
    px = pd.DataFrame(1.0, index=days, columns=[f"S{i}" for i in range(100)])
    holiday = days[40]
    px.loc[holiday, "S1":] = np.nan  # the vendor's holiday file: one symbol only
    keep = mp_panel.session_days(px)
    assert not keep[holiday]
    assert keep.drop(holiday).all()


def test_gradual_growth_in_names_is_kept():
    days = pd.bdate_range("2020-01-01", periods=300)
    counts = np.linspace(50, 100, len(days)).astype(int)
    px = pd.DataFrame(np.nan, index=days, columns=[f"S{i}" for i in range(100)])
    for d, n in zip(days, counts):
        px.loc[d, px.columns[:n]] = 1.0
    assert mp_panel.session_days(px).all()
