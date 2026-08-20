"""The contract Cason imports: load_stocki and the exploration helpers."""

import numpy as np
import pandas as pd
import pytest

from stocki.datasets.loaders import (
    coverage,
    describe,
    load_bars,
    load_fundamentals,
    load_panel,
    load_raw,
    load_stocki,
)
from stocki.errors import StockiEmptyError
from stocki.ingest.columns import CSV_COLUMNS, sql_name
from stocki.ingest.load import ingest_directory

from .conftest import write_session

pytestmark = pytest.mark.db

TICKERS = ("AAPL", "NVDA")
NVDA_DAYS = range(1, 7)
AAPL_DAYS = range(3, 7)  # AAPL starts late, the way it does in the real data


@pytest.fixture
def loaded(db, tmp_path):
    """Two tickers, six days, one of them with a coverage gap."""
    for day in NVDA_DAYS:
        write_session(tmp_path, ticker="NVDA", day=day)
    for day in AAPL_DAYS:
        write_session(tmp_path, ticker="AAPL", day=day)
    report = ingest_directory(db, tmp_path)
    assert report.ok, report.summary()
    return db


# --- guardrails -----------------------------------------------------------


def test_an_empty_database_says_to_run_ingest(db):
    with pytest.raises(StockiEmptyError) as caught:
        load_stocki(conn=db)

    assert "stocki ingest" in str(caught.value)


# --- load_bars ------------------------------------------------------------


def test_load_bars_returns_every_bar(loaded):
    bars = load_bars(conn=loaded)

    assert len(bars) == (len(NVDA_DAYS) + len(AAPL_DAYS)) * 78


def test_load_bars_columns_are_the_time_series_ones(loaded):
    bars = load_bars(conn=loaded)

    assert list(bars.columns) == [
        "ticker",
        "day",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_load_bars_filters_by_ticker_given_a_bare_string(loaded):
    bars = load_bars(tickers="NVDA", conn=loaded)

    assert set(bars["ticker"]) == {"NVDA"}


def test_load_bars_filters_by_several_tickers_and_days(loaded):
    bars = load_bars(tickers=["NVDA", "AAPL"], days=[3, 4], conn=loaded)

    assert set(bars["day"]) == {3, 4}
    assert len(bars) == 4 * 78


def test_load_bars_as_numpy_gives_named_fields(loaded):
    """Works without touching pandas: bars["close"].mean()."""
    bars = load_bars(as_frame=False, conn=loaded)

    assert bars["close"].mean() > 0
    assert bars["ticker"][0] in TICKERS


def test_load_bars_rejects_an_unknown_ticker_by_name(loaded):
    with pytest.raises(ValueError) as caught:
        load_bars(tickers="NOPE", conn=loaded)

    assert "NOPE" in str(caught.value)
    assert "NVDA" in str(caught.value)


# --- load_raw -------------------------------------------------------------


def test_load_raw_returns_the_whole_csv_row(loaded):
    raw = load_raw("NVDA", 3, conn=loaded)

    assert len(raw) == 78
    assert list(raw.columns) == [sql_name(c) for c in CSV_COLUMNS]


def test_load_raw_on_a_missing_session_is_empty_not_an_error(loaded):
    assert len(load_raw("AAPL", 1, conn=loaded)) == 0


# --- fundamentals ---------------------------------------------------------


def test_fundamentals_have_one_row_per_ticker_day(loaded):
    fundamentals = load_fundamentals(conn=loaded)

    assert len(fundamentals) == len(NVDA_DAYS) + len(AAPL_DAYS)
    assert list(fundamentals.columns[:2]) == ["ticker", "day"]


# --- panel ----------------------------------------------------------------


def test_panel_is_a_ticker_by_day_by_bar_cube(loaded):
    panel = load_panel(conn=loaded)

    assert panel.values.shape == (2, 6, 78, 5)
    assert panel.tickers == ["AAPL", "NVDA"]
    assert panel.fields == ("open", "high", "low", "close", "volume")


def test_panel_marks_missing_sessions_as_nan_not_zero(loaded):
    """A plain mean gives NaN so you find out; nanmean gives the truth."""
    panel = load_panel(conn=loaded)
    aapl = panel.values[panel.tickers.index("AAPL")]

    assert np.isnan(aapl[0]).all()
    assert not np.isnan(aapl[2]).any()
    assert np.isfinite(np.nanmean(aapl[..., 3]))


# --- load_stocki ----------------------------------------------------------


def test_load_stocki_shapes(loaded):
    ds = load_stocki(conn=loaded)

    sessions = len(NVDA_DAYS) + len(AAPL_DAYS)
    assert ds.data.shape == (sessions * 46, 32, 8)
    assert ds.target.shape == (sessions * 46,)


def test_bunch_supports_attribute_and_key_access(loaded):
    ds = load_stocki(conn=loaded)

    assert ds.data is ds["data"]
    assert ds.feature_names == list(ds["feature_names"])


def test_bunch_carries_the_metadata_needed_to_slice(loaded):
    ds = load_stocki(conn=loaded)

    nvda = ds.data[ds.ticker == "NVDA"]

    assert len(nvda) == len(NVDA_DAYS) * 46
    assert ds.day.shape == ds.timestamps.shape == ds.target.shape


def test_target_is_binary(loaded):
    ds = load_stocki(conn=loaded)

    assert set(np.unique(ds.target)) <= {0, 1}


def test_descr_is_a_printable_data_card(loaded):
    ds = load_stocki(conn=loaded)

    assert "windows" in ds.DESCR.lower()
    assert "AAPL" in ds.DESCR


def test_subsets_split_chronologically_without_overlap(loaded):
    train = load_stocki(subset="train", test_days=2, conn=loaded)
    test = load_stocki(subset="test", test_days=2, conn=loaded)

    assert train.timestamps.max() < test.timestamps.min()
    assert len(train.data) + len(test.data) == len(load_stocki(conn=loaded).data)


def test_window_and_horizon_are_parameters(loaded):
    ds = load_stocki(window=16, horizon=3, conn=loaded)

    assert ds.data.shape == (10 * (78 - 16 - 3 + 1), 16, 8)


def test_channels_first_for_torch(loaded):
    ds = load_stocki(channels_first=True, conn=loaded)

    assert ds.data.shape[1:] == (8, 32)


def test_as_frame_adds_a_tidy_frame(loaded):
    ds = load_stocki(as_frame=True, conn=loaded)

    assert isinstance(ds.frame, pd.DataFrame)
    assert len(ds.frame) == len(ds.target)
    assert "target" in ds.frame.columns


def test_a_ticker_filter_reaches_the_windows(loaded):
    ds = load_stocki(tickers="NVDA", conn=loaded)

    assert set(np.unique(ds.ticker)) == {"NVDA"}


def test_an_impossible_window_is_rejected_at_the_call(loaded):
    with pytest.raises(ValueError):
        load_stocki(window=1, conn=loaded)


# --- coverage and describe ------------------------------------------------


def test_coverage_lists_every_session(loaded):
    frame = coverage(conn=loaded)

    assert len(frame) == len(NVDA_DAYS) + len(AAPL_DAYS)
    assert set(frame["bar_count"]) == {78}


def test_describe_reports_the_gap_without_loading_windows(loaded):
    card = describe(conn=loaded)

    assert "AAPL" in card
    assert "4" in card  # AAPL contributes 4 of the 6 sessions
