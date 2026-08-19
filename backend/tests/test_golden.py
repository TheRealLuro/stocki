"""The whole real dataset, end to end.

If someone adds, removes, or edits a data file, this is what tells them.
"""

import numpy as np
import pytest

from stocki.config import get_settings
from stocki.datasets.loaders import coverage, load_panel, load_stocki
from stocki.ingest.load import ingest_directory, verify_directory
from stocki.ingest.reader import find_sessions

pytestmark = pytest.mark.db

TICKERS = ["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "META", "MSFT", "NVDA", "TSLA", "V"]
SESSIONS = 187
BARS_PER_SESSION = 78
TOTAL_BARS = SESSIONS * BARS_PER_SESSION  # 14,586
WINDOWS_PER_SESSION = 78 - 32 - 1 + 1  # 46
TOTAL_WINDOWS = SESSIONS * WINDOWS_PER_SESSION  # 8,602


@pytest.fixture(scope="module")
def real_data_dir():
    data_dir = get_settings().data_dir
    if not find_sessions(data_dir):
        pytest.skip(f"no session files under {data_dir}")
    return data_dir


@pytest.fixture(scope="module")
def full(db_settings, real_data_dir):
    """Ingest the real dataset once, then let every test in this module read it."""
    from stocki.db.session import apply_schema, connect

    conn = connect(db_settings)
    apply_schema(conn)
    conn.execute("TRUNCATE bars_raw")
    conn.commit()

    report = ingest_directory(conn, real_data_dir)
    assert report.ok, report.summary()
    conn.commit()

    yield conn
    conn.close()


def test_every_file_on_disk_is_a_valid_session(real_data_dir):
    """Validation only -- no database needed to know the files are sound."""
    from stocki.ingest.reader import read_session
    from stocki.ingest.validate import validate_session

    paths = find_sessions(real_data_dir)
    problems = [p for path in paths for p in validate_session(read_session(path))]

    assert problems == []
    assert len(paths) == SESSIONS


def test_the_expected_number_of_bars_land_in_postgres(full):
    assert full.execute("SELECT count(*) FROM bars_raw").fetchone()[0] == TOTAL_BARS


def test_the_table_matches_every_csv_on_disk(full, real_data_dir):
    """The round-trip guarantee, over all 187 files."""
    assert verify_directory(full, real_data_dir) == []


def test_the_universe_is_the_ten_tickers(full):
    frame = coverage(conn=full)

    assert sorted(frame["ticker"].unique()) == TICKERS


def test_coverage_reports_the_known_gaps(full):
    frame = coverage(conn=full)
    days = {t: sorted(g["day"]) for t, g in frame.groupby("ticker")}

    assert days["AAPL"] == list(range(13, 21))  # AAPL starts on day 13
    assert days["MSFT"] == list(range(2, 21))  # MSFT is missing day 1
    assert days["NVDA"] == list(range(1, 21))


def test_every_session_holds_78_bars(full):
    frame = coverage(conn=full)

    assert set(frame["bar_count"]) == {BARS_PER_SESSION}


def test_the_default_dataset_shape(full):
    ds = load_stocki(conn=full)

    assert ds.data.shape == (TOTAL_WINDOWS, 32, 8)
    assert ds.target.shape == (TOTAL_WINDOWS,)
    assert ds.data.dtype == np.float32


def test_the_classes_are_not_wildly_imbalanced(full):
    """A 50/50-ish split is expected on 5-minute bars; anything extreme is a bug."""
    ds = load_stocki(conn=full)

    assert 0.3 < ds.target.mean() < 0.7


def test_the_chronological_split_covers_the_dataset_without_leaking(full):
    train = load_stocki(subset="train", conn=full)
    test = load_stocki(subset="test", conn=full)

    assert len(train.data) + len(test.data) == TOTAL_WINDOWS
    assert train.timestamps.max() < test.timestamps.min()
    assert set(np.unique(train.day)) == set(range(1, 17))
    assert set(np.unique(test.day)) == set(range(17, 21))


def test_the_panel_marks_the_uncollected_sessions_as_nan(full):
    panel = load_panel(conn=full)

    assert panel.values.shape == (10, 20, 78, 5)
    aapl = panel.values[panel.tickers.index("AAPL")]
    assert np.isnan(aapl[:12]).all()
    assert not np.isnan(aapl[12:]).any()


def test_no_price_or_volume_is_missing_anywhere(full):
    missing = full.execute(
        "SELECT count(*) FROM bars_raw WHERE open IS NULL OR high IS NULL "
        "OR low IS NULL OR close IS NULL OR volume IS NULL"
    ).fetchone()[0]

    assert missing == 0


def test_the_windows_carry_no_nan_or_inf(full):
    ds = load_stocki(conn=full)

    assert np.isfinite(ds.data).all()
