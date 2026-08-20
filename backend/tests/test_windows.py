"""Windowing, labelling, normalisation, and the chronological split.

These run on hand-built arrays with no database, because this is where an
off-by-one or a leaked future bar would quietly ruin every model that follows.
"""

import numpy as np
import pandas as pd
import pytest

from stocki.datasets.labels import direction_labels
from stocki.datasets.windows import (
    DEFAULT_CHANNELS,
    build_windows,
    derive_channels,
    normalize_windows,
    split_mask,
)

BASE_DAY = pd.Timestamp("2026-08-05 13:30:00+00:00")


def make_frame(closes_by_day, ticker="AAPL"):
    """Tidy bars frame from {day: [close, ...]}; OHLC bracket the close sanely."""
    records = []
    for day, closes in closes_by_day.items():
        start = BASE_DAY + pd.Timedelta(days=day - 1)
        for i, close in enumerate(closes):
            records.append(
                {
                    "ticker": ticker,
                    "day": day,
                    "timestamp": start + pd.Timedelta(minutes=5 * i),
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.6,
                    "close": float(close),
                    "volume": 1000 + i * 7,
                }
            )
    return pd.DataFrame.from_records(records)


# --- labels ---------------------------------------------------------------


def test_label_is_up_when_the_next_close_is_higher():
    close = np.array([10.0, 11.0, 12.0, 11.5], dtype=np.float64)

    labels = direction_labels(close, window=2, horizon=1)

    # Windows end at index 1 and 2: close[2] > close[1], then close[3] < close[2].
    assert labels.tolist() == [1, 0]


def test_label_reads_the_bar_after_the_window_not_the_last_bar_inside_it():
    """The classic off-by-one: labelling from inside the window leaks the answer."""
    close = np.array([10.0, 20.0, 5.0], dtype=np.float64)

    labels = direction_labels(close, window=2, horizon=1)

    assert labels.tolist() == [0]  # 5 < 20, even though the window itself rose


def test_horizon_looks_further_ahead():
    """From the same window: down one bar later, up two bars later."""
    close = np.array([10.0, 9.0, 8.0, 30.0], dtype=np.float64)

    assert direction_labels(close, window=2, horizon=1)[0] == 0
    assert direction_labels(close, window=2, horizon=2)[0] == 1


def test_threshold_requires_the_move_to_be_big_enough():
    close = np.array([100.0, 100.0, 100.05], dtype=np.float64)

    assert direction_labels(close, window=2, horizon=1, threshold=0.0).tolist() == [1]
    assert direction_labels(close, window=2, horizon=1, threshold=0.01).tolist() == [0]


def test_labels_are_int8_so_they_drop_straight_into_a_loss_function():
    labels = direction_labels(np.arange(10, dtype=np.float64), window=3, horizon=1)

    assert labels.dtype == np.int8


# --- derived channels -----------------------------------------------------


def test_log_return_uses_only_the_previous_bar():
    frame = make_frame({1: [100.0, 110.0, 121.0]})

    channels = derive_channels(frame)

    assert channels["log_return"][0] == 0.0
    assert channels["log_return"][1] == pytest.approx(np.log(1.1))
    assert channels["log_return"][2] == pytest.approx(np.log(1.1))


def test_bar_local_channels_use_only_that_bar():
    frame = make_frame({1: [100.0]})

    channels = derive_channels(frame)

    assert channels["hl_range"][0] == pytest.approx(1.1 / 100.0)
    assert channels["co_range"][0] == pytest.approx(0.1 / 99.9)


def test_the_default_channel_set_is_the_eight_documented_ones():
    assert DEFAULT_CHANNELS == (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "log_return",
        "hl_range",
        "co_range",
    )


# --- windowing ------------------------------------------------------------


def test_window_count_follows_the_formula():
    frame = make_frame({1: list(np.arange(78, dtype=float) + 100)})

    result = build_windows(frame, window=32, horizon=1)

    assert result.X.shape == (78 - 32 - 1 + 1, 32, 8)


@pytest.mark.parametrize(("window", "horizon"), [(8, 1), (16, 3), (32, 1), (40, 8)])
def test_window_count_holds_across_parameters(window, horizon):
    bars = 78
    frame = make_frame({1: list(np.arange(bars, dtype=float) + 100)})

    result = build_windows(frame, window=window, horizon=horizon)

    assert len(result.X) == bars - window - horizon + 1


def test_no_window_spans_two_sessions():
    """Days are not contiguous; a window across the boundary is meaningless."""
    frame = make_frame({1: list(np.arange(10.0) + 100), 2: list(np.arange(10.0) + 500)})

    result = build_windows(frame, window=4, horizon=1, normalize=None)

    assert len(result.X) == 2 * (10 - 4 - 1 + 1)
    closes = result.X[:, :, DEFAULT_CHANNELS.index("close")]
    straddles = (closes < 200).any(axis=1) & (closes > 400).any(axis=1)
    assert not straddles.any()


def test_a_session_shorter_than_the_window_contributes_nothing():
    frame = make_frame({1: list(np.arange(10.0) + 100), 2: [100.0, 101.0]})

    result = build_windows(frame, window=4, horizon=1)

    assert len(result.X) == 10 - 4 - 1 + 1


def test_metadata_arrays_line_up_with_the_windows():
    frame = make_frame({1: list(np.arange(10.0) + 100), 2: list(np.arange(10.0) + 500)})

    result = build_windows(frame, window=4, horizon=1)

    assert result.ticker.shape == result.y.shape == (len(result.X),)
    assert set(np.unique(result.day)) == {1, 2}
    assert result.timestamps.shape == (len(result.X),)


def test_timestamp_marks_the_last_bar_of_the_window():
    frame = make_frame({1: list(np.arange(10.0) + 100)})

    result = build_windows(frame, window=4, horizon=1)

    expected = (BASE_DAY + pd.Timedelta(minutes=15)).tz_convert(None).to_datetime64()
    assert result.timestamps[0] == expected


def test_timestamps_are_numpy_datetimes_in_utc():
    """Plain datetime64 so numpy can compare and sort them without pandas."""
    result = build_windows(make_frame({1: list(np.arange(10.0) + 100)}), window=4, horizon=1)

    assert result.timestamps.dtype == np.dtype("datetime64[ns]")


def test_channels_first_transposes_for_torch_conv1d():
    frame = make_frame({1: list(np.arange(78.0) + 100)})

    result = build_windows(frame, window=32, horizon=1, channels_first=True)

    assert result.X.shape == (46, 8, 32)


def test_x_is_float32_so_it_feeds_a_model_without_a_copy():
    frame = make_frame({1: list(np.arange(78.0) + 100)})

    assert build_windows(frame, window=32, horizon=1).X.dtype == np.float32


def test_selecting_a_subset_of_channels():
    frame = make_frame({1: list(np.arange(78.0) + 100)})

    result = build_windows(frame, window=32, horizon=1, channels=["close", "volume"])

    assert result.X.shape == (46, 32, 2)
    assert result.feature_names == ("close", "volume")


# --- normalisation --------------------------------------------------------


def test_window_z_centres_every_window_and_channel():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 32, 3)) * 100 + 500

    normalized = normalize_windows(x, "window-z")

    assert np.allclose(normalized.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(normalized.std(axis=1), 1.0, atol=1e-6)


def test_window_z_normalises_each_window_independently():
    """Dataset-wide statistics would leak the test set into the training set."""
    x = np.stack([np.arange(10.0), np.arange(10.0) * 1000]).reshape(2, 10, 1)

    normalized = normalize_windows(x, "window-z")

    assert np.allclose(normalized[0], normalized[1])


def test_a_flat_channel_normalises_to_zero_rather_than_nan():
    x = np.full((2, 10, 1), 7.0)

    normalized = normalize_windows(x, "window-z")

    assert np.isfinite(normalized).all()
    assert np.allclose(normalized, 0.0)


def test_normalize_none_leaves_raw_prices():
    frame = make_frame({1: list(np.arange(78.0) + 100)})

    result = build_windows(frame, window=32, horizon=1, normalize=None)

    assert result.X[0, 0, DEFAULT_CHANNELS.index("close")] == pytest.approx(100.0)


# --- splitting ------------------------------------------------------------


def test_split_puts_the_last_days_in_test():
    days = np.array([1, 5, 16, 17, 20])

    mask = split_mask(days, subset="train", test_days=4)

    assert mask.tolist() == [True, True, True, False, False]


def test_train_and_test_are_disjoint_and_cover_everything():
    days = np.arange(1, 21)

    train = split_mask(days, subset="train", test_days=4)
    test = split_mask(days, subset="test", test_days=4)

    assert not (train & test).any()
    assert (train | test).all()


def test_no_training_window_ends_after_any_test_window_starts():
    """The one assertion that makes the whole split honest."""
    frame = make_frame({d: list(np.arange(20.0) + 100 + d) for d in range(1, 21)})
    result = build_windows(frame, window=8, horizon=1)

    train = result.timestamps[split_mask(result.day, "train", test_days=4)]
    test = result.timestamps[split_mask(result.day, "test", test_days=4)]

    assert train.max() < test.min()
