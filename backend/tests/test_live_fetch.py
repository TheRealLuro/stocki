"""Fetching a live session, end to end, with the network stubbed out.

The claim being tested is narrow and total: a file `stocki fetch` writes is a
file `stocki ingest` accepts. So the assertions here read the written file back
through the real `ingest.reader` and run the real `ingest.validate` over it,
rather than inspecting what the fetch code believes it produced.
"""

import csv
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pytest

from stocki.ingest.columns import CSV_COLUMNS
from stocki.ingest.reader import read_session
from stocki.ingest.validate import EXPECTED_BARS, validate_session
from stocki.live.fetch import (
    existing_name_cn,
    fetch_sessions,
    parse_intraday,
    verify_written,
)

from .conftest import make_rows, write_session

FIXTURES = Path(__file__).parent / "fixtures" / "alphavantage"
SESSION_DATE = date(2026, 8, 17)


def fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def intraday_payload(days, *, bars=EXPECTED_BARS, base=300.0):
    """A TIME_SERIES_INTRADAY response: bars keyed by naive US/Eastern start."""
    series = {}
    for offset, day in enumerate(days):
        opened = datetime.combine(day, time(9, 30))
        for i in range(bars):
            stamp = opened + timedelta(minutes=5 * i)
            price = base + offset + i * 0.1
            series[stamp.strftime("%Y-%m-%d %H:%M:%S")] = {
                "1. open": f"{price:.4f}",
                "2. high": f"{price + 0.25:.4f}",
                "3. low": f"{price - 0.25:.4f}",
                "4. close": f"{price + 0.05:.4f}",
                "5. volume": str(1000 + i),
            }
    return {
        "Meta Data": {"2. Symbol": "NVDA", "6. Time Zone": "US/Eastern"},
        "Time Series (5min)": series,
    }


class FakeAlphaVantage:
    """The client interface, served from the recorded fixtures."""

    def __init__(self, days=(SESSION_DATE,), bars=EXPECTED_BARS):
        self.payload = intraday_payload(days, bars=bars)
        self.requests_made = 0
        self.cache_hits = 0
        self.calls = []

    def _log(self, name):
        self.calls.append(name)
        self.requests_made += 1

    def intraday(self, symbol, **_):
        self._log("intraday")
        return self.payload

    def overview(self, symbol):
        self._log("overview")
        return fixture("AAPL_OVERVIEW.json")

    def income_statement(self, symbol):
        self._log("income")
        return fixture("AAPL_INCOME_STATEMENT.json")

    def balance_sheet(self, symbol):
        self._log("balance")
        return fixture("AAPL_BALANCE_SHEET.json")

    def cash_flow(self, symbol):
        self._log("cash")
        return fixture("AAPL_CASH_FLOW.json")

    def global_quote(self, symbol):
        self._log("quote")
        return {"Global Quote": {"05. price": "311.5000"}}

    def news(self, tickers, **_):
        self._log("news")
        return fixture("NEWS_SENTIMENT.json")


@pytest.fixture
def fetched(tmp_path):
    """One NVDA session written into an otherwise empty data directory."""
    client = FakeAlphaVantage()
    report = fetch_sessions(
        ["NVDA"], dates=[SESSION_DATE], data_dir=tmp_path, client=client
    )
    return report, tmp_path, client


# --- parsing --------------------------------------------------------------


def test_bars_are_stamped_in_utc_not_eastern():
    """09:30 Eastern is the 13:30Z row every committed file opens with."""
    sessions = parse_intraday(intraday_payload([SESSION_DATE]))
    bars = sessions[SESSION_DATE]

    assert len(bars) == EXPECTED_BARS
    assert bars[0].timestamp == datetime(2026, 8, 17, 13, 30, tzinfo=UTC)
    assert bars[-1].timestamp == datetime(2026, 8, 17, 19, 55, tzinfo=UTC)


def test_a_missing_series_says_so():
    with pytest.raises(ValueError, match="no '5min' series"):
        parse_intraday({"Meta Data": {}})


# --- the written file -----------------------------------------------------


def test_it_writes_one_file_per_session(fetched):
    report, data_dir, _ = fetched

    assert report.ok
    assert len(report.written) == 1
    assert (data_dir / "NVDA" / "day1.csv").is_file()


def test_the_header_is_the_committed_header(fetched):
    _, data_dir, _ = fetched

    with (data_dir / "NVDA" / "day1.csv").open(newline="", encoding="utf-8") as handle:
        header = tuple(next(csv.reader(handle)))

    assert header == CSV_COLUMNS


def test_the_file_passes_the_ingest_validator(fetched):
    """The whole point: ingest accepts what fetch wrote, unmodified."""
    _, data_dir, _ = fetched

    assert verify_written(data_dir / "NVDA" / "day1.csv") == []


def test_it_reads_back_as_78_typed_bars(fetched):
    _, data_dir, _ = fetched

    session = read_session(data_dir / "NVDA" / "day1.csv")

    assert len(session.rows) == EXPECTED_BARS
    assert session.rows[0]["timestamp"] == datetime(2026, 8, 17, 13, 30, tzinfo=UTC)
    assert isinstance(session.rows[0]["volume"], int)
    assert session.rows[0]["ticker"] == "NVDA"


def test_hour_and_minute_follow_the_utc_stamp(fetched):
    _, data_dir, _ = fetched

    row = read_session(data_dir / "NVDA" / "day1.csv").rows[0]

    assert (row["hour"], row["minute"]) == (13, 30)


def test_fundamentals_land_in_their_columns(fetched):
    """Spot-check across the four statement groups and the analyst counts."""
    _, data_dir, _ = fetched

    row = read_session(data_dir / "NVDA" / "day1.csv").rows[0]

    assert row["inc_total_revenue"] == 416161000000.0
    assert row["bs_invested_capital"] == 172390000000.0
    assert row["cf_begin_cash"] == 29943000000.0
    assert row["rec_strong_buy"] == 6.0


def test_unavailable_columns_are_null_not_zero(fetched):
    _, data_dir, _ = fetched

    row = read_session(data_dir / "NVDA" / "day1.csv").rows[0]

    assert row["si_employees"] is None
    assert row["si_short_ratio"] is None
    assert row["cf_deferred_tax"] is None


def test_static_columns_repeat_on_every_bar(fetched):
    """As they do in the committed files -- `v_fundamentals` re-collapses them."""
    _, data_dir, _ = fetched

    rows = read_session(data_dir / "NVDA" / "day1.csv").rows

    assert len({row["bs_total_assets"] for row in rows}) == 1
    assert len({row["close"] for row in rows}) > 1


def test_current_price_defaults_to_the_last_close(fetched):
    """GLOBAL_QUOTE costs a request per ticker, so it is opt-in."""
    _, data_dir, client = fetched

    row = read_session(data_dir / "NVDA" / "day1.csv").rows[0]

    assert "quote" not in client.calls
    assert row["si_current_price"] == pytest.approx(300.0 + 77 * 0.1 + 0.05)


def test_the_quote_flag_spends_the_request(tmp_path):
    client = FakeAlphaVantage()

    fetch_sessions(
        ["NVDA"], dates=[SESSION_DATE], data_dir=tmp_path, client=client, with_quote=True
    )

    row = read_session(tmp_path / "NVDA" / "day1.csv").rows[0]
    assert "quote" in client.calls
    assert row["si_current_price"] == 311.5


# --- day numbering --------------------------------------------------------


def test_it_continues_the_existing_day_numbering(tmp_path):
    """A repo already holding days 1-2 gets day 3, not a second day 1."""
    write_session(tmp_path, ticker="NVDA", day=1)
    write_session(tmp_path, ticker="NVDA", day=2)

    fetch_sessions(
        ["NVDA"], dates=[SESSION_DATE], data_dir=tmp_path, client=FakeAlphaVantage()
    )

    assert (tmp_path / "NVDA" / "day3.csv").is_file()


def test_every_ticker_gets_the_same_number_for_the_same_date(tmp_path):
    """The property the chronological train/test split depends on."""
    client = FakeAlphaVantage()

    fetch_sessions(
        ["NVDA", "AAPL"], dates=[SESSION_DATE], data_dir=tmp_path, client=client
    )

    assert (tmp_path / "NVDA" / "day1.csv").is_file()
    assert (tmp_path / "AAPL" / "day1.csv").is_file()


def test_two_sessions_get_consecutive_numbers(tmp_path):
    days = [date(2026, 8, 17), date(2026, 8, 18)]
    client = FakeAlphaVantage(days=days)

    report = fetch_sessions(["NVDA"], dates=days, data_dir=tmp_path, client=client)

    assert len(report.written) == 2
    assert (tmp_path / "NVDA" / "day1.csv").is_file()
    assert (tmp_path / "NVDA" / "day2.csv").is_file()


# --- the refusals ---------------------------------------------------------


def test_an_existing_file_is_not_overwritten(tmp_path):
    """conftest writes day 1 as 2026-08-05, so re-fetching that date resolves to
    the file that is already there."""
    already = date(2026, 8, 5)
    write_session(tmp_path, ticker="NVDA", day=1)
    before = (tmp_path / "NVDA" / "day1.csv").read_bytes()

    report = fetch_sessions(
        ["NVDA"],
        dates=[already],
        data_dir=tmp_path,
        client=FakeAlphaVantage(days=[already]),
    )

    assert (tmp_path / "NVDA" / "day1.csv").read_bytes() == before
    assert any("already exists" in why for why in report.skipped.values())


def test_overwrite_replaces_it(tmp_path):
    already = date(2026, 8, 5)
    write_session(tmp_path, ticker="NVDA", day=1)
    before = (tmp_path / "NVDA" / "day1.csv").read_bytes()

    report = fetch_sessions(
        ["NVDA"],
        dates=[already],
        data_dir=tmp_path,
        client=FakeAlphaVantage(days=[already]),
        overwrite=True,
    )

    assert report.ok and len(report.written) == 1
    assert (tmp_path / "NVDA" / "day1.csv").read_bytes() != before
    assert verify_written(tmp_path / "NVDA" / "day1.csv") == []


def test_a_half_finished_session_is_skipped(tmp_path):
    """The market is still open: 40 bars is not a session, and writing it would
    put a short day into a dataset that assumes 78."""
    client = FakeAlphaVantage(bars=40)

    report = fetch_sessions(
        ["NVDA"], dates=[SESSION_DATE], data_dir=tmp_path, client=client
    )

    assert report.written == []
    assert "40 of 78 bars" in report.skipped["NVDA/2026-08-17"]
    assert not (tmp_path / "NVDA").exists()


def test_allow_partial_writes_the_open_session(tmp_path):
    """Which is what running the model on today's market needs."""
    client = FakeAlphaVantage(bars=40)

    report = fetch_sessions(
        ["NVDA"],
        dates=[SESSION_DATE],
        data_dir=tmp_path,
        client=client,
        allow_partial=True,
    )

    assert report.ok and len(report.written) == 1
    session = read_session(tmp_path / "NVDA" / "day1.csv")
    assert len(session.rows) == 40
    assert validate_session(session, expected_bars=40) == []


def test_a_holiday_is_skipped_not_invented(tmp_path):
    client = FakeAlphaVantage(days=[date(2026, 8, 18)])

    report = fetch_sessions(
        ["NVDA"], dates=[SESSION_DATE], data_dir=tmp_path, client=client
    )

    assert report.written == []
    assert "no bars returned" in report.skipped["NVDA/2026-08-17"]


def test_one_failing_ticker_does_not_lose_the_others(tmp_path):
    class HalfBroken(FakeAlphaVantage):
        def intraday(self, symbol, **kwargs):
            if symbol == "NVDA":
                raise RuntimeError("upstream said no")
            return super().intraday(symbol, **kwargs)

    report = fetch_sessions(
        ["NVDA", "AAPL"], dates=[SESSION_DATE], data_dir=tmp_path, client=HalfBroken()
    )

    assert len(report.written) == 1
    assert "upstream said no" in report.skipped["NVDA/2026-08-17"]
    assert (tmp_path / "AAPL" / "day1.csv").is_file()


def test_dry_run_validates_but_writes_nothing(tmp_path):
    report = fetch_sessions(
        ["NVDA"],
        dates=[SESSION_DATE],
        data_dir=tmp_path,
        client=FakeAlphaVantage(),
        dry_run=True,
    )

    assert report.ok and len(report.written) == 1
    assert not (tmp_path / "NVDA").exists()


# --- carried-over identity ------------------------------------------------


def test_the_chinese_name_is_carried_over_from_disk(tmp_path):
    """Alpha Vantage has no such field. Reusing what is already recorded beats
    inventing one, and beats dropping the column for existing tickers."""
    rows = make_rows(ticker="NVDA", day=1)
    for row in rows:
        row["thsname_cn"] = "英伟达"
    write_session(tmp_path, ticker="NVDA", day=1, rows=rows)

    assert existing_name_cn(tmp_path, "NVDA") == "英伟达"

    fetch_sessions(
        ["NVDA"], dates=[SESSION_DATE], data_dir=tmp_path, client=FakeAlphaVantage()
    )

    assert read_session(tmp_path / "NVDA" / "day2.csv").rows[0]["thsname_cn"] == "英伟达"


def test_an_unknown_ticker_gets_an_empty_name_not_a_guess(tmp_path):
    fetch_sessions(
        ["NVDA"], dates=[SESSION_DATE], data_dir=tmp_path, client=FakeAlphaVantage()
    )

    assert read_session(tmp_path / "NVDA" / "day1.csv").rows[0]["thsname_cn"] is None


# --- the request budget ---------------------------------------------------


def test_news_costs_one_request_for_the_whole_universe(tmp_path):
    """4 fundamentals + 1 intraday per ticker, and a single shared news call."""
    client = FakeAlphaVantage()

    fetch_sessions(
        ["NVDA", "AAPL"], dates=[SESSION_DATE], data_dir=tmp_path, client=client
    )

    assert client.calls.count("news") == 1
    assert client.calls.count("intraday") == 2


def test_each_session_gets_only_its_own_news(tmp_path):
    """One call covers several days, so the feed has to be split by date as well
    as by ticker -- otherwise every day of a multi-day fetch reports the whole
    period's news_count."""
    days = [date(2026, 8, 24), date(2026, 8, 25)]
    client = FakeAlphaVantage(days=days)

    fetch_sessions(["AAPL"], dates=days, data_dir=tmp_path, client=client)

    counts = [
        read_session(tmp_path / "AAPL" / f"day{n}.csv").rows[0]["news_count"]
        for n in (1, 2)
    ]

    assert counts == [5, 5], "the fixture holds five AAPL articles on each date"


def test_skipping_fundamentals_leaves_those_columns_empty(tmp_path):
    client = FakeAlphaVantage()

    fetch_sessions(
        ["NVDA"],
        dates=[SESSION_DATE],
        data_dir=tmp_path,
        client=client,
        with_fundamentals=False,
        with_news=False,
    )

    assert client.calls == ["intraday"]
    row = read_session(tmp_path / "NVDA" / "day1.csv").rows[0]
    assert row["inc_total_revenue"] is None
    assert row["close"] is not None  # the bars are still real
    assert verify_written(tmp_path / "NVDA" / "day1.csv") == []
