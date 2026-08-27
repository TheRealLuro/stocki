"""The Alpha Vantage client: envelopes, budget, cache, credentials.

Alpha Vantage answers HTTP 200 for its failures, so the interesting behaviour is
all in reading the body. Nothing here touches the network -- `_get` is stubbed.
"""

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from stocki.config import Settings
from stocki.errors import StockiCredentialsError, StockiPlanError, StockiQuotaError
from stocki.live.client import (
    AlphaVantage,
    AlphaVantageError,
    months_between,
    parse_number,
)

PREMIUM = {
    "Information": "Thank you for using Alpha Vantage! This is a premium endpoint. "
    "You may subscribe to any of the premium plans at "
    "https://www.alphavantage.co/premium/ to instantly unlock all premium endpoints"
}
THROTTLED = {
    "Information": "Thank you for using Alpha Vantage! Please consider spreading out "
    "your free API requests more sparingly (1 request per second). You may subscribe "
    "to any of the premium plans to lift the free key rate limit (25 requests per day)"
}
BAD_SYMBOL = {"Error Message": "Invalid API call. Please retry or visit the documentation"}
GOOD = {"Symbol": "NVDA", "Name": "NVIDIA Corp"}


@pytest.fixture
def settings(tmp_path):
    return replace(
        Settings(),
        alphavantage_key="test-key",
        live_cache_dir=tmp_path / "cache",
        live_min_interval=0.0,
    )


def client_returning(settings, *payloads, **kwargs):
    """A client whose transport replays `payloads`, one per call."""
    client = AlphaVantage(settings, min_interval=0.0, **kwargs)
    bodies = [json.dumps(p).encode() for p in payloads]
    calls = {"n": 0}

    def fake_get(params):
        index = min(calls["n"], len(bodies) - 1)
        calls["n"] += 1
        return bodies[index]

    client._get = fake_get
    client.transport_calls = calls
    return client


# --- envelopes ------------------------------------------------------------


def test_a_good_body_comes_straight_back(settings):
    client = client_returning(settings, GOOD)

    assert client.request("OVERVIEW", symbol="NVDA") == GOOD


def test_a_premium_gate_is_not_reported_as_a_rate_limit(settings):
    """Both messages advertise the premium plans, so the endpoint gate has to be
    told apart by its own phrase -- otherwise it looks like a quota you can wait
    out, and the caller retries forever."""
    client = client_returning(settings, PREMIUM)

    with pytest.raises(StockiPlanError, match="does not serve TIME_SERIES_INTRADAY"):
        client.request("TIME_SERIES_INTRADAY", symbol="NVDA")


def test_a_persistent_rate_limit_raises_quota(settings):
    client = client_returning(settings, THROTTLED)

    with pytest.raises(StockiQuotaError, match="rate-limited"):
        client.request("OVERVIEW", symbol="NVDA")


def test_a_burst_limit_is_waited_out(settings, monkeypatch):
    """One throttled answer then a good one: the per-second burst clears."""
    monkeypatch.setattr("stocki.live.client.time.sleep", lambda _: None)
    client = client_returning(settings, THROTTLED, GOOD)

    assert client.request("OVERVIEW", symbol="NVDA") == GOOD
    assert client.transport_calls["n"] == 2


def test_an_error_message_names_the_function(settings):
    client = client_returning(settings, BAD_SYMBOL)

    with pytest.raises(AlphaVantageError, match="rejected OVERVIEW"):
        client.request("OVERVIEW", symbol="NOPE")


def test_a_non_object_body_is_rejected(settings):
    client = client_returning(settings, ["not", "an", "object"])

    with pytest.raises(AlphaVantageError, match="expected an object"):
        client.request("OVERVIEW", symbol="NVDA")


# --- credentials ----------------------------------------------------------


def test_a_missing_key_names_every_variable_it_looked_at(monkeypatch):
    for name in (
        "ALPHA_ADVANTAGE_KEY",
        "ALPHAVANTAGE_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "STOCKI_ALPHAVANTAGE_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    client = AlphaVantage(Settings(), api_key=None)

    with pytest.raises(StockiCredentialsError, match="ALPHA_ADVANTAGE_KEY"):
        client.require_key()


def test_the_key_is_kept_out_of_the_repr():
    """Settings gets printed in error paths; the key must not ride along."""
    settings = replace(Settings(), alphavantage_key="super-secret")

    assert "super-secret" not in repr(settings)


# --- budget ---------------------------------------------------------------


def test_the_budget_stops_the_run_before_the_quota_is_gone(settings):
    client = client_returning(settings, GOOD, budget=2)

    client.request("GLOBAL_QUOTE", symbol="A")
    client.request("GLOBAL_QUOTE", symbol="B")

    with pytest.raises(StockiQuotaError, match="budget of 2"):
        client.request("GLOBAL_QUOTE", symbol="C")


def test_cached_answers_do_not_spend_budget(settings):
    client = client_returning(settings, GOOD, budget=1)

    client.request("OVERVIEW", symbol="NVDA")
    again = client.request("OVERVIEW", symbol="NVDA")

    assert again == GOOD
    assert client.requests_made == 1
    assert client.cache_hits == 1


# --- cache ----------------------------------------------------------------


def test_only_company_reports_are_cached(settings):
    client = client_returning(settings, GOOD)

    client.request("GLOBAL_QUOTE", symbol="NVDA")
    client.request("GLOBAL_QUOTE", symbol="NVDA")

    assert client.cache_hits == 0
    assert client.requests_made == 2


def test_a_stale_cache_entry_is_refetched(settings):
    client = client_returning(settings, GOOD)
    client.request("OVERVIEW", symbol="NVDA")

    path = client._cache_path("OVERVIEW", "NVDA")
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["fetched_at"] = (datetime.now(UTC) - timedelta(days=9)).isoformat()
    path.write_text(json.dumps(stored), encoding="utf-8")

    client.request("OVERVIEW", symbol="NVDA")

    assert client.requests_made == 2


def test_refresh_ignores_a_warm_cache(settings):
    client_returning(settings, GOOD).request("OVERVIEW", symbol="NVDA")
    fresh = client_returning(settings, GOOD, refresh=True)

    fresh.request("OVERVIEW", symbol="NVDA")

    assert fresh.cache_hits == 0


def test_a_corrupt_cache_file_is_ignored_not_fatal(settings):
    client = client_returning(settings, GOOD)
    client.request("OVERVIEW", symbol="NVDA")
    client._cache_path("OVERVIEW", "NVDA").write_text("{ not json", encoding="utf-8")

    assert client.request("OVERVIEW", symbol="NVDA") == GOOD


# --- helpers --------------------------------------------------------------


def test_month_slices_cover_the_range():
    """Intraday history is served a month at a time, so a range that crosses a
    boundary is two requests, not one."""
    assert months_between(date(2026, 8, 3), date(2026, 8, 20)) == ["2026-08"]
    assert months_between(date(2026, 7, 28), date(2026, 9, 2)) == [
        "2026-07",
        "2026-08",
        "2026-09",
    ]


@pytest.mark.parametrize(
    "raw,expected",
    [("1.5", 1.5), ("1,500", 1500.0), ("-0.14%", -0.14), (7, 7.0), (None, None)],
)
def test_number_parsing(raw, expected):
    assert parse_number(raw) == expected
