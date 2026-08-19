"""The contract Nathaniel builds against: routes, validation, and hardening."""

import io
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from stocki.api.main import create_app
from stocki.ingest.load import ingest_directory

from .conftest import write_session

pytestmark = pytest.mark.db

V1 = "/api/v1"


@pytest.fixture
def client(db, db_settings, tmp_path):
    for day in range(1, 7):
        write_session(tmp_path, ticker="NVDA", day=day)
    for day in range(3, 7):
        write_session(tmp_path, ticker="AAPL", day=day)
    assert ingest_directory(db, tmp_path).ok

    settings = replace(
        db_settings,
        cors_origins=["http://localhost:5173"],
        rate_limit_per_minute=1000,
        dataset_rate_limit_per_minute=1000,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


# --- meta -----------------------------------------------------------------


def test_health_reports_the_database_and_a_bar_count(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["bar_count"] == 10 * 78


def test_tickers_lists_the_universe(client):
    body = client.get(f"{V1}/tickers").json()

    assert [t["ticker"] for t in body] == ["AAPL", "NVDA"]
    assert body[0]["session_count"] == 4


def test_coverage_exposes_the_gap_as_data(client):
    body = client.get(f"{V1}/coverage").json()

    aapl_days = sorted(row["day"] for row in body if row["ticker"] == "AAPL")
    assert aapl_days == [3, 4, 5, 6]


def test_openapi_is_served_so_the_frontend_can_generate_a_client(client):
    assert client.get("/openapi.json").status_code == 200


# --- bars -----------------------------------------------------------------


def test_one_session_mirrors_the_csv_slice(client):
    body = client.get(f"{V1}/bars/NVDA/3").json()

    assert len(body) == 78
    assert set(body[0]) == {"ticker", "day", "timestamp", "open", "high", "low", "close", "volume"}


def test_a_session_that_was_never_collected_is_a_404(client):
    response = client.get(f"{V1}/bars/AAPL/1")

    assert response.status_code == 404
    assert "AAPL" in response.json()["detail"]


def test_bars_list_paginates(client):
    body = client.get(f"{V1}/bars", params={"ticker": "NVDA", "limit": 10}).json()

    assert len(body["items"]) == 10
    assert body["total"] == 6 * 78
    assert body["limit"] == 10


def test_bars_list_defaults_to_a_bounded_page(client):
    body = client.get(f"{V1}/bars").json()

    assert len(body["items"]) == 200


def test_a_limit_above_the_cap_is_rejected(client):
    assert client.get(f"{V1}/bars", params={"limit": 5000}).status_code == 422


def test_fundamentals_return_a_single_object(client):
    body = client.get(f"{V1}/fundamentals/NVDA/3").json()

    assert body["ticker"] == "NVDA"
    assert body["day"] == 3
    assert "si_market_cap" in body["fundamentals"]


def test_news_returns_the_count_and_headline(client):
    body = client.get(f"{V1}/news/NVDA/3").json()

    assert body["news_count"] == 3


# --- validation -----------------------------------------------------------


def test_an_unknown_ticker_is_a_422_that_lists_the_valid_ones(client):
    response = client.get(f"{V1}/bars/NOPE/3")

    assert response.status_code == 422
    body = response.json()
    assert "NVDA" in str(body["detail"])


@pytest.mark.parametrize("day", [0, 21, -5])
def test_a_day_outside_the_dataset_is_rejected(client, day):
    assert client.get(f"{V1}/bars/NVDA/{day}").status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"window": 4},
        {"window": 200},
        {"horizon": 0},
        {"horizon": 99},
        {"limit": 9999},
    ],
)
def test_dataset_parameters_are_bounded(client, params):
    assert client.get(f"{V1}/dataset/windows", params=params).status_code == 422


def test_every_error_body_has_the_same_shape(client):
    body = client.get(f"{V1}/bars/NOPE/3").json()

    assert set(body) == {"error", "detail", "request_id"}


# --- dataset --------------------------------------------------------------


def test_dataset_stats_report_shape_and_balance(client):
    body = client.get(f"{V1}/dataset/stats").json()

    assert body["n_windows"] == 10 * 46
    assert body["shape"] == [32, 8]
    assert 0.0 <= body["up_rate"] <= 1.0
    assert body["feature_names"][0] == "open"


def test_dataset_stats_follow_the_window_parameters(client):
    body = client.get(f"{V1}/dataset/stats", params={"window": 16, "horizon": 3}).json()

    assert body["n_windows"] == 10 * (78 - 16 - 3 + 1)


def test_windows_as_json_line_up_with_their_labels(client):
    body = client.get(f"{V1}/dataset/windows", params={"limit": 5}).json()

    assert np.array(body["data"]).shape == (5, 32, 8)
    assert len(body["target"]) == 5


def test_windows_as_npz_load_straight_into_numpy(client):
    response = client.get(f"{V1}/dataset/windows", params={"limit": 5, "format": "npz"})

    with np.load(io.BytesIO(response.content), allow_pickle=False) as archive:
        assert archive["data"].shape == (5, 32, 8)
        assert archive["target"].shape == (5,)


def test_the_dataset_endpoint_serves_the_same_code_the_loader_uses(client, db):
    from stocki.datasets import load_stocki

    served = client.get(f"{V1}/dataset/windows", params={"limit": 3, "ticker": "NVDA"}).json()
    imported = load_stocki(tickers="NVDA", conn=db)

    assert np.allclose(np.array(served["data"]), imported.data[:3], atol=1e-6)


def test_subset_splits_are_available_over_http(client):
    train = client.get(f"{V1}/dataset/stats", params={"subset": "train", "test_days": 2}).json()
    test = client.get(f"{V1}/dataset/stats", params={"subset": "test", "test_days": 2}).json()

    assert train["n_windows"] + test["n_windows"] == 10 * 46


# --- hardening ------------------------------------------------------------


def test_cors_allows_the_configured_origin(client):
    response = client.get(f"{V1}/tickers", headers={"Origin": "http://localhost:5173"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_does_not_echo_an_unknown_origin(client):
    response = client.get(f"{V1}/tickers", headers={"Origin": "http://evil.example"})

    assert response.headers.get("access-control-allow-origin") != "http://evil.example"


def test_security_headers_are_set(client):
    headers = client.get("/health").headers

    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"


def test_the_rate_limiter_returns_429(db, db_settings):
    settings = replace(db_settings, rate_limit_per_minute=3, dataset_rate_limit_per_minute=3)

    with TestClient(create_app(settings)) as limited:
        codes = [limited.get(f"{V1}/tickers").status_code for _ in range(6)]

    assert 429 in codes
    assert codes[0] == 200


def test_the_dataset_endpoint_has_its_own_tighter_budget(db, db_settings, tmp_path):
    write_session(tmp_path, ticker="NVDA", day=1)
    ingest_directory(db, tmp_path)
    settings = replace(db_settings, rate_limit_per_minute=1000, dataset_rate_limit_per_minute=2)

    with TestClient(create_app(settings)) as limited:
        codes = [limited.get(f"{V1}/dataset/stats").status_code for _ in range(4)]

    assert codes.count(429) >= 1
    assert limited.get(f"{V1}/tickers").status_code == 200


def test_an_unexpected_failure_returns_an_id_and_never_a_traceback(client):
    from stocki.api.deps import get_connection

    def explode():
        raise RuntimeError("postgresql://stocki:hunter2@localhost/stocki blew up")

    client.app.dependency_overrides[get_connection] = explode
    try:
        response = client.get(f"{V1}/tickers")
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 500
    body = response.json()
    assert body["request_id"]
    assert "hunter2" not in response.text
    assert "Traceback" not in response.text
    assert "RuntimeError" not in response.text
