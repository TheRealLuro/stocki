"""Alpha Vantage payloads -> the 118 CSV columns.

The fixtures under ``fixtures/alphavantage/`` are real responses, trimmed to the
reports the mapping reads. That matters: the derivations in ``live.fields`` were
worked out *from* the committed files, so the last test here closes the loop and
checks that feeding the provider's own numbers back through the mapping
reproduces ``data/AAPL/day13.csv`` line for line.
"""

import csv
import json
from pathlib import Path

import pytest

from stocki.live import fields
from stocki.live.client import parse_number

FIXTURES = Path(__file__).parent / "fixtures" / "alphavantage"
REPO_DATA = Path(__file__).resolve().parents[2] / "data"


def payload(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def overview():
    return payload("AAPL_OVERVIEW.json")


@pytest.fixture(scope="module")
def income():
    return payload("AAPL_INCOME_STATEMENT.json")


@pytest.fixture(scope="module")
def balance():
    return payload("AAPL_BALANCE_SHEET.json")


@pytest.fixture(scope="module")
def cash():
    return payload("AAPL_CASH_FLOW.json")


@pytest.fixture(scope="module")
def si(overview, balance):
    return fields.stock_info_columns(
        overview, fields.report(balance, "quarterly"), current_price=309.9
    )


@pytest.fixture(scope="module")
def inc(income, balance):
    return fields.income_columns(fields.report(income), fields.report(balance))


@pytest.fixture(scope="module")
def bs(balance):
    return fields.balance_columns(fields.report(balance))


@pytest.fixture(scope="module")
def cf(cash, balance):
    return fields.cashflow_columns(fields.report(cash), fields.report(balance))


# --- the number parser ----------------------------------------------------


@pytest.mark.parametrize("raw", ["None", "-", "", None, "nan", "not a number"])
def test_missing_values_become_none(raw):
    """Alpha Vantage writes an absent number as the string "None"."""
    assert parse_number(raw) is None


def test_numbers_parse():
    assert parse_number("416161000000") == 416161000000.0
    assert parse_number("-0.1418%") == pytest.approx(-0.1418)


# --- si_*: the current snapshot -------------------------------------------


def test_total_cash_is_the_most_recent_quarter_not_the_year(si):
    """39544000000 + 22855000000. The annual figure is 35934000000, which the
    files disagree with -- si_* is a snapshot, so it reads the MRQ."""
    assert si["si_total_cash"] == 62399000000.0


def test_dividend_yield_is_converted_to_percent(si):
    """The provider sends 0.0034; the files carry 0.35-style percents."""
    assert si["si_dividend_yield"] == pytest.approx(0.34)


def test_ownership_percentages_are_converted_to_fractions(si):
    """The opposite direction: 1.648 percent arrives, 0.01648 is stored."""
    assert si["si_insider_pct"] == pytest.approx(0.01648)
    assert si["si_inst_pct"] == pytest.approx(0.66417)


def test_payout_ratio_is_dividend_over_eps(si):
    assert si["si_payout_ratio"] == pytest.approx(1.05 / 8.75)


def test_gross_margin_is_derived_from_the_ttm_pair(si):
    assert si["si_gross_margins"] == pytest.approx(227123003000 / 466822988000)


def test_debt_to_equity_is_a_percentage_of_mrq_equity(si):
    assert si["si_debt_to_equity"] == pytest.approx(84307000000 / 107520000000 * 100)


def test_current_and_quick_ratios_come_from_the_mrq(si):
    assert si["si_current_ratio"] == pytest.approx(149818000000 / 149326000000)
    assert si["si_quick_ratio"] == pytest.approx(
        (149818000000 - 11092000000) / 149326000000
    )


def test_recommendation_mean_weights_strong_buy_lowest(si):
    counts = 6 + 22 + 14 + 2 + 2
    weighted = 6 * 1 + 22 * 2 + 14 * 3 + 2 * 4 + 2 * 5
    assert si["si_num_analysts"] == counts
    assert si["si_recommendation_mean"] == pytest.approx(weighted / counts)


def test_unavailable_columns_are_none_not_zero(si, bs, cf):
    """A zero would be a claim about the company. None becomes SQL NULL."""
    everything = {**si, **bs, **cf}
    for column in fields.UNAVAILABLE:
        assert everything[column] is None, column


# --- the statement columns ------------------------------------------------


def test_income_columns_are_copied_verbatim(inc):
    assert inc["inc_total_revenue"] == 416161000000.0
    assert inc["inc_gross_profit"] == 195201000000.0
    assert inc["inc_net_income"] == 112010000000.0
    assert inc["inc_depreciation"] == 11698000000.0


def test_eps_is_net_income_over_the_balance_sheet_share_count(inc):
    assert inc["inc_diluted_shares"] == 15004697000.0
    assert inc["inc_diluted_eps"] == pytest.approx(112010000000 / 15004697000)


def test_total_debt_adds_the_current_and_noncurrent_parts(bs):
    """20329000000 + 78328000000. `shortLongTermDebtTotal` is a different,
    larger number and is only the fallback."""
    assert bs["bs_total_debt"] == 98657000000.0


def test_derived_balance_columns(bs):
    assert bs["bs_net_debt"] == 62723000000.0
    assert bs["bs_cash_short_inv"] == 54697000000.0
    assert bs["bs_working_capital"] == -17674000000.0
    assert bs["bs_invested_capital"] == 172390000000.0
    assert bs["bs_total_capitalization"] == 152061000000.0
    assert bs["bs_tangible_book_value"] == 73733000000.0


def test_outflows_are_written_negative(cf):
    assert cf["cf_capex"] == -12715000000.0
    assert cf["cf_dividends_paid"] == -15421000000.0
    assert cf["cf_stock_repurchases"] == -90711000000.0


def test_cash_flow_closes_on_the_balance_sheet(cf):
    """Net change is operating + investing + financing, and the closing balance
    is the balance sheet's cash, so the opening balance falls out."""
    assert cf["cf_free"] == 98767000000.0
    assert cf["cf_net_change"] == 5991000000.0
    assert cf["cf_end_cash"] == 35934000000.0
    assert cf["cf_begin_cash"] == 29943000000.0


def test_missing_reports_give_none_not_a_crash():
    """A ticker with no fundamentals still produces a writable row."""
    columns = {
        **fields.stock_info_columns({}, {}),
        **fields.income_columns({}, {}),
        **fields.balance_columns({}),
        **fields.cashflow_columns({}, {}),
        **fields.recommendation_columns({}),
    }
    assert set(columns.values()) == {None}


# --- news -----------------------------------------------------------------


def test_one_feed_splits_into_per_ticker_articles():
    """One NEWS_SENTIMENT call covers every ticker, which is what keeps a daily
    refresh inside the free plan's 25 requests."""
    from datetime import UTC, datetime

    feed = payload("NEWS_SENTIMENT.json")
    start = datetime(2026, 8, 24, 0, 0, tzinfo=UTC)
    end = datetime(2026, 8, 26, 0, 0, tzinfo=UTC)

    split = fields.news_by_ticker(feed, ["AAPL", "NVDA"], start, end)

    assert set(split) == {"AAPL", "NVDA"}
    assert split["AAPL"], "the fixture was pulled for AAPL"
    assert all(fields.article_time(a) is not None for a in split["AAPL"])


def test_articles_outside_the_session_window_are_dropped():
    from datetime import UTC, datetime

    feed = payload("NEWS_SENTIMENT.json")
    long_ago = datetime(2020, 1, 1, tzinfo=UTC)

    split = fields.news_by_ticker(feed, ["AAPL"], long_ago, datetime(2020, 1, 2, tzinfo=UTC))

    assert split["AAPL"] == []
    assert fields.news_columns([]) == {"news_count": 0, "news_latest_headline": ""}


def test_headline_is_the_latest_of_the_session():
    from datetime import UTC, datetime

    feed = payload("NEWS_SENTIMENT.json")
    split = fields.news_by_ticker(
        feed,
        ["AAPL"],
        datetime(2026, 8, 24, tzinfo=UTC),
        datetime(2026, 8, 26, tzinfo=UTC),
    )
    articles = split["AAPL"]

    columns = fields.news_columns(articles)

    assert columns["news_count"] == len(articles)
    assert columns["news_latest_headline"] == articles[-1]["title"]


# --- the loop back to the committed data ----------------------------------

#: Columns that read the annual report, which does not move between fetches.
#: The si_* snapshot columns drift with the share price, so they are not here.
STABLE_AGAINST_DISK = (
    "inc_total_revenue",
    "inc_cost_of_revenue",
    "inc_gross_profit",
    "inc_operating_income",
    "inc_net_income",
    "inc_sga",
    "inc_tax_provision",
    "inc_pretax_income",
    "inc_norm_income",
    "inc_depreciation",
    "inc_diluted_shares",
    "bs_total_assets",
    "bs_total_liabilities",
    "bs_stockholders_equity",
    "bs_total_debt",
    "bs_net_debt",
    "bs_cash",
    "bs_cash_short_inv",
    "bs_inventory",
    "bs_long_term_debt",
    "bs_current_liabilities",
    "bs_current_assets",
    "bs_working_capital",
    "bs_retained_earnings",
    "bs_common_stock",
    "bs_capital_stock",
    "bs_total_equity",
    "bs_tangible_book_value",
    "bs_invested_capital",
    "bs_total_capitalization",
    "cf_operating",
    "cf_free",
    "cf_capex",
    "cf_dividends_paid",
    "cf_stock_repurchases",
    "cf_financing",
    "cf_investing",
    "cf_net_change",
    "cf_end_cash",
    "cf_begin_cash",
    "cf_stock_comp",
    "cf_depreciation",
    "cf_net_income_ops",
)


def test_the_mapping_reproduces_the_committed_file(inc, bs, cf):
    """Feed the provider's numbers through the mapping and the answer is the
    file that is already in the repo.

    This is the whole claim of `stocki fetch` in one assertion: a fetched
    session is not merely shaped like `data/AAPL/day13.csv`, it holds the same
    values.
    """
    committed = REPO_DATA / "AAPL" / "day13.csv"
    if not committed.is_file():
        pytest.skip(f"{committed} is not in this checkout")

    with committed.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    mapped = {**inc, **bs, **cf}
    for column in STABLE_AGAINST_DISK:
        assert mapped[column] == pytest.approx(float(row[column])), column
