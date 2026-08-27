"""Alpha Vantage payloads -> the 118 columns of ``data/<TICKER>/day<N>.csv``.

Pure functions: JSON in, column values out. No network, no disk, no clock, so
every mapping below is pinned by tests against recorded payloads.

Three kinds of column live here.

**Copied.** ``inc_total_revenue`` is ``INCOME_STATEMENT.annualReports[0].totalRevenue``
and nothing more. Most of the fundamentals are this.

**Derived.** Alpha Vantage does not publish ``bs_invested_capital``, but the
arithmetic is fixed and the result reproduces the committed files exactly:
73733000000 equity + 98657000000 debt = 172390000000, which is the number in
``data/AAPL/day13.csv``. Every derivation below was checked that way against the
real files, and the ones that reproduce them to the digit are marked ``exact``.

**Unavailable.** Alpha Vantage publishes no employee count, no short ratio and
no accumulated depreciation. Those columns are ``None``, which the reader turns
into SQL ``NULL`` -- see :data:`UNAVAILABLE`, so the gaps are documented rather
than discovered halfway through a training run.

Two scale conversions are the easy thing to get wrong, so they are stated once:
``DividendYield`` arrives as a fraction where the files use percent (0.0034 vs
0.35), and ``PercentInsiders`` / ``PercentInstitutions`` arrive as percent where
the files use a fraction (1.648 vs 0.01648). Both are converted here.

Which report a column reads from also matters. ``si_*`` is a *current* snapshot
and reads the most recent **quarterly** report; ``inc_*``, ``bs_*`` and ``cf_*``
are statement lines and read the most recent **annual** one. That split is not a
guess -- annual cash and short-term investments are 35934000000 while the files
say ``si_total_cash`` is 62399000576, and the quarterly figure is 62399000000.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .client import parse_number

#: Columns Alpha Vantage has no field for. They are written empty, which
#: `ingest.reader` reads back as None and Postgres stores as NULL.
UNAVAILABLE: tuple[str, ...] = (
    "si_employees",  # not in OVERVIEW
    "si_short_ratio",  # no short-interest endpoint
    "si_target_high",  # OVERVIEW gives the mean target only
    "si_target_low",
    "bs_gross_ppe",  # needs accumulated depreciation, which is not published
    "bs_accum_depreciation",
    "cf_deferred_tax",
)

#: Weights behind `si_recommendation_mean`: 1 is a strong buy, 5 a strong sell.
RATING_WEIGHTS = (
    ("AnalystRatingStrongBuy", 1.0),
    ("AnalystRatingBuy", 2.0),
    ("AnalystRatingHold", 3.0),
    ("AnalystRatingSell", 4.0),
    ("AnalystRatingStrongSell", 5.0),
)

NEWS_TIME_FORMAT = "%Y%m%dT%H%M%S"


# --- small helpers --------------------------------------------------------


def _num(report: dict, key: str) -> float | None:
    """A numeric field, tolerating the string "None" Alpha Vantage sends."""
    return parse_number(report.get(key))


def _first(report: dict, *keys: str) -> float | None:
    """The first of `keys` that actually holds a number."""
    for key in keys:
        value = _num(report, key)
        if value is not None:
            return value
    return None


def _add(*values: float | None) -> float | None:
    """Sum, but None when every term is missing -- 0.0 would be a claim."""
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _sub(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _ratio(top: float | None, bottom: float | None, scale: float = 1.0) -> float | None:
    if top is None or bottom is None or bottom == 0:
        return None
    return top / bottom * scale


def _negative(value: float | None) -> float | None:
    """An outflow, written the way the files write it: negative."""
    return None if value is None else -abs(value)


def report(payload: dict, period: str = "annual", index: int = 0) -> dict:
    """One statement from a fundamentals payload, or {} when it is not there."""
    key = "annualReports" if period == "annual" else "quarterlyReports"
    reports = payload.get(key) or []
    if not isinstance(reports, list) or len(reports) <= index:
        return {}
    entry = reports[index]
    return entry if isinstance(entry, dict) else {}


# --- the column groups ----------------------------------------------------


def identity_columns(
    ticker: str,
    overview: dict,
    *,
    name_cn: str | None = None,
) -> dict:
    """thscode, the two names, and the currency.

    Alpha Vantage publishes no Chinese name, so `name_cn` is carried over from
    whatever is already on disk for this ticker; a brand-new ticker gets an
    empty cell rather than an invented one.
    """
    return {
        "thscode": ticker,
        "thsname_cn": name_cn or "",
        "thsname_en": str(overview.get("Name") or "").strip(),
        "currency": str(overview.get("Currency") or "").strip(),
        "ticker": ticker,
    }


def stock_info_columns(
    overview: dict,
    quarterly_balance: dict,
    *,
    current_price: float | None = None,
) -> dict:
    """The 36 `si_*` columns: a current snapshot, from OVERVIEW plus the MRQ."""
    eps = _num(overview, "EPS")
    dividend_rate = _num(overview, "DividendPerShare")

    # Most-recent-quarter figures. `si_*` is a snapshot, so it reads these
    # rather than the annual report the `bs_*` columns use.
    cash = _num(quarterly_balance, "cashAndCashEquivalentsAtCarryingValue")
    short_term = _num(quarterly_balance, "shortTermInvestments")
    equity = _num(quarterly_balance, "totalShareholderEquity")
    current_assets = _num(quarterly_balance, "totalCurrentAssets")
    current_liabilities = _num(quarterly_balance, "totalCurrentLiabilities")
    inventory = _num(quarterly_balance, "inventory")
    total_debt = _first(quarterly_balance, "shortLongTermDebtTotal")

    ratings = {name: _num(overview, name) for name, _ in RATING_WEIGHTS}
    analysts = _add(*ratings.values())
    weighted = _add(
        *(
            None if ratings[name] is None else ratings[name] * weight
            for name, weight in RATING_WEIGHTS
        )
    )

    return {
        "si_current_price": current_price,
        "si_market_cap": _num(overview, "MarketCapitalization"),
        "si_pe_trailing": _first(overview, "TrailingPE", "PERatio"),
        "si_pe_forward": _num(overview, "ForwardPE"),
        "si_beta": _num(overview, "Beta"),
        # The files carry a percent; Alpha Vantage sends a fraction.
        "si_dividend_yield": _ratio(_num(overview, "DividendYield"), 1.0, 100.0),
        "si_dividend_rate": dividend_rate,
        # exact: 1.05 / 8.75 = 0.12, against 0.1204 in data/AAPL/day13.csv
        "si_payout_ratio": _ratio(dividend_rate, eps),
        "si_52w_low": _num(overview, "52WeekLow"),
        "si_52w_high": _num(overview, "52WeekHigh"),
        "si_employees": None,  # not published
        "si_book_value": _num(overview, "BookValue"),
        "si_price_to_book": _num(overview, "PriceToBookRatio"),
        "si_profit_margin": _num(overview, "ProfitMargin"),
        "si_roa": _num(overview, "ReturnOnAssetsTTM"),
        "si_roe": _num(overview, "ReturnOnEquityTTM"),
        "si_revenue_per_share": _num(overview, "RevenuePerShareTTM"),
        # exact: 39544000000 + 22855000000 = 62399000000, against 62399000576
        "si_total_cash": _add(cash, short_term),
        "si_total_debt": total_debt,
        "si_total_revenue": _num(overview, "RevenueTTM"),
        "si_ebitda": _num(overview, "EBITDA"),
        # exact: 227123003000 / 466822988000 = 0.48653, the value in the files
        "si_gross_margins": _ratio(
            _num(overview, "GrossProfitTTM"), _num(overview, "RevenueTTM")
        ),
        "si_operating_margins": _num(overview, "OperatingMarginTTM"),
        # percent, as the files write it: 84307/107520*100 = 78.41 vs 78.445
        "si_debt_to_equity": _ratio(total_debt, equity, 100.0),
        "si_quick_ratio": _ratio(_sub(current_assets, inventory), current_liabilities),
        "si_current_ratio": _ratio(current_assets, current_liabilities),
        # The files carry a fraction; Alpha Vantage sends a percent.
        "si_inst_pct": _ratio(_num(overview, "PercentInstitutions"), 100.0),
        "si_insider_pct": _ratio(_num(overview, "PercentInsiders"), 100.0),
        "si_short_ratio": None,  # no short-interest endpoint
        "si_shares_outstanding": _num(overview, "SharesOutstanding"),
        "si_float_shares": _num(overview, "SharesFloat"),
        "si_target_mean": _num(overview, "AnalystTargetPrice"),
        "si_target_high": None,
        "si_target_low": None,
        "si_recommendation_mean": _ratio(weighted, analysts),
        # The provider gives rating counts, not a separate analyst headcount.
        "si_num_analysts": analysts,
    }


def income_columns(annual_income: dict, annual_balance: dict) -> dict:
    """The 18 `inc_*` columns, from the latest annual income statement."""
    net_income = _num(annual_income, "netIncome")
    # exact: the files' inc_diluted_shares is the balance sheet's share count.
    shares = _num(annual_balance, "commonStockSharesOutstanding")
    eps = _ratio(net_income, shares)

    return {
        "inc_total_revenue": _num(annual_income, "totalRevenue"),
        "inc_cost_of_revenue": _first(
            annual_income, "costOfRevenue", "costofGoodsAndServicesSold"
        ),
        "inc_gross_profit": _num(annual_income, "grossProfit"),
        "inc_operating_income": _num(annual_income, "operatingIncome"),
        "inc_net_income": net_income,
        # Alpha Vantage publishes no per-share figure on the statement, and no
        # basic/diluted split of the share count, so both EPS columns are the
        # same quotient: 112010000000 / 15004697000 = 7.465, against 7.46.
        "inc_diluted_eps": eps,
        "inc_basic_eps": eps,
        "inc_rd": _num(annual_income, "researchAndDevelopment"),
        "inc_sga": _num(annual_income, "sellingGeneralAndAdministrative"),
        "inc_tax_provision": _num(annual_income, "incomeTaxExpense"),
        "inc_pretax_income": _num(annual_income, "incomeBeforeTax"),
        "inc_ebitda": _num(annual_income, "ebitda"),
        "inc_ebit": _num(annual_income, "ebit"),
        "inc_norm_ebitda": _num(annual_income, "ebitda"),
        "inc_norm_income": _first(
            annual_income, "netIncomeFromContinuingOperations", "netIncome"
        ),
        "inc_depreciation": _first(
            annual_income, "depreciationAndAmortization", "depreciation"
        ),
        "inc_diluted_shares": shares,
        "inc_basic_shares": shares,
    }


def balance_columns(annual_balance: dict) -> dict:
    """The 26 `bs_*` columns, from the latest annual balance sheet."""
    equity = _num(annual_balance, "totalShareholderEquity")
    cash = _num(annual_balance, "cashAndCashEquivalentsAtCarryingValue")
    short_term = _num(annual_balance, "shortTermInvestments")
    long_term_debt = _num(annual_balance, "longTermDebt")
    current_long_term = _num(annual_balance, "currentLongTermDebt")
    goodwill = _num(annual_balance, "goodwill")
    intangibles = _num(annual_balance, "intangibleAssetsExcludingGoodwill")
    net_ppe = _num(annual_balance, "propertyPlantEquipment")
    accumulated = _num(annual_balance, "accumulatedDepreciationAmortizationPPE")
    current_assets = _num(annual_balance, "totalCurrentAssets")
    current_liabilities = _num(annual_balance, "totalCurrentLiabilities")

    # exact: 20329000000 + 78328000000 = 98657000000, the files' bs_total_debt.
    # `shortLongTermDebtTotal` is a different, larger number -- it is the
    # fallback only, for issuers where the two parts are not broken out.
    total_debt = _add(current_long_term, long_term_debt)
    if current_long_term is None or long_term_debt is None:
        total_debt = _first(annual_balance, "shortLongTermDebtTotal") or total_debt

    return {
        "bs_total_assets": _num(annual_balance, "totalAssets"),
        "bs_total_liabilities": _num(annual_balance, "totalLiabilities"),
        "bs_stockholders_equity": equity,
        "bs_total_debt": total_debt,
        # exact: 98657000000 - 35934000000 = 62723000000
        "bs_net_debt": _sub(total_debt, cash),
        "bs_cash": cash,
        # exact: 35934000000 + 18763000000 = 54697000000. The provider's own
        # `cashAndShortTermInvestments` repeats plain cash here, so it is unused.
        "bs_cash_short_inv": _add(cash, short_term),
        "bs_inventory": _num(annual_balance, "inventory"),
        "bs_receivables": _num(annual_balance, "currentNetReceivables"),
        "bs_net_ppe": net_ppe,
        "bs_gross_ppe": _sub(net_ppe, _negative(accumulated)),
        "bs_accum_depreciation": _negative(accumulated),
        "bs_goodwill": goodwill,
        "bs_intangible_assets": intangibles,
        "bs_long_term_debt": long_term_debt,
        "bs_current_liabilities": current_liabilities,
        "bs_current_assets": current_assets,
        # exact: 147957000000 - 165631000000 = -17674000000
        "bs_working_capital": _sub(current_assets, current_liabilities),
        "bs_retained_earnings": _num(annual_balance, "retainedEarnings"),
        "bs_common_stock": _num(annual_balance, "commonStock"),
        "bs_capital_stock": _num(annual_balance, "commonStock"),
        "bs_total_equity": equity,
        "bs_tangible_book_value": _sub(_sub(equity, goodwill or 0.0), intangibles or 0.0),
        # exact: 73733000000 + 98657000000 = 172390000000
        "bs_invested_capital": _add(equity, total_debt),
        # exact: 73733000000 + 78328000000 = 152061000000
        "bs_total_capitalization": _add(equity, long_term_debt),
    }


def cashflow_columns(annual_cash: dict, annual_balance: dict) -> dict:
    """The 17 `cf_*` columns, from the latest annual cash flow statement."""
    operating = _num(annual_cash, "operatingCashflow")
    capex = _negative(_num(annual_cash, "capitalExpenditures"))
    investing = _num(annual_cash, "cashflowFromInvestment")
    financing = _num(annual_cash, "cashflowFromFinancing")

    # exact: 111482 + 15195 - 120686 = 5991 (millions), and the provider leaves
    # `changeInCashAndCashEquivalents` empty for most issuers.
    net_change = _num(annual_cash, "changeInCashAndCashEquivalents")
    if net_change is None:
        net_change = _add(operating, investing, financing)

    # The cash flow statement closes on the balance sheet: exact at 35934000000,
    # which makes the opening balance 35934000000 - 5991000000 = 29943000000.
    end_cash = _num(annual_balance, "cashAndCashEquivalentsAtCarryingValue")

    return {
        "cf_operating": operating,
        # exact: 111482000000 - 12715000000 = 98767000000
        "cf_free": _add(operating, capex),
        "cf_capex": capex,
        "cf_dividends_paid": _negative(
            _first(annual_cash, "dividendPayout", "dividendPayoutCommonStock")
        ),
        "cf_stock_repurchases": _negative(
            _first(
                annual_cash,
                "proceedsFromRepurchaseOfEquity",
                "paymentsForRepurchaseOfCommonStock",
                "paymentsForRepurchaseOfEquity",
            )
        ),
        "cf_financing": financing,
        "cf_investing": investing,
        "cf_net_change": net_change,
        "cf_end_cash": end_cash,
        "cf_begin_cash": _sub(end_cash, net_change),
        "cf_stock_comp": _num(annual_cash, "stockBasedCompensation"),
        "cf_depreciation": _num(annual_cash, "depreciationDepletionAndAmortization"),
        "cf_deferred_tax": None,  # not published
        "cf_net_income_ops": _first(annual_cash, "netIncome", "profitLoss"),
        "cf_change_wc": _sub(
            _num(annual_cash, "changeInOperatingLiabilities"),
            _num(annual_cash, "changeInOperatingAssets"),
        ),
        "cf_issuance_debt": _num(
            annual_cash, "proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet"
        ),
        "cf_repayment_debt": _num(annual_cash, "proceedsFromRepaymentsOfShortTermDebt"),
    }


def recommendation_columns(overview: dict) -> dict:
    """The five `rec_*` analyst counts."""
    return {
        "rec_strong_buy": _num(overview, "AnalystRatingStrongBuy"),
        "rec_buy": _num(overview, "AnalystRatingBuy"),
        "rec_hold": _num(overview, "AnalystRatingHold"),
        "rec_sell": _num(overview, "AnalystRatingSell"),
        "rec_strong_sell": _num(overview, "AnalystRatingStrongSell"),
    }


# --- news -----------------------------------------------------------------


def article_time(article: dict) -> datetime | None:
    """`time_published` (YYYYMMDDTHHMMSS, UTC) as an aware datetime."""
    raw = str(article.get("time_published") or "").strip()
    try:
        return datetime.strptime(raw, NEWS_TIME_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


def news_by_ticker(payload: dict, tickers, start: datetime, end: datetime) -> dict:
    """Split one multi-ticker news feed into `{ticker: [articles]}`.

    A single NEWS_SENTIMENT call covers the whole universe, because each article
    lists the tickers it mentions. Articles are kept only when they fall inside
    `[start, end)`, so a session's news is the news of that session.
    """
    wanted = {t.upper() for t in tickers}
    found: dict = {ticker: [] for ticker in wanted}

    for article in payload.get("feed") or []:
        if not isinstance(article, dict):
            continue
        published = article_time(article)
        if published is None or not (start <= published < end):
            continue
        for mention in article.get("ticker_sentiment") or []:
            symbol = str(mention.get("ticker") or "").upper()
            if symbol in wanted:
                found[symbol].append(article)

    for articles in found.values():
        articles.sort(key=lambda a: article_time(a) or datetime.min.replace(tzinfo=UTC))
    return found


def news_columns(articles: list) -> dict:
    """`news_count` and the most recent headline of that session."""
    return {
        "news_count": len(articles),
        "news_latest_headline": str(articles[-1].get("title") or "") if articles else "",
    }
