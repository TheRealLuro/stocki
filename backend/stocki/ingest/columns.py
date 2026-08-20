"""The CSV -> Postgres column contract.

`bars_raw` mirrors ``data/<TICKER>/day<N>.csv`` column for column, in order, so
``SELECT * FROM bars_raw WHERE ticker = 'AAPL' AND day = 13 ORDER BY timestamp``
returns exactly what ``data/AAPL/day13.csv`` holds.

Two names are normalised, because Postgres folds unquoted identifiers to
lower case and ``Stock Splits`` contains a space -- keeping them verbatim would
mean double-quoting them in every query forever. Three provenance columns are
appended: ``day`` because the day number lives only in the *filename* and would
otherwise be lost, plus the source path and load time for auditing.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA_SQL_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"

#: Column names exactly as they appear in the CSV header, in file order.
CSV_COLUMNS: tuple[str, ...] = (
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'Dividends', 'Stock Splits',
    'thscode', 'thsname_cn', 'thsname_en', 'currency', 'hour', 'minute', 'ticker',
    'si_current_price', 'si_market_cap', 'si_pe_trailing', 'si_pe_forward', 'si_beta',
    'si_dividend_yield', 'si_dividend_rate', 'si_payout_ratio', 'si_52w_low', 'si_52w_high',
    'si_employees', 'si_book_value', 'si_price_to_book', 'si_profit_margin', 'si_roa',
    'si_roe', 'si_revenue_per_share', 'si_total_cash', 'si_total_debt', 'si_total_revenue',
    'si_ebitda', 'si_gross_margins', 'si_operating_margins', 'si_debt_to_equity',
    'si_quick_ratio', 'si_current_ratio', 'si_inst_pct', 'si_insider_pct', 'si_short_ratio',
    'si_shares_outstanding', 'si_float_shares', 'si_target_mean', 'si_target_high',
    'si_target_low', 'si_recommendation_mean', 'si_num_analysts', 'inc_total_revenue',
    'inc_cost_of_revenue', 'inc_gross_profit', 'inc_operating_income', 'inc_net_income',
    'inc_diluted_eps', 'inc_basic_eps', 'inc_rd', 'inc_sga', 'inc_tax_provision',
    'inc_pretax_income', 'inc_ebitda', 'inc_ebit', 'inc_norm_ebitda', 'inc_norm_income',
    'inc_depreciation', 'inc_diluted_shares', 'inc_basic_shares', 'bs_total_assets',
    'bs_total_liabilities', 'bs_stockholders_equity', 'bs_total_debt', 'bs_net_debt',
    'bs_cash', 'bs_cash_short_inv', 'bs_inventory', 'bs_receivables', 'bs_net_ppe',
    'bs_gross_ppe', 'bs_accum_depreciation', 'bs_goodwill', 'bs_intangible_assets',
    'bs_long_term_debt', 'bs_current_liabilities', 'bs_current_assets',
    'bs_working_capital', 'bs_retained_earnings', 'bs_common_stock', 'bs_capital_stock',
    'bs_total_equity', 'bs_tangible_book_value', 'bs_invested_capital',
    'bs_total_capitalization', 'cf_operating', 'cf_free', 'cf_capex', 'cf_dividends_paid',
    'cf_stock_repurchases', 'cf_financing', 'cf_investing', 'cf_net_change', 'cf_end_cash',
    'cf_begin_cash', 'cf_stock_comp', 'cf_depreciation', 'cf_deferred_tax',
    'cf_net_income_ops', 'cf_change_wc', 'cf_issuance_debt', 'cf_repayment_debt',
    'rec_strong_buy', 'rec_buy', 'rec_hold', 'rec_sell', 'rec_strong_sell', 'news_count',
    'news_latest_headline',
)

#: The only two columns whose names change on the way into Postgres.
RENAMED: dict[str, str] = {
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}

#: Appended after the CSV columns; not present in the files.
PROVENANCE_COLUMNS: tuple[str, ...] = ("day", "source_file", "ingested_at")

TIMESTAMPTZ_COLUMNS = frozenset({"timestamp", "ingested_at"})
BIGINT_COLUMNS = frozenset({"volume"})
INTEGER_COLUMNS = frozenset({"news_count"})
SMALLINT_COLUMNS = frozenset({"hour", "minute", "day"})
TEXT_COLUMNS = frozenset(
    {
        "thscode",
        "thsname_cn",
        "thsname_en",
        "currency",
        "ticker",
        "news_latest_headline",
        "source_file",
    }
)

#: Identity + fundamentals: constant within a (ticker, day), so `v_fundamentals`
#: collapses the 78 duplicate rows down to one.
STATIC_PREFIXES = ("si_", "inc_", "bs_", "cf_", "rec_")
IDENTITY_COLUMNS = ("thscode", "thsname_cn", "thsname_en", "currency")


def sql_name(csv_column: str) -> str:
    """Postgres column name for a CSV header field."""
    return RENAMED.get(csv_column, csv_column)


def pg_type(column: str) -> str:
    """Postgres type for a table column (SQL name, not CSV name)."""
    if column in TIMESTAMPTZ_COLUMNS:
        return "timestamptz"
    if column in BIGINT_COLUMNS:
        return "bigint"
    if column in INTEGER_COLUMNS:
        return "integer"
    if column in SMALLINT_COLUMNS:
        return "smallint"
    if column in TEXT_COLUMNS:
        return "text"
    return "double precision"


def table_columns() -> tuple[str, ...]:
    """Every column of `bars_raw`: the CSV columns, then provenance."""
    return tuple(sql_name(c) for c in CSV_COLUMNS) + PROVENANCE_COLUMNS


def static_columns() -> tuple[str, ...]:
    """Columns that never vary within a (ticker, day)."""
    return IDENTITY_COLUMNS + tuple(
        sql_name(c) for c in CSV_COLUMNS if c.startswith(STATIC_PREFIXES)
    )
