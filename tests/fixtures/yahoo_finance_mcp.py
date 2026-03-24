"""Yahoo Finance MCP server for agentji.

A FastMCP server that exposes financial data tools powered by yfinance.
Provides income statements, balance sheets, cash flow statements, and
key metrics for any publicly traded company.

Run standalone:
    python yahoo_finance_mcp.py

Used by agentji via MCP bridge (stdio transport).
"""

from __future__ import annotations

import json
import random
from typing import Any

import yfinance as yf
from fastmcp import FastMCP

mcp = FastMCP(
    name="Yahoo Finance",
    instructions=(
        "Provides financial statement data via Yahoo Finance. "
        "Use get_financial_statements to fetch income statement, balance sheet, "
        "and cash flow data for any ticker. "
        "Use get_key_metrics for pre-computed financial ratios and market data."
    ),
)

# ── FTSE 100 tickers (London Stock Exchange) ──────────────────────────────────
# Using Yahoo Finance ticker format (suffix .L for LSE)
FTSE100_TICKERS: list[str] = [
    "AZN.L",   # AstraZeneca
    "SHEL.L",  # Shell
    "HSBA.L",  # HSBC
    "ULVR.L",  # Unilever
    "BP.L",    # BP
    "RIO.L",   # Rio Tinto
    "GSK.L",   # GSK
    "DGE.L",   # Diageo
    "REL.L",   # Relx
    "BHP.L",   # BHP
    "BATS.L",  # British American Tobacco
    "LSEG.L",  # London Stock Exchange Group
    "NG.L",    # National Grid
    "VOD.L",   # Vodafone
    "GLEN.L",  # Glencore
    "AAL.L",   # Anglo American
    "BARC.L",  # Barclays
    "LLOY.L",  # Lloyds Banking
    "PRU.L",   # Prudential
    "BA.L",    # BAE Systems
    "CPG.L",   # Compass Group
    "WPP.L",   # WPP
    "ABF.L",   # Associated British Foods
    "SBRY.L",  # Sainsbury
    "TSCO.L",  # Tesco
    "IMB.L",   # Imperial Brands
    "SGE.L",   # Sage Group
    "SSE.L",   # SSE
    "NWG.L",   # NatWest Group
    "STAN.L",  # Standard Chartered
    "MNG.L",   # M&G
    "III.L",   # 3i Group
    "EXPN.L",  # Experian
    "ITRK.L",  # Intertek
    "ADM.L",   # Admiral Group
    "AVV.L",   # Aveva (now part of Schneider)
    "AV.L",    # Aviva
    "MNDI.L",  # Mondi
    "PSON.L",  # Pearson
    "RKT.L",   # Reckitt Benckiser
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _df_to_dict(df: Any) -> dict[str, Any]:
    """Convert a pandas DataFrame to a JSON-serialisable dict."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return {}
    try:
        return json.loads(df.to_json())
    except Exception:
        return {}


def _safe_get(info: dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely retrieve a value from the info dict, returning default on None."""
    val = info.get(key)
    return val if val is not None else default


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def list_ftse100_tickers() -> str:
    """List all FTSE 100 tickers available for analysis.

    Returns a JSON list of ticker symbols (Yahoo Finance format, .L suffix for LSE).
    """
    return json.dumps(FTSE100_TICKERS, indent=2)


@mcp.tool()
def get_financial_statements(
    ticker: str = "",
    random_ftse100: bool = False,
    periods: int = 3,
) -> str:
    """Fetch the latest financial statements for a company from Yahoo Finance.

    Retrieves annual income statement, balance sheet, and cash flow statement.
    Returns structured JSON suitable for financial analysis.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. 'AZN.L', 'AAPL', 'TSLA').
                Leave empty and set random_ftse100=True to pick a random FTSE 100 company.
        random_ftse100: If True, randomly selects a FTSE 100 company. Ignored if ticker is set.
        periods: Number of annual periods to return (1-5). Defaults to 3.
    """
    if not ticker:
        if random_ftse100:
            ticker = random.choice(FTSE100_TICKERS)
        else:
            return json.dumps({"error": "Provide a ticker symbol or set random_ftse100=True."})

    ticker = ticker.upper().strip()
    periods = max(1, min(5, periods))

    try:
        company = yf.Ticker(ticker)
        info: dict[str, Any] = company.info or {}

        # Financial statements — truncate to requested periods
        income_stmt = _df_to_dict(company.income_stmt)
        balance_sheet = _df_to_dict(company.balance_sheet)
        cash_flow = _df_to_dict(company.cashflow)

        def _trim(stmt: dict[str, Any]) -> dict[str, Any]:
            """Keep only the most recent N period columns."""
            if not stmt:
                return stmt
            # Columns are period timestamps; sort descending and trim
            cols = sorted(stmt.get(next(iter(stmt)), {}).keys(), reverse=True)[:periods]
            return {row: {c: v for c, v in periods_dict.items() if c in cols}
                    for row, periods_dict in stmt.items()}

        result = {
            "ticker": ticker,
            "company_name": _safe_get(info, "longName", ticker),
            "sector": _safe_get(info, "sector", "Unknown"),
            "industry": _safe_get(info, "industry", "Unknown"),
            "country": _safe_get(info, "country", "Unknown"),
            "currency": _safe_get(info, "currency", "Unknown"),
            "exchange": _safe_get(info, "exchange", "Unknown"),
            "periods_returned": periods,
            "income_statement": _trim(income_stmt),
            "balance_sheet": _trim(balance_sheet),
            "cash_flow_statement": _trim(cash_flow),
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as exc:
        return json.dumps({
            "error": f"Failed to fetch data for '{ticker}': {exc}",
            "ticker": ticker,
        })


@mcp.tool()
def get_key_metrics(
    ticker: str = "",
    random_ftse100: bool = False,
) -> str:
    """Fetch pre-computed key financial metrics and market data for a company.

    Returns valuation ratios, profitability metrics, leverage, dividends, and
    analyst recommendations in a compact JSON summary.

    Args:
        ticker: Yahoo Finance ticker symbol. Leave empty with random_ftse100=True
                to pick a random FTSE 100 company.
        random_ftse100: If True, randomly selects a FTSE 100 company.
    """
    if not ticker:
        if random_ftse100:
            ticker = random.choice(FTSE100_TICKERS)
        else:
            return json.dumps({"error": "Provide a ticker symbol or set random_ftse100=True."})

    ticker = ticker.upper().strip()

    try:
        company = yf.Ticker(ticker)
        info: dict[str, Any] = company.info or {}

        metrics = {
            "ticker": ticker,
            "company_name": _safe_get(info, "longName", ticker),
            "sector": _safe_get(info, "sector"),
            "industry": _safe_get(info, "industry"),
            "currency": _safe_get(info, "currency"),
            # Market
            "market_cap": _safe_get(info, "marketCap"),
            "enterprise_value": _safe_get(info, "enterpriseValue"),
            "current_price": _safe_get(info, "currentPrice"),
            "52_week_high": _safe_get(info, "fiftyTwoWeekHigh"),
            "52_week_low": _safe_get(info, "fiftyTwoWeekLow"),
            # Valuation
            "trailing_pe": _safe_get(info, "trailingPE"),
            "forward_pe": _safe_get(info, "forwardPE"),
            "price_to_book": _safe_get(info, "priceToBook"),
            "price_to_sales": _safe_get(info, "priceToSalesTrailing12Months"),
            "ev_to_ebitda": _safe_get(info, "enterpriseToEbitda"),
            "ev_to_revenue": _safe_get(info, "enterpriseToRevenue"),
            # Profitability
            "revenue_ttm": _safe_get(info, "totalRevenue"),
            "gross_margin": _safe_get(info, "grossMargins"),
            "operating_margin": _safe_get(info, "operatingMargins"),
            "net_margin": _safe_get(info, "profitMargins"),
            "return_on_equity": _safe_get(info, "returnOnEquity"),
            "return_on_assets": _safe_get(info, "returnOnAssets"),
            "ebitda": _safe_get(info, "ebitda"),
            # Balance sheet health
            "total_debt": _safe_get(info, "totalDebt"),
            "total_cash": _safe_get(info, "totalCash"),
            "debt_to_equity": _safe_get(info, "debtToEquity"),
            "current_ratio": _safe_get(info, "currentRatio"),
            "quick_ratio": _safe_get(info, "quickRatio"),
            # Cash flow
            "free_cash_flow": _safe_get(info, "freeCashflow"),
            "operating_cash_flow": _safe_get(info, "operatingCashflow"),
            # Dividend
            "dividend_yield": _safe_get(info, "dividendYield"),
            "dividend_rate": _safe_get(info, "dividendRate"),
            "payout_ratio": _safe_get(info, "payoutRatio"),
            # Growth
            "earnings_growth": _safe_get(info, "earningsGrowth"),
            "revenue_growth": _safe_get(info, "revenueGrowth"),
            # Analyst
            "analyst_recommendation": _safe_get(info, "recommendationKey"),
            "target_mean_price": _safe_get(info, "targetMeanPrice"),
            "number_of_analyst_opinions": _safe_get(info, "numberOfAnalystOpinions"),
        }

        return json.dumps(metrics, indent=2, default=str)

    except Exception as exc:
        return json.dumps({
            "error": f"Failed to fetch metrics for '{ticker}': {exc}",
            "ticker": ticker,
        })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
