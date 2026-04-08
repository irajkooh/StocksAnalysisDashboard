"""
capitol_trades_mcp.py — MCP Server for Capitol Trades
Exposes tools for querying US politician stock trade disclosures
from https://www.capitoltrades.com/

Run directly:
    python capitol_trades_mcp.py

Or via the MCP CLI:
    mcp run capitol_trades_mcp.py
"""

from mcp.server.fastmcp import FastMCP
from utils.capitol_trades_scraper import (
    get_recent_trades as _get_recent_trades,
    get_trades_by_politician as _get_trades_by_politician,
    get_trades_by_ticker as _get_trades_by_ticker,
    get_top_issuers as _get_top_issuers,
    get_politicians as _get_politicians,
    get_trade_summary_stats as _get_trade_summary_stats,
    get_latest_insights as _get_latest_insights,
)


mcp = FastMCP("Capitol Trades")


# ── MCP Tools (thin wrappers around utils/capitol_trades_scraper) ─────────────

@mcp.tool()
def get_recent_trades(page: int = 1, page_size: int = 20, trade_type: str = "") -> dict:
    """Get the most recent US politician stock trade disclosures from Capitol Trades."""
    return _get_recent_trades(page=page, page_size=page_size, trade_type=trade_type)


@mcp.tool()
def get_trades_by_politician(politician_id: str, page: int = 1, page_size: int = 20) -> dict:
    """Get stock trade disclosures for a specific US politician by their Capitol Trades ID."""
    return _get_trades_by_politician(politician_id=politician_id, page=page, page_size=page_size)


@mcp.tool()
def get_trades_by_ticker(ticker: str, page: int = 1, page_size: int = 20) -> dict:
    """Find all politician trades for a specific stock ticker."""
    return _get_trades_by_ticker(ticker=ticker, page=page, page_size=page_size)


@mcp.tool()
def get_top_issuers(page: int = 1, page_size: int = 20) -> dict:
    """Get the most actively traded stocks (issuers) by US politicians."""
    return _get_top_issuers(page=page, page_size=page_size)


@mcp.tool()
def get_politicians(page: int = 1, page_size: int = 20) -> dict:
    """Get a list of US politicians tracked on Capitol Trades."""
    return _get_politicians(page=page, page_size=page_size)


@mcp.tool()
def get_trade_summary_stats() -> dict:
    """Get overall summary statistics from Capitol Trades homepage."""
    return _get_trade_summary_stats()


@mcp.tool()
def get_latest_insights(page: int = 1) -> dict:
    """Get the latest news articles and insights from Capitol Trades."""
    return _get_latest_insights(page=page)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
