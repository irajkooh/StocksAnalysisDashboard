"""
capitol_trades_mcp.py — MCP Server for Capitol Trades
Exposes tools for querying US politician stock trade disclosures
from https://www.capitoltrades.com/

Run directly:
    python capitol_trades_mcp.py

Or via the MCP CLI:
    mcp run capitol_trades_mcp.py
"""

import re
import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://www.capitoltrades.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
DEFAULT_TIMEOUT = 15

mcp = FastMCP("Capitol Trades")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None) -> BeautifulSoup:
    """Fetch a page and return a BeautifulSoup object."""
    response = requests.get(url, headers=HEADERS, params=params, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _parse_trades_table(soup: BeautifulSoup) -> list[dict]:
    """Parse the standard trades table present on /trades and filtered views."""
    rows = soup.select("table tbody tr")
    trades = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 8:
            continue
        links = row.find_all("a", href=True)
        politician_href = next((a["href"] for a in links if "/politicians/" in a["href"]), "")
        issuer_href = next((a["href"] for a in links if "/issuers/" in a["href"]), "")
        trade_href = next((a["href"] for a in links if "/trades/" in a["href"]), "")

        politician_raw = cells[0].get_text(" | ", strip=True)
        # Split "Name | Party | Chamber | State" from concatenated text
        issuer_raw = cells[1].get_text(" | ", strip=True)

        trades.append(
            {
                "politician": politician_raw,
                "politician_id": politician_href.split("/")[-1] if politician_href else "",
                "politician_url": f"{BASE_URL}{politician_href}" if politician_href else "",
                "issuer": issuer_raw,
                "issuer_id": issuer_href.split("/")[-1] if issuer_href else "",
                "issuer_url": f"{BASE_URL}{issuer_href}" if issuer_href else "",
                "reported_date": cells[2].get_text(" ", strip=True),
                "trade_date": cells[3].get_text(" ", strip=True),
                "latency": cells[4].get_text(" ", strip=True),
                "owner": cells[5].get_text(strip=True),
                "type": cells[6].get_text(strip=True),
                "size": cells[7].get_text(strip=True),
                "price": cells[8].get_text(strip=True) if len(cells) > 8 else "",
                "trade_url": f"{BASE_URL}{trade_href}" if trade_href else "",
            }
        )
    return trades


def _parse_pagination(soup: BeautifulSoup) -> dict:
    """Extract basic pagination info from the page."""
    text = soup.get_text(" ", strip=True)
    match = re.search(r"Page\s+(\d+)\s+of\s+([\d,]+)", text)
    if match:
        return {"current_page": int(match.group(1)), "total_pages": int(match.group(2).replace(",", ""))}
    return {}


# ── MCP Tools ────────────────────────────────────────────────────────────────


@mcp.tool()
def get_recent_trades(
    page: int = 1,
    page_size: int = 20,
    trade_type: str = "",
) -> dict:
    """
    Get the most recent US politician stock trade disclosures from Capitol Trades.

    Args:
        page: Page number to retrieve (default 1).
        page_size: Number of results per page — 10, 20, 50 or 100 (default 20).
        trade_type: Filter by trade type — "buy", "sell", or "" for all (default "").

    Returns:
        A dict with 'trades' (list of trade dicts) and 'pagination' info.
    """
    params: dict = {"pageSize": page_size, "page": page}
    if trade_type:
        params["txType"] = trade_type.lower()

    soup = _get(f"{BASE_URL}/trades", params=params)
    trades = _parse_trades_table(soup)
    pagination = _parse_pagination(soup)
    return {"trades": trades, "pagination": pagination, "total_results": len(trades)}


@mcp.tool()
def get_trades_by_politician(
    politician_id: str,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """
    Get stock trade disclosures for a specific US politician.

    Args:
        politician_id: Capitol Trades politician ID (e.g. "P000197" for Nancy Pelosi,
                       "H001082" for Kevin Hern). Found in politician profile URLs like
                       https://www.capitoltrades.com/politicians/P000197
        page: Page number (default 1).
        page_size: Results per page — 10, 20, 50 or 100 (default 20).

    Returns:
        A dict with 'trades' list and 'pagination' info.
    """
    params = {"politician": politician_id, "pageSize": page_size, "page": page}
    soup = _get(f"{BASE_URL}/trades", params=params)
    trades = _parse_trades_table(soup)
    pagination = _parse_pagination(soup)
    return {"politician_id": politician_id, "trades": trades, "pagination": pagination}


@mcp.tool()
def get_trades_by_ticker(
    ticker: str,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """
    Find all politician trades for a specific stock ticker by searching issuers first.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA", "TSLA", "AAPL".
        page: Page number (default 1).
        page_size: Results per page — 10, 20, 50 or 100 (default 20).

    Returns:
        A dict with 'ticker', 'issuer_id', 'trades' list, and 'pagination' info.
        Returns an error message if the ticker cannot be found.
    """
    # Step 1: search issuers to find the issuer ID for this ticker
    soup = _get(f"{BASE_URL}/issuers", params={"pageSize": 100})
    rows = soup.select("table tbody tr")
    issuer_id = ""
    ticker_upper = ticker.upper()
    for row in rows:
        text = row.get_text(" ", strip=True)
        if ticker_upper + ":US" in text or ticker_upper + ":us" in text.upper():
            link = row.find("a", href=re.compile(r"/issuers/"))
            if link:
                issuer_id = link["href"].split("/")[-1]
                break

    if not issuer_id:
        return {
            "ticker": ticker,
            "error": f"Could not find issuer for ticker '{ticker}'. "
                     "Try get_top_issuers() to browse available tickers.",
        }

    # Step 2: fetch trades filtered by issuer
    params = {"issuer": issuer_id, "pageSize": page_size, "page": page}
    soup2 = _get(f"{BASE_URL}/trades", params=params)
    trades = _parse_trades_table(soup2)
    pagination = _parse_pagination(soup2)
    return {
        "ticker": ticker_upper,
        "issuer_id": issuer_id,
        "issuer_url": f"{BASE_URL}/issuers/{issuer_id}",
        "trades": trades,
        "pagination": pagination,
    }


@mcp.tool()
def get_top_issuers(page: int = 1, page_size: int = 20) -> dict:
    """
    Get the most actively traded stocks (issuers) by US politicians on Capitol Trades.

    Args:
        page: Page number (default 1).
        page_size: Results per page — 10, 20, 50 or 100 (default 20).

    Returns:
        A dict with 'issuers' list, each containing name, ticker, issuer_id, last_trade,
        total_volume, trade_count, politician_count, sector, and performance.
    """
    params = {"pageSize": page_size, "page": page}
    soup = _get(f"{BASE_URL}/issuers", params=params)
    rows = soup.select("table tbody tr")
    issuers = []
    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue
        link = row.find("a", href=re.compile(r"/issuers/\d+"))
        issuer_id = link["href"].split("/")[-1] if link else ""
        name_cell = cells[0].get_text(" ", strip=True)
        issuers.append(
            {
                "name_ticker": name_cell,
                "issuer_id": issuer_id,
                "issuer_url": f"{BASE_URL}/issuers/{issuer_id}" if issuer_id else "",
                "last_trade_date": cells[1].get_text(strip=True) if len(cells) > 1 else "",
                "total_volume": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                "trade_count": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                "politician_count": cells[4].get_text(strip=True) if len(cells) > 4 else "",
                "sector": cells[5].get_text(strip=True) if len(cells) > 5 else "",
                "performance": cells[7].get_text(strip=True) if len(cells) > 7 else "",
            }
        )
    pagination = _parse_pagination(soup)
    return {"issuers": issuers, "pagination": pagination}


@mcp.tool()
def get_politicians(page: int = 1, page_size: int = 20) -> dict:
    """
    Get a list of US politicians tracked on Capitol Trades.

    Args:
        page: Page number (default 1).
        page_size: Results per page — 10, 20, 50 or 100 (default 20).

    Returns:
        A dict with 'politicians' list, each containing name, party, chamber, state,
        politician_id, trade_count, and profile_url.
        Note: Returns summary stats from the overview page.
    """
    params = {"pageSize": page_size, "page": page}
    soup = _get(f"{BASE_URL}/politicians", params=params)
    rows = soup.select("table tbody tr")
    politicians = []
    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue
        link = row.find("a", href=re.compile(r"/politicians/"))
        politician_id = link["href"].split("/")[-1] if link else ""
        politicians.append(
            {
                "name_info": cells[0].get_text(" | ", strip=True),
                "politician_id": politician_id,
                "profile_url": f"{BASE_URL}/politicians/{politician_id}" if politician_id else "",
                "trade_count": cells[1].get_text(strip=True) if len(cells) > 1 else "",
                "filing_count": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                "issuer_count": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                "total_volume": cells[4].get_text(strip=True) if len(cells) > 4 else "",
            }
        )
    pagination = _parse_pagination(soup)
    return {"politicians": politicians, "pagination": pagination}


@mcp.tool()
def get_trade_summary_stats() -> dict:
    """
    Get overall summary statistics from Capitol Trades homepage — total trades tracked,
    filings, volume, number of politicians and issuers.

    Returns:
        A dict with summary stats: total_trades, total_filings, total_volume,
        total_politicians, total_issuers, and recent_trades (last 5).
    """
    soup = _get(BASE_URL)

    # Extract summary numbers (shown prominently on trades page too)
    soup_trades = _get(f"{BASE_URL}/trades", params={"pageSize": 5})
    stats_text = soup_trades.get_text(" ", strip=True)

    # Parse the stat bar: "34,579 TRADES  1,733 FILINGS  $2.314B VOLUME  199 POLITICIANS  3,116 ISSUERS"
    stats = {}
    patterns = {
        "total_trades": r"([\d,]+)\s+TRADES",
        "total_filings": r"([\d,]+)\s+FILINGS",
        "total_volume": r"\$([\d.]+[BMK]?)\s+VOLUME",
        "total_politicians": r"([\d,]+)\s+POLITICIANS",
        "total_issuers": r"([\d,]+)\s+ISSUERS",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stats_text, re.IGNORECASE)
        stats[key] = match.group(1) if match else "N/A"

    recent = _parse_trades_table(soup_trades)[:5]
    stats["recent_trades"] = recent
    return stats


@mcp.tool()
def get_latest_insights(page: int = 1) -> dict:
    """
    Get the latest news articles and insights from Capitol Trades about politician trading.

    Args:
        page: Page number (default 1).

    Returns:
        A dict with 'articles' list, each containing title, url, and date.
    """
    soup = _get(f"{BASE_URL}/articles", params={"page": page})
    articles = []

    # Articles appear as anchor tags with /articles/ in their href
    seen = set()
    for a in soup.find_all("a", href=re.compile(r"/articles/[a-z0-9\-]+")):
        href = a["href"]
        if href in seen:
            continue
        seen.add(href)
        title = a.get_text(" ", strip=True)
        if title and len(title) > 10:
            articles.append(
                {
                    "title": title,
                    "url": f"{BASE_URL}{href}" if href.startswith("/") else href,
                }
            )

    return {"articles": articles[:20], "page": page}


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
