"""
utils/sentiment_scraper.py — Multi-source Sentiment Aggregator
Sources: NewsAPI (or yfinance fallback), Reddit (PRAW), SEC EDGAR,
         Google News RSS (replaces dead nitter/X.com scraping)
"""

import re
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from utils.config import (
    NEWS_API_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT, SEC_EDGAR_BASE_URL, SENTIMENT_WEIGHTS
)

logger = logging.getLogger(__name__)

# ─── Simple Keyword Sentiment Scorer ──────────────────────────────────────────

POSITIVE_WORDS = {
    "beat", "surge", "soar", "rally", "gain", "profit", "record", "strong",
    "upgrade", "buy", "bullish", "growth", "outperform", "exceed", "positive",
    "rise", "jump", "boost", "recover", "rebound", "high", "best", "win",
    "milestone", "breakthrough", "expansion", "momentum", "upside",
}
NEGATIVE_WORDS = {
    "miss", "fall", "drop", "decline", "loss", "weak", "downgrade", "sell",
    "bearish", "cut", "reduce", "disappoint", "concern", "risk", "warn",
    "plunge", "crash", "fear", "trouble", "debt", "lawsuit", "investigation",
    "fraud", "recall", "layoff", "bankruptcy", "downside", "short",
}


def score_text(text: str) -> float:
    """Return sentiment score in [-1, 1] from keyword analysis."""
    if not text:
        return 0.0
    tokens = set(re.sub(r"[^a-z\s]", "", text.lower()).split())
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def sentiment_label(score: float) -> str:
    if score >= 0.25:  return "Bullish 🟢"
    if score >= 0.05:  return "Mildly Bullish 🟡"
    if score <= -0.25: return "Bearish 🔴"
    if score <= -0.05: return "Mildly Bearish 🟠"
    return "Neutral ⚪"


# ─── NewsAPI (with yfinance fallback) ─────────────────────────────────────────

def _fetch_news_via_yfinance(ticker: str) -> Dict:
    """Free fallback: use yfinance .news when no NEWS_API_KEY."""
    headlines, score = [], 0.0
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        for item in news[:20]:
            content = item.get("content", {})
            title   = content.get("title", "")
            summary = content.get("summary", content.get("description", ""))
            source  = content.get("provider", {}).get("displayName", "")
            url     = content.get("canonicalUrl", {}).get("url", "")
            s = score_text(f"{title} {summary}")
            score += s
            headlines.append({"title": title, "score": round(s, 2),
                              "url": url, "source": source})
        if headlines:
            score /= len(headlines)
    except Exception as e:
        logger.warning(f"yfinance news error: {e}")
    return {"score": round(score, 3), "label": sentiment_label(score),
            "headlines": headlines[:8], "count": len(headlines)}


def fetch_news_sentiment(ticker: str) -> Dict:
    headlines, score, articles = [], 0.0, []
    if not NEWS_API_KEY:
        return _fetch_news_via_yfinance(ticker)
    try:
        since = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        url   = (
            f"https://newsapi.org/v2/everything"
            f"?q={ticker}+stock&from={since}&sortBy=relevancy"
            f"&language=en&pageSize=20&apiKey={NEWS_API_KEY}"
        )
        resp  = requests.get(url, timeout=10)
        data  = resp.json()
        for art in data.get("articles", []):
            title = art.get("title", "")
            desc  = art.get("description", "")
            s     = score_text(f"{title} {desc}")
            score += s
            headlines.append({"title": title, "score": round(s, 2),
                               "url": art.get("url", ""), "source": art.get("source", {}).get("name","")})
        if headlines:
            score /= len(headlines)
        articles = headlines[:8]
    except Exception as e:
        logger.warning(f"NewsAPI error: {e}")
    return {"score": round(score, 3), "label": sentiment_label(score),
            "headlines": articles, "count": len(articles)}


# ─── Reddit (PRAW or requests fallback) ───────────────────────────────────────

def fetch_reddit_sentiment(ticker: str) -> Dict:
    posts, score = [], 0.0
    subreddits = ["wallstreetbets", "investing", "stocks", "stockmarket"]
    try:
        if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            import praw
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
            for sub in subreddits[:2]:
                for post in reddit.subreddit(sub).search(ticker, limit=10, time_filter="week"):
                    s = score_text(f"{post.title} {post.selftext}")
                    score += s
                    posts.append({"title": post.title, "score": round(s, 2),
                                  "subreddit": sub, "upvotes": post.score})
        else:
            # Public JSON fallback (rate-limited)
            headers = {"User-Agent": REDDIT_USER_AGENT}
            for sub in subreddits[:2]:
                url = f"https://www.reddit.com/r/{sub}/search.json?q={ticker}&sort=relevance&t=week&limit=8"
                r   = requests.get(url, headers=headers, timeout=8)
                if r.status_code == 200:
                    for child in r.json().get("data", {}).get("children", []):
                        d = child.get("data", {})
                        s = score_text(f"{d.get('title','')} {d.get('selftext','')}")
                        score += s
                        posts.append({"title": d.get("title",""), "score": round(s,2),
                                      "subreddit": sub, "upvotes": d.get("score", 0)})
                time.sleep(0.5)

        if posts:
            score /= len(posts)
    except Exception as e:
        logger.warning(f"Reddit error: {e}")

    return {"score": round(score, 3), "label": sentiment_label(score),
            "posts": posts[:8], "count": len(posts)}


# ─── SEC EDGAR ────────────────────────────────────────────────────────────────

# 8-K item codes that often carry sentiment signal
_SEC_ITEM_SENTIMENT = {
    "1.01":  0.15,   # Material Agreement → mildly positive
    "1.02": -0.20,   # Termination of Agreement → negative
    "2.01":  0.10,   # Acquisition/Disposition
    "2.02":  0.0,    # Results of Operations (need content to score)
    "2.04": -0.15,   # Material Modifications
    "2.05": -0.25,   # Delisting / Failure to Meet Standard
    "2.06": -0.30,   # Material Impairment
    "3.01": -0.10,   # Deregistration
    "5.02":  0.0,    # Officer Changes (neutral w/o context)
    "7.01":  0.0,    # Regulation FD
    "8.01":  0.0,    # Other Events
}


def fetch_sec_sentiment(ticker: str) -> Dict:
    filings, score = [], 0.0
    try:
        end   = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
        url   = (f"https://efts.sec.gov/LATEST/search-index"
                 f"?q=%22{ticker}%22&dateRange=custom&startdt={start}"
                 f"&enddt={end}&forms=8-K,10-K,10-Q")
        headers = {"User-Agent": "StocksDashboard/1.0 contact@example.com"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            for hit in r.json().get("hits", {}).get("hits", [])[:8]:
                src   = hit.get("_source", {})
                form  = src.get("form", src.get("root_forms", [""])[0] if src.get("root_forms") else "")
                names = src.get("display_names", [])
                company = names[0].split("(")[0].strip() if names else ""
                items = src.get("items", [])
                desc  = src.get("file_description", "")
                # Score from item codes (8-K specific)
                item_score = 0.0
                for it in items:
                    item_score += _SEC_ITEM_SENTIMENT.get(it, 0.0)
                # Also score the description text if available
                text_score = score_text(f"{desc} {company} {form}")
                s = (item_score + text_score) / 2 if items else text_score
                s = max(-1.0, min(1.0, s))
                score += s
                filings.append({
                    "form":    form,
                    "date":    src.get("file_date", ""),
                    "company": company,
                    "score":   round(s, 2),
                })
        if filings:
            score /= len(filings)
    except Exception as e:
        logger.warning(f"SEC EDGAR error: {e}")

    return {"score": round(score, 3), "label": sentiment_label(score),
            "filings": filings, "count": len(filings)}


# ─── Web / Social (Google News RSS — replaces dead nitter scraping) ───────────

def fetch_twitter_sentiment(ticker: str) -> Dict:
    """
    Score social/web sentiment via Google News RSS headlines.
    Replaces dead nitter instances.  Keeps the 'twitter' key for
    backward compatibility with the frontend display (shown as "X.com").
    """
    posts, score = [], 0.0
    try:
        url = (f"https://news.google.com/rss/search"
               f"?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en")
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            titles = re.findall(r"<title>(.*?)</title>", r.text)
            # Skip first two (feed title + "Google News")
            for t in titles[2:22]:
                clean = re.sub(r"<[^>]+>", "", t).strip()
                if clean:
                    s = score_text(clean)
                    score += s
                    posts.append({"text": clean[:120], "score": round(s, 2)})
        if posts:
            score /= len(posts)
    except Exception as e:
        logger.warning(f"Google News RSS error: {e}")

    if not posts:
        return {"score": 0.0, "label": "Unavailable", "posts": [], "count": 0}

    return {"score": round(score, 3), "label": sentiment_label(score),
            "posts": posts[:8], "count": len(posts),
            "source": "Google News"}


# ─── Aggregated Sentiment ─────────────────────────────────────────────────────

def get_aggregate_sentiment(ticker: str) -> Dict:
    """Fetch all sources and return weighted aggregate sentiment.
    Sources that returned no data are excluded from the weighted average
    so they don't dilute the score toward zero."""
    twitter = fetch_twitter_sentiment(ticker)
    news    = fetch_news_sentiment(ticker)
    reddit  = fetch_reddit_sentiment(ticker)
    sec     = fetch_sec_sentiment(ticker)

    sources = {
        "twitter": twitter,
        "news":    news,
        "reddit":  reddit,
        "sec":     sec,
    }
    w = dict(SENTIMENT_WEIGHTS)  # copy

    # Zero out weights for sources with no data
    for key, src in sources.items():
        if src.get("count", 0) == 0:
            w[key] = 0.0

    total_w = sum(w.values())
    if total_w > 0:
        agg = sum(sources[k]["score"] * w[k] for k in w) / total_w
    else:
        agg = 0.0

    return {
        "aggregate_score":  round(agg, 3),
        "aggregate_label":  sentiment_label(agg),
        "twitter":          twitter,
        "news":             news,
        "reddit":           reddit,
        "sec":              sec,
        "weights":          w,
    }
