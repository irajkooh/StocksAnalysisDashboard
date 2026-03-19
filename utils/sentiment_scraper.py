"""
utils/sentiment_scraper.py — Multi-source Sentiment Aggregator
Sources: NewsAPI, Reddit (PRAW), SEC EDGAR, X.com (public search fallback)
"""

import re
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from config import (
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


# ─── NewsAPI ──────────────────────────────────────────────────────────────────

def fetch_news_sentiment(ticker: str) -> Dict:
    headlines, score, articles = [], 0.0, []
    if not NEWS_API_KEY:
        return {"score": 0.0, "label": "No API Key", "headlines": [], "count": 0}
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

def fetch_sec_sentiment(ticker: str) -> Dict:
    filings, score = [], 0.0
    try:
        end   = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
        url   = f"https://efts.sec.gov/LATEST/search-index?q={ticker}&dateRange=custom&startdt={start}&enddt={end}&forms=8-K,10-K,10-Q"
        headers = {"User-Agent": "StocksDashboard/1.0 contact@example.com"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            for hit in r.json().get("hits", {}).get("hits", [])[:8]:
                src  = hit.get("_source", {})
                text = f"{src.get('period_of_report','')} {src.get('file_date','')} {src.get('form_type','')}"
                s    = score_text(text)
                score += s
                filings.append({
                    "form":    src.get("form_type",""),
                    "date":    src.get("file_date",""),
                    "company": src.get("entity_name",""),
                    "score":   round(s, 2),
                })
        if filings:
            score /= len(filings)
    except Exception as e:
        logger.warning(f"SEC EDGAR error: {e}")

    return {"score": round(score, 3), "label": sentiment_label(score),
            "filings": filings, "count": len(filings)}


# ─── X.com / Twitter (public search scrape, no API key) ──────────────────────

def fetch_twitter_sentiment(ticker: str) -> Dict:
    """
    Uses nitter public instance or web search to approximate Twitter sentiment.
    Falls back gracefully if unavailable.
    """
    posts, score = [], 0.0
    try:
        # Attempt public nitter instance (may be blocked)
        nitter_instances = [
            "https://nitter.net",
            "https://nitter.privacydev.net",
            "https://nitter.poast.org",
        ]
        cashtag = f"${ticker.upper()}"
        for base in nitter_instances:
            try:
                url  = f"{base}/search?q={cashtag}&f=tweets"
                r    = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200:
                    # Quick regex extraction of tweet text
                    tweets = re.findall(r'class="tweet-content[^"]*"[^>]*>(.*?)</div>', r.text, re.S)
                    for t in tweets[:15]:
                        clean = re.sub(r"<[^>]+>", "", t).strip()
                        if clean:
                            s = score_text(clean)
                            score += s
                            posts.append({"text": clean[:120], "score": round(s, 2)})
                    break
            except Exception:
                continue
        if posts:
            score /= len(posts)
    except Exception as e:
        logger.warning(f"Twitter/X scrape error: {e}")

    if not posts:
        return {"score": 0.0, "label": "Unavailable (no API)", "posts": [], "count": 0}

    return {"score": round(score, 3), "label": sentiment_label(score),
            "posts": posts[:8], "count": len(posts)}


# ─── Aggregated Sentiment ─────────────────────────────────────────────────────

def get_aggregate_sentiment(ticker: str) -> Dict:
    """Fetch all sources and return weighted aggregate sentiment."""
    twitter = fetch_twitter_sentiment(ticker)
    news    = fetch_news_sentiment(ticker)
    reddit  = fetch_reddit_sentiment(ticker)
    sec     = fetch_sec_sentiment(ticker)

    w = SENTIMENT_WEIGHTS
    agg = (
        twitter["score"] * w["twitter"] +
        news["score"]    * w["news"]    +
        reddit["score"]  * w["reddit"]  +
        sec["score"]     * w["sec"]
    )

    return {
        "aggregate_score":  round(agg, 3),
        "aggregate_label":  sentiment_label(agg),
        "twitter":          twitter,
        "news":             news,
        "reddit":           reddit,
        "sec":              sec,
        "weights":          w,
    }
