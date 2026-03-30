"""
agents/sentiment_agent.py — Sentiment Analysis Agent
Aggregates sentiment from News, Reddit, SEC EDGAR, and X.com.
"""

import logging
from agents.state import AnalysisState
from utils.sentiment_scraper import get_aggregate_sentiment

logger = logging.getLogger(__name__)


def sentiment_agent(state: AnalysisState) -> AnalysisState:
    """
    Fetch and aggregate multi-source sentiment.
    Populates: sentiment
    """
    ticker = state.get("ticker", "")
    errors = list(state.get("errors", []))

    try:
        sentiment = get_aggregate_sentiment(ticker)
        return {**state, "sentiment": sentiment, "errors": errors}
    except Exception as e:
        errors.append(f"SentimentAgent error: {str(e)}")
        logger.error(f"SentimentAgent [{ticker}]: {e}")
        return {**state, "sentiment": {}, "errors": errors}
