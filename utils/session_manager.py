"""
utils/session_manager.py — Persistent session save/load
Stores active symbols and their "I own this stock" state to a JSON file.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timezone

from utils.config import SESSION_FILE

logger = logging.getLogger(__name__)


def load_session() -> Dict:
    """Load saved session from disk; return empty session if not found."""
    if SESSION_FILE.exists():
        try:
            with open(SESSION_FILE, "r") as f:
                data = json.load(f)
            logger.info(f"Session loaded: {len(data.get('symbols', []))} symbols")
            return data
        except Exception as e:
            logger.warning(f"Could not load session: {e}")
    return _empty_session()


_SKIP_KEYS = {"chart_json", "_ts"}


def _strip_snapshot(data: dict) -> dict:
    """Remove non-serialisable / oversized keys before writing to disk."""
    return {k: v for k, v in data.items() if k not in _SKIP_KEYS}


def save_session(symbols: List[str], owned: Dict[str, bool],
                 watchlist: List[str] = None,
                 refresh_interval: str = "Off",
                 snapshots: Dict[str, dict] = None):
    """Persist current dashboard state to disk.
    Returns (True, "") on success or (False, error_message) on failure."""
    data = {
        "version":          "1.0",
        "saved_at":         datetime.now(timezone.utc).isoformat(),
        "symbols":          symbols,
        "owned":            owned,
        "watchlist":        watchlist or [],
        "refresh_interval": refresh_interval,
        "snapshots":        {k: _strip_snapshot(v) for k, v in (snapshots or {}).items()},
    }
    try:
        with open(SESSION_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Session saved to {SESSION_FILE}: {symbols}")
        return True, ""
    except Exception as e:
        logger.error(f"Could not save session to {SESSION_FILE}: {e}")
        return False, str(e)


def _empty_session() -> Dict:
    return {
        "version":          "1.0",
        "saved_at":         None,
        "symbols":          [],
        "owned":            {},
        "watchlist":        [],
        "refresh_interval": "Off",
        "snapshots":        {},
    }
