"""
utils/session_manager.py — Persistent session save/load
Stores active symbols and their "I own this stock" state to a JSON file.

On HF Spaces the session is additionally synced to a private HF dataset
(irajkoohi/stocks-dashboard-session) so it survives Space restarts.
"""

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timezone

from utils.config import SESSION_FILE, IS_HF_SPACE, HF_TOKEN, HF_USER

logger = logging.getLogger(__name__)

_HF_DATASET_REPO = f"{HF_USER}/stocks-dashboard-session"
_HF_FILENAME     = "session.json"


# ─── HF Hub sync helpers ──────────────────────────────────────────────────────

def _hf_pull_session() -> bool:
    """Download session.json from the HF dataset into SESSION_FILE.
    Returns True on success.  No-op when not on HF Spaces.
    Works with both public datasets (no token) and private datasets (token required).
    Tries with token first, then falls back to no-auth for public datasets."""
    if not IS_HF_SPACE:
        return False
    url = f"https://huggingface.co/datasets/{_HF_DATASET_REPO}/resolve/main/{_HF_FILENAME}"
    logger.info(f"HF pull: IS_HF_SPACE=True HF_TOKEN={'set' if HF_TOKEN else 'missing'} url={url}")
    try:
        import requests as _requests
        # Try with token first (private dataset), then without (public dataset)
        attempts = []
        if HF_TOKEN:
            attempts.append({"Authorization": f"Bearer {HF_TOKEN}"})
        attempts.append({})  # no-auth attempt (works if dataset is public)
        for headers in attempts:
            resp = _requests.get(url, headers=headers, timeout=15)
            logger.info(f"HF pull HTTP {resp.status_code} (auth={'yes' if headers else 'no'})")
            if resp.status_code == 200:
                SESSION_FILE.write_bytes(resp.content)
                logger.info(f"Session pulled from HF Hub ({_HF_DATASET_REPO}) → {SESSION_FILE}")
                return True
        logger.warning(f"HF pull failed all attempts — last HTTP {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        logger.warning(f"Could not pull session from HF Hub: {e}")
        return False


def _hf_push_session() -> None:
    """Upload SESSION_FILE to the HF dataset in a daemon thread (fire-and-forget)."""
    if not IS_HF_SPACE or not HF_TOKEN:
        return

    def _push():
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.create_repo(
                repo_id=_HF_DATASET_REPO,
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )
            api.upload_file(
                path_or_fileobj=str(SESSION_FILE),
                path_in_repo=_HF_FILENAME,
                repo_id=_HF_DATASET_REPO,
                repo_type="dataset",
            )
            logger.info(f"Session pushed to HF Hub ({_HF_DATASET_REPO})")
        except Exception as e:
            logger.warning(f"Could not push session to HF Hub: {e}")

    threading.Thread(target=_push, daemon=True).start()


# ─── Public API ───────────────────────────────────────────────────────────────

def load_session() -> Dict:
    """Load saved session from disk (pulling from HF Hub first on Spaces)."""
    # On HF Spaces pull the latest persisted copy before reading the local file
    if IS_HF_SPACE:
        _hf_pull_session()

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
    """Persist current dashboard state to disk (and HF Hub on Spaces).
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
        # Push to HF Hub asynchronously so the UI isn't blocked
        _hf_push_session()
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
