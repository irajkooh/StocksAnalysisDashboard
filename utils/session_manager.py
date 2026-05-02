"""
utils/session_manager.py — Persistent session save/load
Stores active symbols and their "I own this stock" state to a JSON file.

Supports per-user sessions stored in utils/sessions/{username}.json.
On HF Spaces each user's file is additionally synced to a private HF dataset.
"""

import json
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone

from utils.config import UTILS_DIR, IS_HF_SPACE, HF_TOKEN, HF_USER

logger = logging.getLogger(__name__)

_HF_DATASET_REPO = f"{HF_USER}/StocksAnalysisDashboard"

SESSIONS_DIR = UTILS_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

_USERNAME_RE = re.compile(r'^[\w\-]{1,32}$')

DEFAULT_USER       = "Default User"
_DEFAULT_FILE_STEM = "_default_"

# ─── Migrate legacy anonymous session (.json → _default_.json) ───────────────
_legacy = SESSIONS_DIR / ".json"
if _legacy.exists():
    _legacy.rename(SESSIONS_DIR / f"{_DEFAULT_FILE_STEM}.json")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _user_file(username: str) -> Path:
    if not username or username == DEFAULT_USER:
        return SESSIONS_DIR / f"{_DEFAULT_FILE_STEM}.json"
    return SESSIONS_DIR / f"{username}.json"


def _hf_user_path(username: str) -> str:
    return f"sessions/{username}.json"


# ─── HF Hub sync (per-user) ───────────────────────────────────────────────────

def _hf_pull_user(username: str) -> bool:
    if not IS_HF_SPACE or not HF_TOKEN:
        return False
    import concurrent.futures
    def _do_pull():
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(
            repo_id=_HF_DATASET_REPO,
            filename=_hf_user_path(username),
            repo_type="dataset",
            token=HF_TOKEN,
            force_download=True,
        )
        shutil.copy2(cached, _user_file(username))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(_do_pull).result(timeout=5)
        logger.info(f"Session pulled for '{username}' from HF Hub")
        return True
    except concurrent.futures.TimeoutError:
        logger.warning(f"HF pull timed out for '{username}', using local file")
        return False
    except Exception as e:
        logger.warning(f"Could not pull session for '{username}' from HF Hub: {e}")
        return False


def _hf_push_user(username: str) -> None:
    if not IS_HF_SPACE or not HF_TOKEN:
        return
    ufile    = _user_file(username)
    hf_path  = _hf_user_path(username)

    def _push():
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.create_repo(repo_id=_HF_DATASET_REPO, repo_type="dataset",
                            private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=str(ufile), path_in_repo=hf_path,
                            repo_id=_HF_DATASET_REPO, repo_type="dataset")
            logger.info(f"Session pushed for '{username}' to HF Hub")
        except Exception as e:
            logger.warning(f"Could not push session for '{username}' to HF Hub: {e}")

    threading.Thread(target=_push, daemon=True).start()


# ─── Public API ───────────────────────────────────────────────────────────────

def list_users() -> List[str]:
    """Return sorted list of existing usernames (DEFAULT_USER listed first)."""
    names = []
    for p in SESSIONS_DIR.glob("*.json"):
        names.append(DEFAULT_USER if p.stem == _DEFAULT_FILE_STEM else p.stem)
    others = sorted(n for n in names if n != DEFAULT_USER)
    return ([DEFAULT_USER] if DEFAULT_USER in names else []) + others


def create_user(username: str) -> Tuple[bool, str]:
    """Create a new empty session for username. Returns (ok, error_message)."""
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if username == DEFAULT_USER:
        return False, f"'{DEFAULT_USER}' is a reserved name."
    if not _USERNAME_RE.match(username):
        return False, "Use only letters, numbers, underscores, hyphens (max 32 chars)."
    ufile = _user_file(username)
    if ufile.exists():
        return False, f"User '{username}' already exists."
    try:
        with open(ufile, "w") as f:
            json.dump(_empty_session(), f, indent=2)
        logger.info(f"Created user '{username}'")
        return True, ""
    except Exception as e:
        return False, str(e)


def delete_user(username: str) -> Tuple[bool, str]:
    """Delete a user's session file. Returns (ok, error_message)."""
    if username == DEFAULT_USER:
        return False, f"'{DEFAULT_USER}' cannot be deleted."
    ufile = _user_file(username)
    if not ufile.exists():
        return False, f"User '{username}' not found."
    try:
        ufile.unlink()
        logger.info(f"Deleted user '{username}'")
        return True, ""
    except Exception as e:
        return False, str(e)


def rename_user(old_name: str, new_name: str) -> Tuple[bool, str]:
    """Rename a user. Returns (ok, error_message)."""
    if old_name == DEFAULT_USER:
        return False, f"'{DEFAULT_USER}' cannot be renamed."
    new_name = new_name.strip()
    if not new_name:
        return False, "New username cannot be empty."
    if new_name == DEFAULT_USER:
        return False, f"'{DEFAULT_USER}' is a reserved name."
    if not _USERNAME_RE.match(new_name):
        return False, "Use only letters, numbers, underscores, hyphens (max 32 chars)."
    old_file = _user_file(old_name)
    new_file = _user_file(new_name)
    if not old_file.exists():
        return False, f"User '{old_name}' not found."
    if new_file.exists():
        return False, f"User '{new_name}' already exists."
    try:
        old_file.rename(new_file)
        logger.info(f"Renamed user '{old_name}' → '{new_name}'")
        return True, ""
    except Exception as e:
        return False, str(e)


def load_session(username: str) -> Dict:
    """Load saved session for the given username."""
    if IS_HF_SPACE:
        _hf_pull_user(username)
    ufile = _user_file(username)
    if ufile.exists():
        try:
            with open(ufile, "r") as f:
                data = json.load(f)
            logger.info(f"Session loaded for '{username}': {len(data.get('symbols', []))} symbols")
            return data
        except Exception as e:
            logger.warning(f"Could not load session for '{username}': {e}")
    return _empty_session()


def save_session(symbols: List[str], owned: Dict[str, bool],
                 watchlist: List[str] = None,
                 refresh_interval: str = "Off",
                 snapshots: Dict[str, dict] = None,
                 username: str = "") -> Tuple[bool, str]:
    """Persist dashboard state for the given username."""
    data = {
        "version":          "1.0",
        "saved_at":         datetime.now(timezone.utc).isoformat(),
        "symbols":          symbols,
        "owned":            owned,
        "watchlist":        watchlist or [],
        "refresh_interval": refresh_interval,
        "snapshots":        {},
    }

    ufile = _user_file(username)
    try:
        with open(ufile, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Session saved for '{username}': {symbols}")
        _hf_push_user(username)
        return True, ""
    except Exception as e:
        logger.error(f"Could not save session for '{username}': {e}")
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
