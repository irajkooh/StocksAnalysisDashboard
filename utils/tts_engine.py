"""
utils/tts_engine.py — Text-to-Speech using gTTS (Google TTS, American accent)
Generates an MP3 base64 data URI for use in Gradio Audio component.
"""

import io
import base64
import logging
import re
from typing import Optional

from config import TTS_LANG, TTS_SLOW, TTS_ACCENT

logger = logging.getLogger(__name__)


def text_to_speech_b64(text: str) -> Optional[str]:
    """
    Convert text to speech and return a base64-encoded WAV/MP3 data URI.
    Returns None if gTTS is unavailable.
    """
    try:
        from gtts import gTTS
        clean = _clean_for_tts(text)
        tts   = gTTS(text=clean, lang=TTS_LANG, slow=TTS_SLOW, tld=TTS_ACCENT)
        buf   = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:audio/mp3;base64,{b64}"
    except ImportError:
        logger.warning("gTTS not installed; TTS unavailable")
        return None
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None


def text_to_speech_file(text: str, path: str = "/tmp/tts_output.mp3") -> Optional[str]:
    """Save TTS to a file and return the path (for Gradio Audio file mode)."""
    try:
        from gtts import gTTS
        clean = _clean_for_tts(text)
        gTTS(text=clean, lang=TTS_LANG, slow=TTS_SLOW, tld=TTS_ACCENT).save(path)
        return path
    except Exception as e:
        logger.error(f"TTS file error: {e}")
        return None


def _clean_for_tts(text: str) -> str:
    """Strip markdown/HTML/emoji from text before TTS."""
    text = re.sub(r"<[^>]+>", " ", text)           # HTML tags
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # bold/italic
    text = re.sub(r"#{1,6}\s", "", text)            # headings
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links
    text = re.sub(r"[🟢🔴🟡🟠⚪📊📈📉💹🔵⚠️✅❌]", "", text)  # common emoji
    text = re.sub(r"\s+", " ", text).strip()
    return text[:3000]  # gTTS character safety limit
