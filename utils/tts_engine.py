"""utils/tts_engine.py — Text-to-Speech using gTTS (American accent)
Saves to system temp dir (allowed by Gradio) or cwd fallback.
"""
import io, os, re, tempfile, logging
from pathlib import Path

logger = logging.getLogger(__name__)

def text_to_speech_file(text: str) -> str | None:
    """Convert text to speech. Returns path in system temp dir."""
    try:
        from gtts import gTTS
        clean = _clean(text)
        if not clean.strip():
            return None
        tts = gTTS(text=clean, lang="en", slow=False, tld="com")
        # Use system temp dir — always allowed by Gradio
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False,
                                          dir=tempfile.gettempdir())
        tts.save(tmp.name)
        tmp.close()
        return tmp.name
    except ImportError:
        logger.warning("gTTS not installed")
        return None
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"#{1,6}\s", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)  # strip non-ASCII (emoji)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:3000]
