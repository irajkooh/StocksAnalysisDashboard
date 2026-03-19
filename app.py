"""
app.py — Application Launcher
Starts the FastAPI backend and Gradio frontend in parallel threads.
Run:  python app.py

Startup sequence:
  1. Clear terminal screen
  2. Kill any processes already bound to the required ports
  3. Print banner
  4. Start FastAPI backend thread
  5. Start Gradio frontend thread
  6. Open browser automatically once frontend is ready
"""

import os
import sys
import time
import signal
import socket
import logging
import threading
import subprocess
import webbrowser

# ── Logging (clean format, no timestamps in banner phase) ─────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")


# ─── 1. Clear Screen ──────────────────────────────────────────────────────────

def clear_screen():
    """Cross-platform terminal clear."""
    os.system("cls" if os.name == "nt" else "clear")


# ─── 2. Kill Ports ────────────────────────────────────────────────────────────

def kill_port(port: int):
    """
    Find and kill any process currently listening on `port`.
    Works on macOS, Linux, and Windows.
    """
    killed = False

    if os.name == "nt":
        # Windows: netstat + taskkill
        try:
            result = subprocess.check_output(
                f'netstat -ano | findstr :{port}', shell=True, text=True
            )
            for line in result.strip().splitlines():
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pid = int(parts[-1])
                    subprocess.call(f"taskkill /F /PID {pid}", shell=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    killed = True
        except subprocess.CalledProcessError:
            pass  # nothing on this port
    else:
        # macOS / Linux: lsof
        try:
            result = subprocess.check_output(
                ["lsof", "-ti", f"tcp:{port}"], text=True
            ).strip()
            if result:
                for pid_str in result.splitlines():
                    pid = int(pid_str.strip())
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed = True
                    except ProcessLookupError:
                        pass
        except (subprocess.CalledProcessError, FileNotFoundError):
            # lsof not available — try fuser
            try:
                subprocess.call(
                    ["fuser", "-k", f"{port}/tcp"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                killed = True
            except FileNotFoundError:
                pass

    if killed:
        print(f"  ⚠️  Killed existing process on port {port}")
        time.sleep(0.5)   # brief pause to let the OS release the port
    else:
        print(f"  ✅  Port {port} is free")


# ─── 3. Port-ready probe ──────────────────────────────────────────────────────

def wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    """Poll until a TCP listener appears on host:port, or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.4)
    return False


# ─── 4. Banner ────────────────────────────────────────────────────────────────

def print_banner(backend_port, frontend_port, env, device, llm):
    BLUE  = "\033[94m"
    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"

    width = 62  # inner width between ║ and ║

    def _row(label, value, vcolor=""):
        """Render a labeled row, accounting for wide emoji in value."""
        wide = sum(1 for c in value if ord(c) > 0xFFFF or c in "🔵🟢⚪🟡🔴")
        pad  = width - 16 - len(value) - wide
        return (f"{BLUE}{BOLD}║{RESET}  {CYAN}{label}:{RESET}  "
                f"{vcolor}{value}{RESET}{' ' * max(pad, 0)}{BLUE}{BOLD}║{RESET}")

    def _url_row(label, port):
        url = f"http://localhost:{port}"
        pad = width - 16 - len(url)
        return (f"{BLUE}{BOLD}║{RESET}  {CYAN}{label}:{RESET}  "
                f"{GREEN}{url}{RESET}{' ' * max(pad, 0)}{BLUE}{BOLD}║{RESET}")

    # Title — 📊 is 2 columns wide, Python len() counts it as 1
    title     = "📊  STOCKS ANALYSIS DASHBOARD"
    vis_len   = len(title) + 1          # +1 for the wide emoji
    pad_total = width - vis_len         # 62 - 29 = 33
    pad_l     = pad_total // 2          # 16
    pad_r     = pad_total - pad_l       # 17
    title_str = f"{' ' * pad_l}{title}{' ' * pad_r}"

    print(f"\n{BLUE}{BOLD}{'╔' + '═'*width + '╗'}{RESET}")
    print(f"{BLUE}{BOLD}║{RESET}{' ' * width}{BLUE}{BOLD}║{RESET}")
    print(f"{BLUE}{BOLD}║{RESET}{CYAN}{BOLD}{title_str}{RESET}{BLUE}{BOLD}║{RESET}")
    print(f"{BLUE}{BOLD}║{RESET}{' ' * width}{BLUE}{BOLD}║{RESET}")
    print(f"{BLUE}{BOLD}{'╠' + '═'*width + '╣'}{RESET}")
    print(_row("Environment", env))
    print(_row("Device     ", device))
    print(_row("LLM        ", llm))
    print(_url_row("Backend    ", backend_port))
    print(_url_row("Frontend   ", frontend_port))
    print(f"{BLUE}{BOLD}║{RESET}{' ' * width}{BLUE}{BOLD}║{RESET}")
    print(f"{BLUE}{BOLD}{'╚' + '═'*width + '╝'}{RESET}\n")


# ─── 5. Server starters ───────────────────────────────────────────────────────

def start_backend():
    import uvicorn
    from config import BACKEND_PORT
    logger.info(f"FastAPI backend binding on port {BACKEND_PORT}…")
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=BACKEND_PORT,
        reload=False,
        log_level="warning",
    )


def launch_frontend():
    """Must be called from the main thread — Gradio 5 requires it."""
    from config import FRONTEND_PORT
    from frontend import build_app
    import tempfile
    logger.info(f"Gradio frontend binding on port {FRONTEND_PORT}…")
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=FRONTEND_PORT,
        share=False,
        show_error=True,
        quiet=True,
        allowed_paths=[tempfile.gettempdir()],
    )


# ─── 6. Browser opener ────────────────────────────────────────────────────────

def open_browser_when_ready(frontend_port: int):
    """Wait until the Gradio server is actually accepting connections, then open."""
    url = f"http://localhost:{frontend_port}"
    print(f"  ⏳  Waiting for frontend to be ready…")
    if wait_for_port("localhost", frontend_port, timeout=60):
        time.sleep(0.8)   # tiny extra delay for Gradio to finish route setup
        print(f"  🌐  Opening browser → {url}\n")
        webbrowser.open(url)
    else:
        print(f"  ⚠️  Timed out waiting for frontend. Open manually: {url}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from config import BACKEND_PORT, FRONTEND_PORT, IS_HF_SPACE, LLM_PROVIDER, OLLAMA_MODEL, GROQ_MODEL, HF_MODEL
    from utils.device import get_device_label

    # Step 1 — clear screen
    clear_screen()

    env    = "HuggingFace Spaces" if IS_HF_SPACE else "Local"
    device = get_device_label()
    llm    = {"ollama": f"Ollama / {OLLAMA_MODEL}", "groq": f"Groq / {GROQ_MODEL}", "hf": f"huggingface ({HF_MODEL})"}.get(LLM_PROVIDER, LLM_PROVIDER)

    # Step 2 — kill occupied ports
    print("\n  🔧  Checking ports…")
    kill_port(BACKEND_PORT)
    kill_port(FRONTEND_PORT)

    # Step 3 — banner
    print_banner(BACKEND_PORT, FRONTEND_PORT, env, device, llm)

    # Step 4 — start FastAPI backend in a background thread
    backend_thread = threading.Thread(target=start_backend, daemon=True, name="backend")
    backend_thread.start()
    time.sleep(2)   # give backend a moment to bind

    # Step 5 — open browser in background (local only)
    if not IS_HF_SPACE:
        browser_thread = threading.Thread(
            target=open_browser_when_ready,
            args=(FRONTEND_PORT,),
            daemon=True,
            name="browser",
        )
        browser_thread.start()

    # Step 6 — launch Gradio in the main thread (required by Gradio 5)
    try:
        launch_frontend()
    except KeyboardInterrupt:
        print("\n  👋  Shutting down StocksAnalysisDashboard. Goodbye!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
