# 📊 StocksAnalysisDashboard — Installation Guide

## Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.ai) installed (for local LLM)
- Git

---

## 1. Clone / Download the Project

```bash
git clone https://github.com/irajkoohi/StocksAnalysisDashboard.git
cd StocksAnalysisDashboard
```

Or unzip the downloaded archive.

---

## 2. Create a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon (MPS)**: PyTorch is auto-detected. No extra install needed — the `utils/device.py` module picks up MPS automatically.

> **Voice Input (optional)**: Uncomment `openai-whisper` in `requirements.txt` then:
> ```bash
> pip install openai-whisper
> brew install ffmpeg   # macOS
> # or: sudo apt install ffmpeg  (Linux)
> ```

---

## 4. Configure Ollama (Local LLM)

```bash
# Start Ollama service
ollama serve

# Pull the default model (in a new terminal)
ollama pull llama3.2:3b
```

> You can change the model in `config.py` → `OLLAMA_MODEL`

---

## 5. Set API Keys (Optional — for richer data)

Create a `.env` file in the project root:

```env
# LLM (only needed on HuggingFace Spaces — locally Ollama is used)
GROQ_API_KEY=your_groq_key_here

# News sentiment
NEWS_API_KEY=your_newsapi_key_here

# Reddit sentiment (get from https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret

# HuggingFace (for deployment)
HF_TOKEN=your_hf_token
```

> **Without API keys**: The app still works! yfinance provides market data, and Reddit uses the public JSON API as a fallback. NewsAPI and SEC EDGAR also have free tiers.

---

## 6. Run the Application

```bash
python app.py
```

This starts:
- **FastAPI backend** at `http://localhost:8000`
- **Gradio frontend** at `http://localhost:7860`

Open `http://localhost:7860` in your browser.

---

## 7. First Use

1. Type a stock symbol (e.g., `AAPL`) in the top input box
2. Click **➕ Add Symbol**
3. Refresh the browser page (tab is saved to `session.json`)
4. Click **🔍 Analyze AAPL** in the new tab
5. Wait 15–30 seconds for full multi-agent analysis
6. Check the **I OWN THIS** toggle if you hold the stock
7. Click **💾 Save** to persist your session

---

## 8. Deploy to HuggingFace Spaces

#### On Terminal:

cd "/Users/ik/UVcodes/StocksAnalysisDashboard"
git clone https://huggingface.co/spaces/irajkoohi/StocksAnalysisDashboard
cd StocksAnalysisDashboard

SRC='/Users/ik/UVcodes/StocksAnalysisDashboard'

cp "$SRC/app.py"              .
cp "$SRC/backend.py"          .
cp "$SRC/frontend.py"         .
cp "$SRC/config.py"           .
cp "$SRC/session.json" .
cp "$SRC/README.md"           .
cp "$SRC/requirements.txt"    .
cp "$SRC/_Plan.md"            .
cp "$SRC/_Instructions.md"    .
cp "$SRC/_HowItWorks.md"      .
cp -r "$SRC/agents"         .
cp -r "$SRC/utils"         .

git add -A
git commit -m "deploy: StocksAnalysisDashboard"
git push


```bash
# Create the Space (one-time)
pip install huggingface_hub
huggingface-cli login

# Create space
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('StocksAnalysisDashboard', repo_type='space', space_sdk='gradio')
"

# Push code
git init
git remote add origin https://huggingface.co/spaces/irajkoohi/StocksAnalysisDashboard
git add .
git commit -m 'Initial deployment'
git push origin main
```

Then in your HF Space settings, add Secrets:
- `GROQ_API_KEY` — required for LLM on HF (Ollama not available)
- `NEWS_API_KEY` — optional
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` — optional
- `HF_TOKEN` — your token

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Connection refused on port 8000` | Make sure backend started — check terminal logs |
| `Ollama not found` | Run `ollama serve` in a separate terminal |
| `No data for ticker` | Check ticker symbol is valid on yfinance (e.g., `GOOGL` not `GOOGLE`) |
| `Chart not showing` | Plotly may need: `pip install plotly kaleido` |
| `TTS not working` | `pip install gTTS` and check internet connection |
| `Reddit rate limited` | Wait 1 min; or set `REDDIT_CLIENT_ID` / `SECRET` for higher limits |

---

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2:3b` | Local LLM model |
| `DEFAULT_PERIOD` | `3mo` | Chart period (1d, 5d, 1mo, 3mo, 6mo, 1y) |
| `RSI_OVERSOLD` | `30` | RSI oversold threshold |
| `RSI_OVERBOUGHT` | `70` | RSI overbought threshold |
| `DCF_DISCOUNT_RATE` | `0.10` | WACC for intrinsic value calculation |
| `CACHE_TTL` | `300` | Analysis cache duration in seconds |
