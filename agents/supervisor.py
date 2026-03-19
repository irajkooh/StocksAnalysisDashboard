"""
agents/supervisor.py — LangGraph Supervisor / Orchestrator
Sequential pipeline: data → technical → sentiment → valuation → risk → decision
(Parallel fan-out removed — LangGraph raises INVALID_CONCURRENT_GRAPH_UPDATE
 when multiple nodes write to the same state keys simultaneously.)
"""

import time
import logging

from langgraph.graph import StateGraph, END

from agents.state import AnalysisState
from agents.data_agent      import data_agent
from agents.technical_agent import technical_agent
from agents.sentiment_agent import sentiment_agent
from agents.valuation_agent import valuation_agent
from agents.risk_agent      import risk_agent
from agents.decision_agent  import decision_agent

logger = logging.getLogger(__name__)


def build_graph():
    """
    Strictly sequential pipeline — no fan-out, no concurrent state writes:
      data → technical → sentiment → valuation → risk → decision → END
    """
    g = StateGraph(AnalysisState)

    g.add_node("data",      data_agent)
    g.add_node("technical", technical_agent)
    g.add_node("sentiment", sentiment_agent)
    g.add_node("valuation", valuation_agent)
    g.add_node("risk",      risk_agent)
    g.add_node("decision",  decision_agent)

    g.set_entry_point("data")
    g.add_edge("data",      "technical")
    g.add_edge("technical", "sentiment")
    g.add_edge("sentiment", "valuation")
    g.add_edge("valuation", "risk")
    g.add_edge("risk",      "decision")
    g.add_edge("decision",  END)

    return g.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_analysis(ticker, owns_stock=False, period="3mo", interval="1d"):
    graph = get_graph()
    t0    = time.time()

    initial_state: AnalysisState = {
        "ticker":     ticker.upper().strip(),
        "owns_stock": owns_stock,
        "period":     period,
        "interval":   interval,
        "errors":     [],
    }

    try:
        final_state = graph.invoke(initial_state)
        final_state["duration_ms"] = round((time.time() - t0) * 1000)
        return final_state
    except Exception as e:
        logger.error(f"Graph execution failed for {ticker}: {e}")
        return {**initial_state, "errors": [str(e)], "duration_ms": 0}


def get_mermaid_diagram():
    try:
        return get_graph().get_graph().draw_mermaid()
    except Exception as e:
        logger.warning(f"Mermaid generation failed: {e}")
        return _static_mermaid()


def _static_mermaid():
    return """graph TD
    A([START]) --> B[Data Agent\nyfinance OHLCV + Info]
    B --> C[Technical Agent\nRSI MACD SMA BB Fib]
    C --> D[Sentiment Agent\nNews Reddit X.com SEC]
    D --> E[Valuation Agent\nDCF Intrinsic Value]
    E --> F[Risk Agent\nVolatility Drawdown ATR]
    F --> G[Decision Agent\nBuy Hold Sell + LLM]
    G --> H([END])
    style A fill:#1e40af,color:#fff
    style B fill:#0f766e,color:#fff
    style C fill:#7c3aed,color:#fff
    style D fill:#b45309,color:#fff
    style E fill:#0369a1,color:#fff
    style F fill:#b91c1c,color:#fff
    style G fill:#065f46,color:#fff
    style H fill:#1e40af,color:#fff"""
