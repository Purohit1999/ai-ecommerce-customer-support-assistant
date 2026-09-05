"""Streamlit chat interface for the local customer-support demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from agent import CustomerSupportAgent
from config import CHROMA_STORE_DIR, WORKING_DATA_DIR
from llm import LLMEnhancedSupportAgent, configure_optional_llm
from prompts import HUMAN_ESCALATION_MESSAGE
from rag import RAGService


APP_TITLE = "AI-Powered E-Commerce Customer Support Assistant"
QUICK_ACTIONS = (
    ("📦 Track Order", "Where is my order?"),
    ("↩️ Start a Return", "I want to return my item."),
    ("💳 Check Refund", "Where is my refund?"),
    ("📚 Return Policy", "What is your return policy?"),
)

DARK_UI_CSS = """
<style>
/* Dark application shell */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #080d16;
    color: #e8eef8;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 50% -18%, rgba(55, 80, 150, 0.20), transparent 30rem),
        #080d16;
}
[data-testid="stHeader"] {
    background: rgba(8, 13, 22, 0.88);
    backdrop-filter: blur(10px);
}
.block-container {
    width: min(1100px, 94vw);
    max-width: 1100px;
    padding-top: 4.75rem;
    padding-bottom: 3rem;
}
.stApp p, .stApp li, .stApp label {
    color: #c4cee0;
}

/* Compact header and centred hero */
h1 {
    margin: 1.2rem 0 0.3rem !important;
    color: #f6f8fc !important;
    font-size: clamp(2rem, 4vw, 3.25rem) !important;
    letter-spacing: -0.035em;
    text-align: center;
}
h3 {
    color: #f1f5f9 !important;
    font-size: 1.2rem !important;
    font-weight: 650 !important;
    line-height: 1.4 !important;
}
h5 {
    margin: 0 0 0.9rem !important;
    color: #9daac0 !important;
    font-size: clamp(0.95rem, 1.6vw, 1.08rem) !important;
    font-weight: 400 !important;
    text-align: center;
}
hr {
    margin: 0.45rem 0 0.8rem !important;
    border-color: rgba(144, 158, 187, 0.14) !important;
}

/* Native controls */
.stButton > button {
    min-height: 2.75rem;
    border: 1px solid rgba(137, 153, 189, 0.22);
    border-radius: 11px;
    background: rgba(18, 27, 45, 0.92);
    color: #dce5f5;
    font-weight: 600;
    transition: border-color 150ms ease, background 150ms ease, transform 150ms ease;
}
.stButton > button:hover {
    border-color: rgba(104, 132, 224, 0.72);
    background: rgba(31, 44, 73, 0.96);
    color: #ffffff;
    transform: translateY(-1px);
}
.stButton > button:focus {
    box-shadow: 0 0 0 3px rgba(75, 105, 205, 0.18);
}
[data-testid="stChatInput"] {
    border: 1px solid rgba(105, 132, 220, 0.55);
    border-radius: 15px;
    background: #111a2b;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
}
[data-testid="stChatInput"] textarea {
    color: #f2f5fb;
}

/* Centred conversation panel */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(139, 154, 187, 0.18) !important;
    border-radius: 16px !important;
    background: rgba(11, 18, 31, 0.72);
}
[data-testid="stChatMessage"] {
    margin: 0.7rem 0;
    padding: 0.95rem 1.05rem;
    border: 1px solid rgba(139, 154, 187, 0.15);
    border-radius: 14px;
    background: rgba(17, 26, 44, 0.92);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    margin-left: 8%;
    border-color: rgba(92, 124, 223, 0.34);
    background: rgba(30, 44, 78, 0.88);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    margin-right: 8%;
}
[data-testid="stExpander"] {
    border: 1px solid rgba(139, 154, 187, 0.17);
    border-radius: 11px;
    background: rgba(8, 14, 25, 0.66);
}
[data-testid="stAlert"] {
    border-radius: 11px;
}

@media (max-width: 700px) {
    .block-container {
        width: 100%;
        padding: 4.25rem 0.8rem 2rem;
    }
    h1 {
        margin-top: 0.8rem !important;
    }
    [data-testid="stChatMessage"] {
        margin-left: 0 !important;
        margin-right: 0 !important;
        padding: 0.8rem;
    }
    .stButton > button {
        min-height: 2.5rem;
        font-size: 0.82rem;
    }
}
</style>
"""


def initialize_agent(
    persist_directory: Path = CHROMA_STORE_DIR,
    data_directory: Path = WORKING_DATA_DIR,
) -> CustomerSupportAgent | LLMEnhancedSupportAgent:
    """Create the local agent, building its index only when required."""
    rag_service = RAGService(
        persist_directory=persist_directory,
        data_directory=data_directory,
        include_products=True,
    )
    if rag_service.index_count() == 0:
        rag_service.build_index()
    base_agent = CustomerSupportAgent(rag_service=rag_service)
    return configure_optional_llm(base_agent)


def safe_agent_response(
    agent: CustomerSupportAgent | LLMEnhancedSupportAgent, query: str
) -> dict[str, Any]:
    """Run the agent and replace unexpected failures with a safe response."""
    try:
        result = agent.respond(query)
        required_keys = {"route", "answer", "sources", "tool_result", "escalation"}
        if not isinstance(result, dict) or not required_keys.issubset(result):
            raise ValueError("Agent returned an invalid response structure")
        return result
    except Exception:
        return {
            "route": "human_escalation",
            "answer": HUMAN_ESCALATION_MESSAGE,
            "sources": [],
            "tool_result": None,
            "escalation": True,
        }


def response_notice(result: dict[str, Any]) -> tuple[str, str] | None:
    """Return the UI notice level and text appropriate for a response."""
    if result.get("route") == "clarification":
        return "warning", "Clarification needed"
    if result.get("escalation"):
        return "error", "Human support escalation recommended"

    tool_result = result.get("tool_result")
    if (
        result.get("route") == "return_request"
        and isinstance(tool_result, dict)
        and tool_result.get("simulated") is True
        and tool_result.get("persisted") is False
    ):
        return "info", "Demo simulation only — this return request was not persisted"
    return None


@st.cache_resource(show_spinner=False)
def _cached_agent() -> CustomerSupportAgent | LLMEnhancedSupportAgent:
    return initialize_agent()


def _apply_theme() -> None:
    """Apply presentation CSS; all visible layout remains Streamlit-native."""
    st.markdown(DARK_UI_CSS, unsafe_allow_html=True)


def _render_header() -> None:
    brand, status, clear = st.columns(
        [5.8, 1.35, 1.05], gap="small", vertical_alignment="center"
    )
    with brand:
        st.markdown("### 🛍 E-Commerce Support")
    with status:
        st.markdown("**● Offline Demo**")
    with clear:
        if st.button(
            "Clear Chat",
            key="clear-chat",
            help="Remove this session's conversation",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()
    st.divider()


def _render_quick_actions() -> str | None:
    selected_prompt = None
    columns = st.columns(4, gap="small")
    for column, (label, prompt) in zip(columns, QUICK_ACTIONS):
        with column:
            if st.button(
                label,
                key=f"quick-{label}",
                use_container_width=True,
            ):
                selected_prompt = prompt
    return selected_prompt


def _route_badge(route: object) -> str:
    labels = {
        "rag": "RAG",
        "order_status": "ORDER STATUS",
        "return_request": "RETURN",
        "refund_status": "REFUND",
        "clarification": "CLARIFICATION",
        "human_escalation": "HUMAN ESCALATION",
    }
    normalized = str(route)
    return labels.get(normalized, normalized.replace("_", " ").upper())


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander("Sources and citations"):
        for source in sources:
            metadata = source.get("metadata", {})
            section = metadata.get("section", "Unspecified section")
            st.markdown(f"- **{source['citation']}** — {section}")
            st.caption(
                f"Source: {source['source']} · relevance score: "
                f"{float(source['score']):.3f}"
            )


def _render_assistant_result(result: dict[str, Any]) -> None:
    notice = response_notice(result)
    if notice:
        level, message = notice
        getattr(st, level)(message)

    st.markdown(str(result["answer"]))
    st.caption(_route_badge(result["route"]))
    _render_sources(result.get("sources", []))

    tool_result = result.get("tool_result")
    if tool_result is not None:
        with st.expander("Local tool result"):
            st.json(tool_result)


def _render_history() -> None:
    for message in st.session_state.messages:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                _render_assistant_result(message["result"])
            else:
                st.markdown(message["content"])


def _startup_error_result() -> dict[str, Any]:
    return {
        "route": "human_escalation",
        "answer": (
            "The local support knowledge base could not be initialized. "
            "Please check the project data files or contact human support."
        ),
        "sources": [],
        "tool_result": None,
        "escalation": True,
    }


def main() -> None:
    """Render and run the end-to-end local support demo."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_theme()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    _render_header()
    st.title("How can we help today?")
    st.markdown(
        "##### Ask about your order, return, refund, products or store policies."
    )

    input_left, input_centre, input_right = st.columns([1, 7, 1])
    with input_centre:
        typed_message = st.chat_input("Type your question here…")

    action_left, action_centre, action_right = st.columns([1, 7, 1])
    with action_centre:
        selected_prompt = _render_quick_actions()

    try:
        with st.spinner("Preparing local support information..."):
            agent = _cached_agent()
        startup_error = None
    except Exception:
        agent = None
        startup_error = _startup_error_result()

    user_message = typed_message or selected_prompt
    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})
        result = (
            safe_agent_response(agent, user_message)
            if agent is not None
            else startup_error
        )
        st.session_state.messages.append({"role": "assistant", "result": result})

    chat_left, chat_centre, chat_right = st.columns([1, 7, 1])
    with chat_centre:
        st.subheader("Conversation")
        with st.container(border=True):
            if st.session_state.messages:
                _render_history()
            else:
                st.caption(
                    "Your conversation will appear here. Choose a quick action "
                    "or ask a question above."
                )


if __name__ == "__main__":
    main()
