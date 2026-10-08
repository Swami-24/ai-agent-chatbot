import os
import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools import DuckDuckGoSearchRun


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL ENTERPRISE CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

   :root {
    --bg: #F4F7FB;
    --panel: #FFFFFF;
    --panel-2: #F8FAFC;
    --border: #D9E2EC;
    --border-light: #CBD5E1;

    --text: #172033;
    --text-soft: #475569;
    --text-muted: #64748B;

    --blue: #2563EB;
    --blue-dark: #1D4ED8;
    --green: #16A34A;
    --orange: #F59E0B;
}


    /* ========================================================
       REMOVE DEFAULT WHITE AREAS
       ======================================================== */

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    .main,
    .block-container {
        background: var(--bg) !important;
        color: var(--text) !important;
    }


    .stApp {
        min-height: 100vh !important;
    }


    [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
    }


    [data-testid="stAppViewContainer"] > section {
        background: var(--bg) !important;
    }


    .main {
        background: var(--bg) !important;
    }


    .block-container {
        max-width: 1500px !important;

        padding-top: 1.2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;

        padding-bottom: 8rem !important;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    header[data-testid="stHeader"] {
        background: var(--bg) !important;
    }


    [data-testid="stToolbar"] {
        background: transparent !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #081321 !important;
    }


    [data-testid="stSidebar"] {
        background: #081321 !important;

        border-right: 1px solid #1C2D41 !important;
    }


    [data-testid="stSidebar"] > div:first-child {
        background: #081321 !important;

        padding: 22px 16px !important;
    }


    [data-testid="stSidebar"] p {
        color: #B9C6D5 !important;
    }


    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #F8FAFC !important;
    }


    [data-testid="stSidebar"] hr {
        border-color: #1B2B3E !important;

        margin: 21px 0 !important;
    }


    /* ========================================================
       SIDEBAR BRAND
       ======================================================== */

    .side-brand {
        display: flex;

        align-items: center;

        gap: 11px;

        margin-bottom: 5px;
    }


    .side-logo {
        width: 43px;
        height: 43px;

        display: flex;

        align-items: center;
        justify-content: center;

        border-radius: 11px;

        background: #0D2639;

        border: 1px solid #20516E;

        color: #67E8F9 !important;

        font-size: 20px;
    }


    .side-title {
        color: #FFFFFF !important;

        font-size: 1.05rem;

        font-weight: 800;

        line-height: 1.2;
    }


    .side-subtitle {
        color: #6F8298 !important;

        font-size: 0.62rem;

        letter-spacing: 1.3px;

        font-weight: 800;

        margin-top: 4px;
    }


    /* ========================================================
       SIDEBAR SECTION
       ======================================================== */

    .side-section {
        color: #67D7F4 !important;

        font-size: 0.68rem;

        font-weight: 800;

        letter-spacing: 1.4px;

        text-transform: uppercase;

        margin-bottom: 10px;
    }


    /* ========================================================
       WORKFLOW
       ======================================================== */

    .workflow {
        display: flex;

        align-items: flex-start;

        gap: 10px;

        margin: 9px 0;
    }


    .workflow-num {
        min-width: 25px;

        height: 25px;

        display: flex;

        align-items: center;

        justify-content: center;

        background: #0D1D2E;

        border: 1px solid #28445B;

        border-radius: 6px;

        color: #67E8F9 !important;

        font-size: 0.65rem;

        font-weight: 800;
    }


    .workflow-text {
        color: #B7C4D3 !important;

        font-size: 0.78rem;

        line-height: 1.5;

        padding-top: 2px;
    }


    /* ========================================================
       SIDEBAR TOOLS
       ======================================================== */

    .tool-box {
        display: flex;

        align-items: center;

        gap: 9px;

        padding: 9px 11px;

        margin-bottom: 7px;

        background: #0C1928;

        border: 1px solid #1D3045;

        border-radius: 8px;

        color: #C9D5E2 !important;

        font-size: 0.77rem;
    }


    .tool-box:hover {
        border-color: #2A526C;
        background: #0F2032;
    }


    /* ========================================================
       MODEL CARD
       ======================================================== */

    .model-card {
        background: #0C1928;

        border: 1px solid #24445D;

        border-radius: 10px;

        padding: 12px;
    }


    .model-label {
        color: #718399 !important;

        font-size: 0.63rem;

        text-transform: uppercase;

        letter-spacing: 1px;

        font-weight: 800;

        margin-bottom: 6px;
    }


    .model-value {
        color: #7DD3FC !important;

        font-family: monospace;

        font-size: 0.75rem;

        font-weight: 700;

        word-break: break-word;
    }


    /* ========================================================
       SECURITY
       ======================================================== */

    .security-card {
        background: #0A1919;

        border: 1px solid #1E3D3B;

        border-radius: 9px;

        padding: 11px;

        color: #A7BFBC !important;

        font-size: 0.71rem;

        line-height: 1.55;
    }


    /* ========================================================
       SIDEBAR BUTTON
       ======================================================== */

    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;

        background: #0E1A2A !important;

        color: #CBD5E1 !important;

        border: 1px solid #2A3D53 !important;

        border-radius: 9px !important;

        min-height: 40px !important;

        font-size: 0.79rem !important;

        font-weight: 700 !important;
    }


    [data-testid="stSidebar"] .stButton > button:hover {
        background: #13263A !important;

        color: #FFFFFF !important;

        border-color: #38BDF8 !important;
    }


    [data-testid="stSidebar"] .stButton > button p {
        color: inherit !important;
    }


    /* ========================================================
       TOP NAVBAR
       ======================================================== */

    .topbar {
        min-height: 61px;

        display: flex;

        align-items: center;

        justify-content: space-between;

        padding: 0 4px 14px 4px;

        border-bottom: 1px solid #18283B;

        margin-bottom: 22px;
    }


    .top-left {
        display: flex;

        align-items: center;

        gap: 12px;
    }


    .top-logo {
        width: 40px;
        height: 40px;

        display: flex;

        align-items: center;

        justify-content: center;

        background: #0D2639;

        border: 1px solid #21516D;

        border-radius: 9px;

        color: #67E8F9 !important;

        font-size: 19px;
    }


    .top-title {
        color: #F8FAFC !important;

        font-size: 1rem;

        font-weight: 800;
    }


    .top-subtitle {
        color: #718198 !important;

        font-size: 0.67rem;

        margin-top: 3px;
    }


    .online {
        display: flex;

        align-items: center;

        gap: 7px;

        padding: 6px 11px;

        background: #0A1E19;

        border: 1px solid #1B493D;

        border-radius: 20px;

        color: #6EE7B7 !important;

        font-size: 0.66rem;

        font-weight: 800;
    }


    .online-dot {
        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: #34D399;

        box-shadow: 0 0 8px #34D399;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        background:
            linear-gradient(
                110deg,
                #0D1B2C 0%,
                #10283B 55%,
                #0C2A38 100%
            );

        border: 1px solid #27465C;

        border-radius: 15px;

        padding: 34px 38px;

        margin-bottom: 22px;

        box-shadow:
            0 16px 45px rgba(0,0,0,0.22);
    }


    .hero-badge {
        display: inline-block;

        padding: 6px 11px;

        border-radius: 6px;

        background: #0D2A3B;

        border: 1px solid #24566F;

        color: #7DD3FC !important;

        font-size: 0.66rem;

        font-weight: 800;

        letter-spacing: 1.1px;

        margin-bottom: 13px;
    }


    .hero-title {
        color: #FFFFFF !important;

        font-size: 2.35rem;

        font-weight: 850;

        line-height: 1.15;

        margin-bottom: 11px;
    }


    .hero-title span {
        color: #67E8F9 !important;
    }


    .hero-description {
        color: #B9C8D8 !important;

        font-size: 0.94rem;

        max-width: 920px;

        line-height: 1.7;
    }


    /* ========================================================
       FEATURE CARDS
       ======================================================== */

    .feature-card {
        background: #0C1726;

        border: 1px solid #20354A;

        border-radius: 12px;

        padding: 19px;

        min-height: 145px;

        height: 100%;

        box-shadow:
            0 8px 25px rgba(0,0,0,0.13);
    }


    .feature-card:hover {
        border-color: #2D5C77;

        background: #0E1B2C;
    }


    .feature-icon {
        font-size: 18px;

        margin-bottom: 9px;
    }


    .feature-title {
        color: #E8F0F8 !important;

        font-size: 0.91rem;

        font-weight: 800;

        margin-bottom: 7px;
    }


    .feature-text {
        color: #8999AC !important;

        font-size: 0.78rem;

        line-height: 1.6;
    }


    /* ========================================================
       SECTION HEADER
       ======================================================== */

    .section-header {
        display: flex;

        align-items: center;

        justify-content: space-between;

        margin: 25px 0 9px 0;
    }


    .section-title {
        color: #DCE6F2 !important;

        font-size: 0.74rem;

        font-weight: 800;

        letter-spacing: 1.2px;

        text-transform: uppercase;
    }


    .section-status {
        color: #66788D !important;

        font-size: 0.67rem;
    }


    /* ========================================================
       NORMAL BUTTONS
       ======================================================== */

    .stButton > button {
        background: #0E1A2A !important;

        color: #D8E3EE !important;

        border: 1px solid #263A50 !important;

        border-radius: 8px !important;

        min-height: 42px !important;

        font-size: 0.8rem !important;

        font-weight: 700 !important;

        box-shadow: none !important;
    }


    .stButton > button:hover {
        background: #13263A !important;

        color: #FFFFFF !important;

        border-color: #3282A8 !important;
    }


    .stButton > button p {
        color: inherit !important;
    }


    /* ========================================================
       CHAT MESSAGES
       ======================================================== */

    [data-testid="stChatMessage"] {
        background: #0C1726 !important;

        border: 1px solid #21354A !important;

        border-radius: 12px !important;

        padding: 13px 16px !important;

        margin-bottom: 10px !important;
    }


    [data-testid="stChatMessage"] p {
        color: #E6EDF4 !important;

        line-height: 1.65 !important;
    }


    [data-testid="stChatMessage"] strong {
        color: #FFFFFF !important;
    }


    /* ========================================================
       BOTTOM CHAT AREA
       ======================================================== */

    .stBottom {
        background: #07111F !important;

        background-color: #07111F !important;

        border-top: 1px solid #17283B !important;
    }


    [data-testid="stBottom"] {
        background: #07111F !important;

        background-color: #07111F !important;
    }


    [data-testid="stBottomBlockContainer"] {
        background: #07111F !important;

        background-color: #07111F !important;

        padding-top: 9px !important;

        padding-bottom: 12px !important;
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    [data-testid="stChatInput"] {
        background: #0D1929 !important;

        background-color: #0D1929 !important;

        border: 1px solid #294158 !important;

        border-radius: 12px !important;

        padding: 6px !important;

        box-shadow:
            0 8px 30px rgba(0,0,0,0.35) !important;
    }


    [data-testid="stChatInput"] > div {
        background: #0D1929 !important;

        background-color: #0D1929 !important;
    }


    [data-testid="stChatInput"] textarea {
        background: #0D1929 !important;

        background-color: #0D1929 !important;

        color: #F8FAFC !important;

        -webkit-text-fill-color: #F8FAFC !important;

        caret-color: #67E8F9 !important;

        border: none !important;

        outline: none !important;

        font-size: 0.9rem !important;
    }


    [data-testid="stChatInput"] textarea::placeholder {
        color: #718198 !important;

        -webkit-text-fill-color: #718198 !important;

        opacity: 1 !important;
    }


    [data-testid="stChatInput"] button {
        background: #0EA5E9 !important;

        background-color: #0EA5E9 !important;

        color: #FFFFFF !important;

        border: none !important;

        border-radius: 8px !important;
    }


    [data-testid="stChatInput"] button:hover {
        background: #0284C7 !important;

        background-color: #0284C7 !important;
    }


    /* ========================================================
       EXPANDER
       ======================================================== */

    [data-testid="stExpander"] {
        background: #0A1422 !important;

        border: 1px solid #20344A !important;

        border-radius: 9px !important;
    }


    [data-testid="stExpander"] summary {
        color: #CBD5E1 !important;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        border-radius: 9px !important;
    }


    /* ========================================================
       CODE
       ======================================================== */

    .stCodeBlock {
        background: #070D16 !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;

        padding: 30px 0 15px;

        color: #56677C !important;

        font-size: 0.7rem;

        line-height: 1.8;
    }


    .footer strong {
        color: #7890A8 !important;
    }


    /* ========================================================
       SCROLLBAR
       ======================================================== */

    ::-webkit-scrollbar {
        width: 7px;
    }


    ::-webkit-scrollbar-track {
        background: #07111F;
    }


    ::-webkit-scrollbar-thumb {
        background: #263A50;

        border-radius: 10px;
    }


    ::-webkit-scrollbar-thumb:hover {
        background: #315B73;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        .hero {
            padding: 26px 22px;
        }

        .hero-title {
            font-size: 1.9rem;
        }

        .online {
            display: none;
        }

        .topbar {
            min-height: 55px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOOLS
# ============================================================

@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


search_tool = DuckDuckGoSearchRun()

tools = [
    search_tool,
    add,
    multiply,
]


# ============================================================
# BUILD AGENT
# ============================================================

@st.cache_resource(show_spinner=False)
def build_agent():

    api_key = (
        st.secrets.get("GROQ_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
        max_tokens=2048,
    )

    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):

        system_message = """
You are a professional Agentic AI Assistant.

You are powered by OpenAI GPT-OSS 120B through Groq.

Answer users clearly, accurately and professionally.

Use web search when:
- Current information is requested.
- The user explicitly asks for web search.
- The question requires online verification.

Use addition and multiplication tools for arithmetic
when appropriate.

Never claim that you searched the web unless the
search tool was actually used.

Keep responses useful, structured and easy to understand.
"""

        response = llm_with_tools.invoke(
            [
                ("system", system_message)
            ] + state["messages"]
        )

        return {
            "messages": [response]
        }


    builder = StateGraph(MessagesState)

    builder.add_node(
        "assistant",
        assistant,
    )

    builder.add_node(
        "tools",
        ToolNode(tools),
    )

    builder.add_edge(
        START,
        "assistant",
    )

    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )

    builder.add_edge(
        "tools",
        "assistant",
    )

    return builder.compile()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # Brand
    st.html(
        """
        <div class="side-brand">

            <div class="side-logo">
                🤖
            </div>

            <div>
                <div class="side-title">
                    AI Agent
                </div>

                <div class="side-subtitle">
                    INTELLIGENT WORKSPACE
                </div>
            </div>

        </div>
        """
    )


    st.divider()


    # Workflow
    st.html(
        """
        <div class="side-section">
            Workflow
        </div>
        """
    )


    workflow = [
        ("01", "User sends a request"),
        ("02", "GPT-OSS 120B analyzes it"),
        ("03", "LangGraph selects a tool"),
        ("04", "Tool executes the task"),
        ("05", "AI returns the result"),
    ]


    for number, item in workflow:

        st.html(
            f"""
            <div class="workflow">

                <div class="workflow-num">
                    {number}
                </div>

                <div class="workflow-text">
                    {item}
                </div>

            </div>
            """
        )


    st.divider()


    # Tools
    st.html(
        """
        <div class="side-section">
            Tools
        </div>
        """
    )


    st.html(
        """
        <div class="tool-box">
            🔎 Web Search
        </div>

        <div class="tool-box">
            ➕ Addition
        </div>

        <div class="tool-box">
            ✖️ Multiplication
        </div>
        """
    )


    st.divider()


    # Model
    st.html(
        """
        <div class="side-section">
            Model
        </div>

        <div class="model-card">

            <div class="model-label">
                Groq Inference
            </div>

            <div class="model-value">
                openai/gpt-oss-120b
            </div>

        </div>
        """
    )


    st.divider()


    # Security
    st.html(
        """
        <div class="side-section">
            Security
        </div>

        <div class="security-card">
            🔐 API credentials are loaded securely
            through Streamlit Secrets.
            Never expose your API key in GitHub.
        </div>
        """
    )


    st.write("")


    if st.button(
        "🗑️  Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# TOP NAVBAR
# ============================================================

st.html(
    """
    <div class="topbar">

        <div class="top-left">

            <div class="top-logo">
                🤖
            </div>

            <div>
                <div class="top-title">
                    AI Agent Dashboard
                </div>

                <div class="top-subtitle">
                    Agentic AI • LangGraph • Groq
                </div>
            </div>

        </div>

        <div class="online">

            <div class="online-dot"></div>

            SYSTEM ONLINE

        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-badge">
            ✦ AGENTIC AI • LANGGRAPH
        </div>

        <div class="hero-title">
            AI Agent <span>Workspace</span>
        </div>

        <div class="hero-description">
            A production-style conversational AI workspace
            powered by GPT-OSS 120B through Groq. Ask questions,
            search the web and perform calculations through an
            intelligent LangGraph agent workflow.
        </div>

    </div>
    """
)


# ============================================================
# FEATURE CARDS
# ============================================================

c1, c2, c3 = st.columns(
    3,
    gap="medium",
)


with c1:

    st.html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                🧠
            </div>

            <div class="feature-title">
                Agentic Reasoning
            </div>

            <div class="feature-text">
                The agent analyzes each request and decides
                whether a tool is required before responding.
            </div>

        </div>
        """
    )


with c2:

    st.html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                🔧
            </div>

            <div class="feature-title">
                Tool Orchestration
            </div>

            <div class="feature-text">
                LangGraph manages the workflow between the
                language model and external tools.
            </div>

        </div>
        """
    )


with c3:

    st.html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                ⚡
            </div>

            <div class="feature-title">
                Fast Inference
            </div>

            <div class="feature-text">
                GPT-OSS 120B is served through Groq for fast,
                responsive agentic AI interactions.
            </div>

        </div>
        """
    )


# ============================================================
# QUICK ACTION HEADER
# ============================================================

st.html(
    """
    <div class="section-header">

        <div class="section-title">
            Quick Actions
        </div>

        <div class="section-status">
            Select an example to start
        </div>

    </div>
    """
)


# ============================================================
# QUICK ACTION BUTTONS
# ============================================================

b1, b2, b3 = st.columns(
    3,
    gap="medium",
)


examples = [
    (
        "🔎  Search the Web",
        "Search the web for the latest developments in AI agents.",
    ),
    (
        "🧮  Calculate",
        "Multiply 125 by 24 and add 500.",
    ),
    (
        "📘  Learn LangGraph",
        "Explain LangGraph in simple terms with an example.",
    ),
]


for col, (label, question) in zip(
    (b1, b2, b3),
    examples,
):

    with col:

        if st.button(
            label,
            use_container_width=True,
        ):

            st.session_state.pending_prompt = question

            st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if isinstance(
        message,
        HumanMessage,
    ):

        with st.chat_message("user"):

            st.markdown(
                message.content
            )


    elif isinstance(
        message,
        AIMessage,
    ):

        if message.content:

            with st.chat_message("assistant"):

                st.markdown(
                    message.content
                )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask your AI agent anything..."
)


if "pending_prompt" in st.session_state:

    prompt = st.session_state.pop(
        "pending_prompt"
    )


# ============================================================
# AGENT EXECUTION
# ============================================================

if prompt:

    user_message = HumanMessage(
        content=prompt
    )

    st.session_state.messages.append(
        user_message
    )


    with st.chat_message("user"):

        st.markdown(prompt)


    try:

        with st.chat_message("assistant"):

            with st.spinner(
                "Agent is processing your request..."
            ):

                agent = build_agent()


                result = agent.invoke(
                    {
                        "messages":
                        st.session_state.messages
                    }
                )


                all_messages = result["messages"]


                previous_count = len(
                    st.session_state.messages
                )


                generated = all_messages[
                    previous_count:
                ]


                st.session_state.messages.extend(
                    generated
                )


                # ====================================================
                # TOOL ACTIVITY
                # ====================================================

                tool_messages = [
                    m
                    for m in generated
                    if isinstance(
                        m,
                        ToolMessage
                    )
                ]


                if tool_messages:

                    with st.expander(
                        "🔧 Tool activity",
                        expanded=False,
                    ):

                        for tool_message in tool_messages:

                            st.markdown(
                                f"**{tool_message.name}**"
                            )

                            st.code(
                                str(
                                    tool_message.content
                                )[:4000]
                            )


                # ====================================================
                # FINAL RESPONSE
                # ====================================================

                final_response = ""


                for message in reversed(
                    generated
                ):

                    if (
                        isinstance(
                            message,
                            AIMessage
                        )
                        and message.content
                    ):

                        final_response = (
                            message.content
                        )

                        break


                if final_response:

                    st.markdown(
                        final_response
                    )

                else:

                    st.warning(
                        "No final response was generated. "
                        "Please try again."
                    )


    except Exception as error:

        st.error(
            "The AI agent could not process this request."
        )


        with st.expander(
            "Technical details",
        ):

            st.code(
                str(error)
            )


        if (
            st.session_state.messages
            and
            st.session_state.messages[-1]
            == user_message
        ):

            st.session_state.messages.pop()


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        <strong>AI Agent Dashboard</strong>
        &nbsp; • &nbsp;
        LangGraph
        &nbsp; • &nbsp;
        LangChain
        &nbsp; • &nbsp;
        Groq
        &nbsp; • &nbsp;
        GPT-OSS 120B

        <br>

        Intelligent tool-calling workspace for Agentic AI.

    </div>
    """
)
