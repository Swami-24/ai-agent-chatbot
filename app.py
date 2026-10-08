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
       ROOT / GLOBAL
       ======================================================== */

    :root {
        --bg: #080D16;
        --panel: #0D1522;
        --panel-2: #111B2A;
        --border: #233147;
        --border-light: #2C3B52;

        --text: #F1F5F9;
        --text-soft: #B6C2D2;
        --text-muted: #718096;

        --blue: #38BDF8;
        --blue-dark: #0EA5E9;
        --green: #34D399;
        --orange: #F59E0B;
    }

    /* ========================================================
       BODY
       ======================================================== */

    html,
    body,
    .stApp {
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    .stApp {
        min-height: 100vh !important;
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

    [data-testid="stSidebar"] {
        background: #0A111D !important;

        border-right: 1px solid #1D2A3C !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #0A111D !important;

        padding: 22px 16px !important;
    }

    [data-testid="stSidebar"] p {
        color: #AEBBCB !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #F8FAFC !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #1D2A3C !important;

        margin: 22px 0 !important;
    }

    /* Sidebar brand */

    .side-brand {
        display: flex;

        align-items: center;

        gap: 11px;

        margin-bottom: 8px;
    }

    .side-logo {
        width: 42px;
        height: 42px;

        display: flex;

        align-items: center;
        justify-content: center;

        border-radius: 10px;

        background: #12304A;

        border: 1px solid #1C5577;

        color: #67E8F9 !important;

        font-size: 20px;
    }

    .side-title {
        color: #FFFFFF !important;

        font-size: 1.05rem;

        font-weight: 800;
    }

    .side-subtitle {
        color: #64748B !important;

        font-size: 0.66rem;

        letter-spacing: 1.2px;

        font-weight: 700;
    }

    /* Sidebar section */

    .side-section {
        color: #7DD3FC !important;

        font-size: 0.69rem;

        font-weight: 800;

        letter-spacing: 1.3px;

        text-transform: uppercase;

        margin-bottom: 10px;
    }

    /* Workflow */

    .workflow {
        display: flex;

        align-items: flex-start;

        gap: 10px;

        margin: 10px 0;
    }

    .workflow-num {
        min-width: 24px;

        height: 24px;

        display: flex;

        align-items: center;

        justify-content: center;

        background: #101D2D;

        border: 1px solid #2A4058;

        border-radius: 6px;

        color: #67E8F9 !important;

        font-size: 0.67rem;

        font-weight: 800;
    }

    .workflow-text {
        color: #B6C2D2 !important;

        font-size: 0.79rem;

        line-height: 1.5;
    }

    /* Tools */

    .tool-box {
        display: flex;

        align-items: center;

        gap: 9px;

        padding: 9px 11px;

        margin-bottom: 7px;

        background: #0E1826;

        border: 1px solid #1F2E43;

        border-radius: 8px;

        color: #C7D2E0 !important;

        font-size: 0.78rem;
    }

    /* Model */

    .model-card {
        background: #0E1826;

        border: 1px solid #27415B;

        border-radius: 10px;

        padding: 12px;
    }

    .model-label {
        color: #64748B !important;

        font-size: 0.65rem;

        text-transform: uppercase;

        letter-spacing: 1px;

        font-weight: 800;

        margin-bottom: 5px;
    }

    .model-value {
        color: #7DD3FC !important;

        font-family: monospace;

        font-size: 0.76rem;

        font-weight: 700;
    }

    /* Security */

    .security-card {
        background: #0B1718;

        border: 1px solid #21403E;

        border-radius: 9px;

        padding: 11px;

        color: #9CB8B5 !important;

        font-size: 0.72rem;

        line-height: 1.55;
    }

    /* ========================================================
       SIDEBAR BUTTON
       ======================================================== */

    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;

        background: #111B29 !important;

        color: #CBD5E1 !important;

        border: 1px solid #2A394E !important;

        border-radius: 9px !important;

        min-height: 40px !important;

        font-size: 0.8rem !important;

        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #172335 !important;

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
        height: 62px;

        display: flex;

        align-items: center;

        justify-content: space-between;

        padding: 0 4px 15px 4px;

        border-bottom: 1px solid #182538;

        margin-bottom: 22px;
    }

    .top-left {
        display: flex;

        align-items: center;

        gap: 12px;
    }

    .top-logo {
        width: 39px;
        height: 39px;

        display: flex;

        align-items: center;

        justify-content: center;

        background: #10283A;

        border: 1px solid #1E5978;

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
        color: #66758A !important;

        font-size: 0.68rem;

        margin-top: 2px;
    }

    .online {
        display: flex;

        align-items: center;

        gap: 7px;

        padding: 6px 10px;

        background: #0B1D18;

        border: 1px solid #1C4639;

        border-radius: 20px;

        color: #6EE7B7 !important;

        font-size: 0.68rem;

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
                #101C30 0%,
                #12283C 55%,
                #102C3B 100%
            );

        border: 1px solid #27445A;

        border-radius: 15px;

        padding: 34px 38px;

        margin-bottom: 22px;

        box-shadow:
            0 15px 45px rgba(0,0,0,0.20);
    }

    .hero-badge {
        display: inline-block;

        padding: 6px 11px;

        border-radius: 6px;

        background: #102D40;

        border: 1px solid #24536D;

        color: #7DD3FC !important;

        font-size: 0.67rem;

        font-weight: 800;

        letter-spacing: 1.1px;

        margin-bottom: 13px;
    }

    .hero-title {
        color: #FFFFFF !important;

        font-size: 2.35rem;

        font-weight: 850;

        line-height: 1.15;

        margin-bottom: 10px;
    }

    .hero-title span {
        color: #67E8F9 !important;
    }

    .hero-description {
        color: #B9C7D8 !important;

        font-size: 0.94rem;

        max-width: 900px;

        line-height: 1.7;
    }

    /* ========================================================
       METRIC / FEATURE CARDS
       ======================================================== */

    .feature-card {
        background: #0D1624;

        border: 1px solid #1F3045;

        border-radius: 12px;

        padding: 19px;

        min-height: 145px;

        box-shadow:
            0 8px 25px rgba(0,0,0,0.12);
    }

    .feature-card:hover {
        border-color: #2E5A73;
    }

    .feature-icon {
        font-size: 18px;

        margin-bottom: 9px;
    }

    .feature-title {
        color: #E5EDF6 !important;

        font-size: 0.91rem;

        font-weight: 800;

        margin-bottom: 7px;
    }

    .feature-text {
        color: #8392A7 !important;

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

        margin: 24px 0 9px 0;
    }

    .section-title {
        color: #DCE6F2 !important;

        font-size: 0.75rem;

        font-weight: 800;

        letter-spacing: 1.2px;

        text-transform: uppercase;
    }

    .section-status {
        color: #64748B !important;

        font-size: 0.68rem;
    }

    /* ========================================================
       NORMAL BUTTONS
       ======================================================== */

    .stButton > button {
        background: #101A29 !important;

        color: #D8E2EE !important;

        border: 1px solid #26384F !important;

        border-radius: 8px !important;

        min-height: 42px !important;

        font-size: 0.8rem !important;

        font-weight: 700 !important;

        box-shadow: none !important;
    }

    .stButton > button:hover {
        background: #142438 !important;

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
        background: #0D1624 !important;

        border: 1px solid #213148 !important;

        border-radius: 12px !important;

        padding: 13px 16px !important;

        margin-bottom: 10px !important;
    }

    [data-testid="stChatMessage"] p {
        color: #E4EBF3 !important;

        line-height: 1.65 !important;
    }

    [data-testid="stChatMessage"] strong {
        color: #FFFFFF !important;
    }

    /* ========================================================
       CRITICAL: STREAMLIT BOTTOM AREA
       ======================================================== */

    .stBottom {
        background: #080D16 !important;

        background-color: #080D16 !important;

        border-top: 1px solid #182538 !important;
    }

    [data-testid="stBottom"] {
        background: #080D16 !important;

        background-color: #080D16 !important;
    }

    [data-testid="stBottomBlockContainer"] {
        background: #080D16 !important;

        background-color: #080D16 !important;

        padding-top: 10px !important;

        padding-bottom: 12px !important;
    }

    /* ========================================================
       CHAT INPUT CONTAINER
       ======================================================== */

    [data-testid="stChatInput"] {
        background: #0D1624 !important;

        background-color: #0D1624 !important;

        border: 1px solid #2A3B52 !important;

        border-radius: 12px !important;

        padding: 6px !important;

        box-shadow:
            0 8px 30px rgba(0,0,0,0.30) !important;
    }

    [data-testid="stChatInput"] > div {
        background: #0D1624 !important;

        background-color: #0D1624 !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #0D1624 !important;

        background-color: #0D1624 !important;

        color: #F8FAFC !important;

        -webkit-text-fill-color: #F8FAFC !important;

        caret-color: #67E8F9 !important;

        border: none !important;

        outline: none !important;

        font-size: 0.9rem !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #66758A !important;

        -webkit-text-fill-color: #66758A !important;

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
        background: #0B1320 !important;

        border: 1px solid #203047 !important;

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
        background: #080E18 !important;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;

        padding: 30px 0 15px;

        color: #526176 !important;

        font-size: 0.7rem;

        line-height: 1.8;
    }

    .footer strong {
        color: #718096 !important;
    }

    /* ========================================================
       SCROLLBAR
       ======================================================== */

    ::-webkit-scrollbar {
        width: 7px;
    }

    ::-webkit-scrollbar-track {
        background: #080D16;
    }

    ::-webkit-scrollbar-thumb {
        background: #26364B;

        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #315A72;
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
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


search_tool = DuckDuckGoSearchRun()

tools = [
    search_tool,
    add,
    multiply
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
        assistant
    )

    builder.add_node(
        "tools",
        ToolNode(tools)
    )

    builder.add_edge(
        START,
        "assistant"
    )

    builder.add_conditional_edges(
        "assistant",
        tools_condition
    )

    builder.add_edge(
        "tools",
        "assistant"
    )

    return builder.compile()


# ============================================================
# SESSION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
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
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="side-section">Workflow</div>',
        unsafe_allow_html=True
    )

    workflow = [
        ("01", "User sends a request"),
        ("02", "GPT-OSS 120B analyzes it"),
        ("03", "LangGraph selects a tool"),
        ("04", "Tool executes the task"),
        ("05", "AI returns the result"),
    ]

    for number, item in workflow:

        st.markdown(
            f"""
            <div class="workflow">

                <div class="workflow-num">
                    {number}
                </div>

                <div class="workflow-text">
                    {item}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown(
        '<div class="side-section">Tools</div>',
        unsafe_allow_html=True
    )

    st.markdown(
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
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="side-section">Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="model-card">

            <div class="model-label">
                Groq Inference
            </div>

            <div class="model-value">
                openai/gpt-oss-120b
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="side-section">Security</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="security-card">
            🔐 API credentials are loaded securely
            through Streamlit Secrets.
            Never expose your API key in GitHub.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button(
        "🗑️  Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
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
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
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
    """,
    unsafe_allow_html=True
)


# ============================================================
# FEATURE CARDS
# ============================================================

c1, c2, c3 = st.columns(
    3,
    gap="medium"
)


with c1:

    st.markdown(
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
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
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
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
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
        """,
        unsafe_allow_html=True
    )


# ============================================================
# EXAMPLES
# ============================================================

st.markdown(
    """
    <div class="section-header">

        <div class="section-title">
            Quick Actions
        </div>

        <div class="section-status">
            Select an example to start
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


b1, b2, b3 = st.columns(
    3,
    gap="medium"
)


examples = [
    (
        "🔎  Search the Web",
        "Search the web for the latest developments in AI agents."
    ),
    (
        "🧮  Calculate",
        "Multiply 125 by 24 and add 500."
    ),
    (
        "📘  Learn LangGraph",
        "Explain LangGraph in simple terms with an example."
    ),
]


for col, (label, question) in zip(
    (b1, b2, b3),
    examples
):

    with col:

        if st.button(
            label,
            use_container_width=True
        ):

            st.session_state.pending_prompt = question

            st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if isinstance(
        message,
        HumanMessage
    ):

        with st.chat_message("user"):

            st.markdown(
                message.content
            )

    elif isinstance(
        message,
        AIMessage
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

                # Tool activity

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
                        expanded=False
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

                # Final response

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
            "Technical details"
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

st.markdown(
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
    """,
    unsafe_allow_html=True
)
