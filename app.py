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
# PROFESSIONAL DARK ENTERPRISE THEME
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   GLOBAL
   ============================================================ */

html,
body,
.stApp,
[data-testid="stAppViewContainer"] {
    background: #07111F !important;
    color: #E8F0F7 !important;
}

.main {
    background: #07111F !important;
}

.block-container {
    max-width: 1450px !important;
    padding-top: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 7rem !important;
}


/* ============================================================
   GLOBAL TEXT
   ============================================================ */

.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6,
.stApp .stMarkdown {
    color: #E8F0F7 !important;
}


/* ============================================================
   STREAMLIT HEADER
   ============================================================ */

[data-testid="stHeader"] {
    background: #07111F !important;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background: #091522 !important;
    border-right: 1px solid #1B3045 !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: #091522 !important;
    padding: 24px 18px !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] li {
    color: #B9C8D8 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] hr {
    border-color: #1B3045 !important;
    margin: 22px 0 !important;
}


/* ============================================================
   SIDEBAR BRAND
   ============================================================ */

.side-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
}

.side-logo {
    width: 44px;
    height: 44px;
    border-radius: 11px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #0C2638;
    border: 1px solid #1B668A;

    color: #55D6FF !important;
    font-size: 21px;
}

.side-title {
    color: #FFFFFF !important;
    font-size: 1.08rem;
    font-weight: 800;
}

.side-subtitle {
    color: #5F748A !important;
    font-size: 0.63rem;
    letter-spacing: 1.3px;
    font-weight: 700;
    margin-top: 2px;
}


/* ============================================================
   SIDEBAR SECTION
   ============================================================ */

.side-section {
    color: #55C7F5 !important;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 11px;
}


/* ============================================================
   WORKFLOW
   ============================================================ */

.workflow {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 11px 0;
}

.workflow-num {
    min-width: 26px;
    height: 26px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #0C1C2B;
    border: 1px solid #244762;
    border-radius: 6px;

    color: #61D5FF !important;
    font-size: 0.67rem;
    font-weight: 800;
}

.workflow-text {
    color: #B7C5D5 !important;
    font-size: 0.78rem;
    line-height: 1.5;
    padding-top: 3px;
}


/* ============================================================
   TOOL CARDS
   ============================================================ */

.tool-box {
    display: flex;
    align-items: center;

    padding: 10px 12px;
    margin-bottom: 7px;

    background: #0C1927;
    border: 1px solid #1D344A;
    border-radius: 8px;

    color: #C7D4E1 !important;
    font-size: 0.78rem;
}

.tool-box:hover {
    border-color: #287A9F;
    background: #0D1E2D;
}


/* ============================================================
   MODEL CARD
   ============================================================ */

.model-card {
    background: #0C1927;
    border: 1px solid #24506A;
    border-radius: 10px;
    padding: 13px;
}

.model-label {
    color: #62788E !important;
    font-size: 0.63rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 800;
    margin-bottom: 6px;
}

.model-value {
    color: #65D8FF !important;
    font-family: monospace;
    font-size: 0.75rem;
    font-weight: 700;
}


/* ============================================================
   SECURITY
   ============================================================ */

.security-card {
    background: #091D1D;
    border: 1px solid #1C4844;
    border-radius: 9px;
    padding: 12px;

    color: #9FC2BD !important;
    font-size: 0.72rem;
    line-height: 1.55;
}


/* ============================================================
   SIDEBAR BUTTON
   ============================================================ */

[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;

    background: #0D1A29 !important;
    color: #CBD7E3 !important;

    border: 1px solid #294056 !important;
    border-radius: 9px !important;

    min-height: 42px !important;

    font-weight: 700 !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #102538 !important;
    color: #FFFFFF !important;
    border-color: #38BDF8 !important;
}


/* ============================================================
   TOP BAR
   ============================================================ */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 0 3px 17px 3px;

    border-bottom: 1px solid #172B3E;
    margin-bottom: 23px;
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

    background: #0C2638;
    border: 1px solid #216482;
    border-radius: 9px;

    font-size: 19px;
}

.top-title {
    color: #F8FAFC !important;
    font-size: 1rem;
    font-weight: 800;
}

.top-subtitle {
    color: #647A90 !important;
    font-size: 0.66rem;
    margin-top: 2px;
}

.online {
    display: flex;
    align-items: center;
    gap: 7px;

    padding: 6px 11px;

    background: #09211A;
    border: 1px solid #1A493C;
    border-radius: 20px;

    color: #6EE7B7 !important;
    font-size: 0.67rem;
    font-weight: 800;
}

.online-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;

    background: #34D399;
    box-shadow: 0 0 8px #34D399;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    background:
        linear-gradient(
            120deg,
            #0C1B2C 0%,
            #102B3D 55%,
            #0C3442 100%
        );

    border: 1px solid #28546B;
    border-radius: 16px;

    padding: 34px 38px;
    margin-bottom: 23px;

    box-shadow:
        0 15px 45px rgba(0, 0, 0, 0.22);
}

.hero-badge {
    display: inline-block;

    padding: 6px 12px;

    background: #0C2A3C;
    border: 1px solid #26617E;
    border-radius: 6px;

    color: #72D8FF !important;

    font-size: 0.66rem;
    font-weight: 800;
    letter-spacing: 1.2px;

    margin-bottom: 13px;
}

.hero-title {
    color: #FFFFFF !important;

    font-size: 2.4rem;
    font-weight: 850;

    line-height: 1.15;

    margin-bottom: 11px;
}

.hero-title span {
    color: #56D4FF !important;
}

.hero-description {
    color: #B8C9D9 !important;

    font-size: 0.95rem;
    line-height: 1.7;

    max-width: 900px;
}


/* ============================================================
   FEATURE CARDS
   ============================================================ */

.feature-card {
    background: #0C1826;

    border: 1px solid #20384D;
    border-radius: 13px;

    padding: 20px;

    min-height: 145px;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.14);
}

.feature-card:hover {
    border-color: #2E6B88;
    transform: translateY(-1px);
}

.feature-icon {
    font-size: 20px;
    margin-bottom: 9px;
}

.feature-title {
    color: #EAF3FA !important;

    font-size: 0.92rem;
    font-weight: 800;

    margin-bottom: 8px;
}

.feature-text {
    color: #91A5B8 !important;

    font-size: 0.78rem;
    line-height: 1.65;
}


/* ============================================================
   SECTION
   ============================================================ */

.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;

    margin: 26px 0 10px 0;
}

.section-title {
    color: #DCE8F2 !important;

    font-size: 0.75rem;
    font-weight: 800;

    letter-spacing: 1.3px;
    text-transform: uppercase;
}

.section-status {
    color: #60758A !important;
    font-size: 0.68rem;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    background: #0D1B2A !important;

    color: #DCE7F0 !important;

    border: 1px solid #284157 !important;
    border-radius: 9px !important;

    min-height: 43px !important;

    font-size: 0.8rem !important;
    font-weight: 700 !important;

    box-shadow: none !important;
}

.stButton > button:hover {
    background: #102638 !important;

    color: #FFFFFF !important;

    border-color: #38BDF8 !important;
}

.stButton > button p {
    color: inherit !important;
}


/* ============================================================
   CHAT MESSAGES
   ============================================================ */

[data-testid="stChatMessage"] {
    background: #0C1826 !important;

    border: 1px solid #20384D !important;

    border-radius: 13px !important;

    padding: 15px 18px !important;

    margin-bottom: 11px !important;
}


/* IMPORTANT: AI RESPONSE TEXT */

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em {
    color: #EAF2F8 !important;
}

[data-testid="stChatMessage"] p {
    line-height: 1.7 !important;
}

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: #FFFFFF !important;
}


/* ============================================================
   CHAT INPUT
   ============================================================ */

.stBottom {
    background: #07111F !important;
    background-color: #07111F !important;

    border-top: 1px solid #172B3E !important;
}

[data-testid="stBottom"] {
    background: #07111F !important;
    background-color: #07111F !important;
}

[data-testid="stBottomBlockContainer"] {
    background: #07111F !important;
    background-color: #07111F !important;

    padding-top: 10px !important;
    padding-bottom: 12px !important;
}

[data-testid="stChatInput"] {
    background: #0C1826 !important;
    background-color: #0C1826 !important;

    border: 1px solid #29465D !important;

    border-radius: 13px !important;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.30) !important;
}

[data-testid="stChatInput"] > div {
    background: #0C1826 !important;
    background-color: #0C1826 !important;
}

[data-testid="stChatInput"] textarea {
    background: #0C1826 !important;
    background-color: #0C1826 !important;

    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;

    caret-color: #5DD9FF !important;

    border: none !important;
    outline: none !important;

    font-size: 0.91rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #72879B !important;
    -webkit-text-fill-color: #72879B !important;
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


/* ============================================================
   EXPANDER
   ============================================================ */

[data-testid="stExpander"] {
    background: #0A1522 !important;

    border: 1px solid #20384D !important;

    border-radius: 10px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: #D6E2EC !important;
}


/* ============================================================
   ALERTS
   ============================================================ */

[data-testid="stAlert"] {
    border-radius: 10px !important;
}


/* ============================================================
   CODE
   ============================================================ */

[data-testid="stCode"] {
    background: #060D16 !important;

    border: 1px solid #203447 !important;

    border-radius: 9px !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    padding: 30px 0 15px;

    color: #53687B !important;

    font-size: 0.7rem;

    line-height: 1.8;
}

.footer strong {
    color: #71879A !important;
}


/* ============================================================
   SCROLLBAR
   ============================================================ */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #07111F;
}

::-webkit-scrollbar-thumb {
    background: #263D51;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #35627A;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .hero {
        padding: 27px 22px;
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
# BUILD LANGGRAPH AGENT
# ============================================================

@st.cache_resource(show_spinner=False)
def build_agent():

    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Please add GROQ_API_KEY in Streamlit Secrets."
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

Your job is to provide accurate, clear and useful answers.

TOOL RULES:

1. Use web search when:
   - The user asks for current information.
   - The user explicitly asks you to search the web.
   - Online verification is required.

2. Use the addition tool for addition calculations.

3. Use the multiplication tool for multiplication calculations.

4. Do not claim that you searched the web unless
   the web search tool was actually used.

5. Give the final answer in a clean and professional format.

6. Use headings, bullet points and examples when useful.

7. If the user asks a simple question, answer directly.

8. Do not unnecessarily explain internal agent processes.
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
# SESSION STATE
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
            <div class="side-logo">🤖</div>

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
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="side-section">Workflow</div>',
        unsafe_allow_html=True,
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
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        '<div class="side-section">Available Tools</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tool-box">
            🔎 &nbsp; Web Search
        </div>

        <div class="tool-box">
            ➕ &nbsp; Addition
        </div>

        <div class="tool-box">
            ✖️ &nbsp; Multiplication
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="side-section">Model</div>',
        unsafe_allow_html=True,
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
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="side-section">Security</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="security-card">
            🔐 API credentials are loaded securely
            through Streamlit Secrets.
            Never expose your API key in GitHub.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if st.button(
        "🗑️  Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# TOP NAVIGATION
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
    unsafe_allow_html=True,
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
            A professional conversational AI workspace powered
            by GPT-OSS 120B through Groq. Ask questions, search
            the web and perform calculations through an intelligent
            LangGraph agent workflow.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FEATURE CARDS
# ============================================================

c1, c2, c3 = st.columns(
    3,
    gap="medium",
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
                whether an external tool is required before
                generating the final response.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
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
                LangGraph manages communication between
                GPT-OSS 120B and the available tools.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with c3:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">
                ⚡
            </div>

            <div class="feature-title">
                Fast AI Inference
            </div>

            <div class="feature-text">
                GPT-OSS 120B runs through Groq infrastructure
                for fast and responsive agentic AI interactions.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# QUICK ACTIONS
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
    unsafe_allow_html=True,
)


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

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):

            st.markdown(
                message.content
            )

    elif isinstance(message, AIMessage):

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

        st.markdown(
            prompt
        )

    try:

        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 Agent is processing your request..."
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

                # ------------------------------------------------
                # TOOL ACTIVITY
                # ------------------------------------------------

                tool_messages = [
                    message
                    for message in generated
                    if isinstance(
                        message,
                        ToolMessage,
                    )
                ]

                if tool_messages:

                    with st.expander(
                        "🔧 Tool Activity",
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

                # ------------------------------------------------
                # FINAL AI RESPONSE
                # ------------------------------------------------

                final_response = ""

                for message in reversed(
                    generated
                ):

                    if (
                        isinstance(
                            message,
                            AIMessage,
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
                        "The agent completed the workflow "
                        "but did not return a final response. "
                        "Please try again."
                    )

    except Exception as error:

        st.error(
            "The AI agent could not process this request."
        )

        with st.expander(
            "Technical Details"
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
    unsafe_allow_html=True,
)
