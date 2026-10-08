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
# PROFESSIONAL DARK UI
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

html, body, [class*="css"] {
    font-family: Inter, Arial, sans-serif !important;
}

.stApp {
    background: #07111F !important;
    color: #F8FAFC !important;
}

.main {
    background: #07111F !important;
}

.block-container {
    max-width: 1450px !important;
    padding-top: 25px !important;
    padding-left: 35px !important;
    padding-right: 35px !important;
    padding-bottom: 130px !important;
}


/* =========================================================
   ALL MARKDOWN TEXT
   ========================================================= */

[data-testid="stMarkdownContainer"] {
    color: #E8EEF7 !important;
}

[data-testid="stMarkdownContainer"] p {
    color: #D5DFEC !important;
}

[data-testid="stMarkdownContainer"] li {
    color: #D5DFEC !important;
}

[data-testid="stMarkdownContainer"] strong {
    color: #FFFFFF !important;
}

[data-testid="stMarkdownContainer"] em {
    color: #D5DFEC !important;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5 {
    color: #F8FAFC !important;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background: #091522 !important;
    border-right: 1px solid #1B3046 !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: #091522 !important;
    padding: 25px 18px !important;
}

[data-testid="stSidebar"] p {
    color: #AEBED0 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] hr {
    border-color: #1C3045 !important;
    margin: 22px 0 !important;
}


/* =========================================================
   SIDEBAR BRAND
   ========================================================= */

.side-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 5px;
}

.side-logo {
    width: 44px;
    height: 44px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 11px;

    background: #0C2A42;
    border: 1px solid #15567D;

    font-size: 21px;
}

.side-title {
    color: #FFFFFF !important;
    font-size: 1.08rem;
    font-weight: 800;
}

.side-subtitle {
    color: #6F849A !important;
    font-size: 0.63rem;
    letter-spacing: 1.3px;
    font-weight: 700;
}


/* =========================================================
   SIDEBAR SECTION
   ========================================================= */

.side-section {
    color: #4CC9F0 !important;

    font-size: 0.68rem;
    font-weight: 800;

    letter-spacing: 1.4px;
    text-transform: uppercase;

    margin-bottom: 11px;
}


/* =========================================================
   WORKFLOW
   ========================================================= */

.workflow {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 11px 0;
}

.workflow-num {
    min-width: 25px;
    height: 25px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #0D2032;
    border: 1px solid #25425B;

    border-radius: 6px;

    color: #67E8F9 !important;

    font-size: 0.65rem;
    font-weight: 800;
}

.workflow-text {
    color: #B9C7D6 !important;

    font-size: 0.77rem;
    line-height: 1.5;
}


/* =========================================================
   TOOL CARDS
   ========================================================= */

.tool-box {
    display: flex;
    align-items: center;

    padding: 10px 12px;
    margin-bottom: 7px;

    background: #0D1A29;

    border: 1px solid #1E344B;
    border-radius: 8px;

    color: #D1DCE8 !important;

    font-size: 0.77rem;
}


/* =========================================================
   MODEL CARD
   ========================================================= */

.model-card {
    background: #0D1B2A;

    border: 1px solid #24435C;

    border-radius: 10px;

    padding: 13px;
}

.model-label {
    color: #71859A !important;

    font-size: 0.63rem;

    text-transform: uppercase;

    letter-spacing: 1px;

    font-weight: 800;

    margin-bottom: 6px;
}

.model-value {
    color: #5ED8FF !important;

    font-family: Consolas, monospace;

    font-size: 0.74rem;

    font-weight: 700;

    word-break: break-word;
}


/* =========================================================
   SECURITY
   ========================================================= */

.security-card {
    background: #0A1D1C;

    border: 1px solid #1B4842;

    border-radius: 9px;

    padding: 12px;

    color: #9FC3BE !important;

    font-size: 0.71rem;

    line-height: 1.55;
}


/* =========================================================
   SIDEBAR BUTTON
   ========================================================= */

[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;

    background: #0D1B2A !important;

    color: #D9E5F0 !important;

    border: 1px solid #284058 !important;

    border-radius: 9px !important;

    min-height: 42px !important;

    font-weight: 700 !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #12263A !important;

    color: #FFFFFF !important;

    border-color: #39BCEB !important;
}


/* =========================================================
   TOP BAR
   ========================================================= */

.topbar {
    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 0 0 18px 0;

    margin-bottom: 24px;

    border-bottom: 1px solid #1A2C40;
}

.top-left {
    display: flex;

    align-items: center;

    gap: 12px;
}

.top-logo {
    width: 42px;
    height: 42px;

    display: flex;

    align-items: center;
    justify-content: center;

    background: #0C2940;

    border: 1px solid #18577B;

    border-radius: 10px;

    font-size: 20px;
}

.top-title {
    color: #FFFFFF !important;

    font-size: 1.02rem;

    font-weight: 800;
}

.top-subtitle {
    color: #71859A !important;

    font-size: 0.67rem;

    margin-top: 3px;
}

.online {
    display: flex;

    align-items: center;

    gap: 7px;

    padding: 7px 11px;

    background: #092019;

    border: 1px solid #18503E;

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

    box-shadow: 0 0 9px #34D399;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    background:
        linear-gradient(
            115deg,
            #0E1D2E 0%,
            #102B3E 50%,
            #0D3541 100%
        );

    border: 1px solid #285069;

    border-radius: 16px;

    padding: 38px 42px;

    margin-bottom: 24px;

    box-shadow:
        0 18px 50px rgba(0,0,0,0.25);
}

.hero-badge {
    display: inline-block;

    padding: 7px 12px;

    border-radius: 7px;

    background: #0B2B3F;

    border: 1px solid #1E5A76;

    color: #67E8F9 !important;

    font-size: 0.67rem;

    font-weight: 800;

    letter-spacing: 1.2px;

    margin-bottom: 15px;
}

.hero-title {
    color: #FFFFFF !important;

    font-size: 2.45rem;

    font-weight: 850;

    line-height: 1.15;

    margin-bottom: 12px;
}

.hero-highlight {
    color: #4DD7FF !important;
}

.hero-description {
    color: #C1CFDD !important;

    font-size: 0.96rem;

    max-width: 900px;

    line-height: 1.75;
}


/* =========================================================
   FEATURE CARDS
   ========================================================= */

.feature-card {
    background: #0C1827;

    border: 1px solid #1F354A;

    border-radius: 13px;

    padding: 20px;

    min-height: 150px;

    box-shadow:
        0 8px 28px rgba(0,0,0,0.16);
}

.feature-card:hover {
    border-color: #2B6078;
}

.feature-icon {
    font-size: 20px;

    margin-bottom: 10px;
}

.feature-title {
    color: #F1F6FB !important;

    font-size: 0.93rem;

    font-weight: 800;

    margin-bottom: 8px;
}

.feature-text {
    color: #A7B6C8 !important;

    font-size: 0.78rem;

    line-height: 1.65;
}


/* =========================================================
   SECTION
   ========================================================= */

.section-header {
    display: flex;

    align-items: center;

    justify-content: space-between;

    margin: 27px 0 10px 0;
}

.section-title {
    color: #DDE7F2 !important;

    font-size: 0.74rem;

    font-weight: 800;

    letter-spacing: 1.3px;

    text-transform: uppercase;
}

.section-status {
    color: #6E8195 !important;

    font-size: 0.67rem;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    background: #0E1B2A !important;

    color: #E1EAF3 !important;

    border: 1px solid #294057 !important;

    border-radius: 9px !important;

    min-height: 44px !important;

    font-size: 0.8rem !important;

    font-weight: 700 !important;
}

.stButton > button:hover {
    background: #12283B !important;

    color: #FFFFFF !important;

    border-color: #35BCEB !important;
}

.stButton > button p {
    color: inherit !important;
}


/* =========================================================
   CHAT MESSAGE
   ========================================================= */

[data-testid="stChatMessage"] {
    background: #0B1725 !important;

    border: 1px solid #20364B !important;

    border-radius: 13px !important;

    padding: 15px 18px !important;

    margin-bottom: 12px !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #EAF1F8 !important;
}

[data-testid="stChatMessage"] p {
    color: #EAF1F8 !important;

    line-height: 1.7 !important;
}

[data-testid="stChatMessage"] li {
    color: #EAF1F8 !important;
}

[data-testid="stChatMessage"] strong {
    color: #FFFFFF !important;
}

[data-testid="stChatMessage"] code {
    color: #8BE8FF !important;
}


/* =========================================================
   CHAT INPUT
   ========================================================= */

.stBottom {
    background: #07111F !important;

    border-top: 1px solid #1B2D41 !important;
}

[data-testid="stBottom"] {
    background: #07111F !important;
}

[data-testid="stBottomBlockContainer"] {
    background: #07111F !important;

    padding-top: 10px !important;

    padding-bottom: 14px !important;
}

[data-testid="stChatInput"] {
    background: #0D1B2A !important;

    border: 1px solid #29435B !important;

    border-radius: 13px !important;

    padding: 5px !important;

    box-shadow:
        0 10px 35px rgba(0,0,0,0.35) !important;
}

[data-testid="stChatInput"] > div {
    background: #0D1B2A !important;
}

[data-testid="stChatInput"] textarea {
    background: #0D1B2A !important;

    color: #FFFFFF !important;

    -webkit-text-fill-color: #FFFFFF !important;

    caret-color: #4DD7FF !important;

    border: none !important;

    font-size: 0.92rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #74879B !important;

    -webkit-text-fill-color: #74879B !important;

    opacity: 1 !important;
}

[data-testid="stChatInput"] button {
    background: #0798D4 !important;

    color: #FFFFFF !important;

    border: none !important;

    border-radius: 9px !important;
}

[data-testid="stChatInput"] button:hover {
    background: #007DB5 !important;
}


/* =========================================================
   EXPANDER
   ========================================================= */

[data-testid="stExpander"] {
    background: #0A1522 !important;

    border: 1px solid #21384D !important;

    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    color: #D9E4EF !important;
}

[data-testid="stExpander"] p {
    color: #D9E4EF !important;
}


/* =========================================================
   ALERTS
   ========================================================= */

[data-testid="stAlert"] {
    border-radius: 10px !important;
}

[data-testid="stAlert"] p {
    color: #E5EDF5 !important;
}


/* =========================================================
   CODE BLOCK
   ========================================================= */

[data-testid="stCodeBlock"] {
    background: #050B12 !important;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;

    padding: 35px 0 15px;

    color: #64778B !important;

    font-size: 0.7rem;

    line-height: 1.8;
}

.footer strong {
    color: #8A9BAE !important;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .block-container {
        padding-left: 15px !important;
        padding-right: 15px !important;
    }

    .hero {
        padding: 27px 23px;
    }

    .hero-title {
        font-size: 1.9rem;
    }

    .online {
        display: none;
    }

    .topbar {
        margin-bottom: 18px;
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

    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add GROQ_API_KEY "
            "inside Streamlit Secrets."
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

Your responsibilities:

1. Answer questions clearly and professionally.
2. Keep answers structured and easy to read.
3. Use web search when current information is needed.
4. Use web search when the user explicitly asks to search online.
5. Use addition and multiplication tools for arithmetic when useful.
6. Never claim to have searched the web unless the search tool was actually used.
7. If a tool is used, use its result to produce a clear final answer.
8. Do not expose internal reasoning or hidden chain-of-thought.
"""

        response = llm_with_tools.invoke(
            [
                ("system", system_message)
            ] + state["messages"]
        )

        return {
            "messages": [response]
        }

    graph = StateGraph(MessagesState)

    graph.add_node(
        "assistant",
        assistant
    )

    graph.add_node(
        "tools",
        ToolNode(tools)
    )

    graph.add_edge(
        START,
        "assistant"
    )

    graph.add_conditional_edges(
        "assistant",
        tools_condition
    )

    graph.add_edge(
        "tools",
        "assistant"
    )

    return graph.compile()


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
                <div class="side-title">AI Agent</div>
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

    workflow_items = [
        ("01", "User sends a request"),
        ("02", "GPT-OSS 120B analyzes it"),
        ("03", "LangGraph selects a tool"),
        ("04", "Tool executes the task"),
        ("05", "AI generates the answer"),
    ]

    for number, text in workflow_items:

        st.markdown(
            f"""
            <div class="workflow">
                <div class="workflow-num">{number}</div>

                <div class="workflow-text">
                    {text}
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
        <div class="tool-box">🔎 &nbsp; Web Search</div>
        <div class="tool-box">➕ &nbsp; Addition</div>
        <div class="tool-box">✖️ &nbsp; Multiplication</div>
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
                Groq Inference Model
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
            <br><br>
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
            AI Agent <span class="hero-highlight">Workspace</span>
        </div>

        <div class="hero-description">
            A professional conversational AI workspace powered by
            GPT-OSS 120B through Groq. Ask questions, search the web,
            perform calculations and interact with an intelligent
            LangGraph agent workflow.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(
    3,
    gap="medium",
)


with col1:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">🧠</div>

            <div class="feature-title">
                Agentic Reasoning
            </div>

            <div class="feature-text">
                The AI analyzes the request and decides whether
                an external tool is required before responding.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">🔧</div>

            <div class="feature-title">
                Tool Orchestration
            </div>

            <div class="feature-text">
                LangGraph manages communication between the
                language model and available tools.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">⚡</div>

            <div class="feature-title">
                Fast AI Inference
            </div>

            <div class="feature-text">
                GPT-OSS 120B is served through Groq for fast and
                responsive Agentic AI interactions.
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


q1, q2, q3 = st.columns(
    3,
    gap="medium",
)


quick_actions = [
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
        "Explain LangGraph in simple terms with a practical example.",
    ),
]


for column, (label, question) in zip(
    (q1, q2, q3),
    quick_actions,
):

    with column:

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
                "AI Agent is processing your request..."
            ):

                agent = build_agent()

                previous_messages = list(
                    st.session_state.messages
                )

                result = agent.invoke(
                    {
                        "messages": previous_messages
                    }
                )

                all_messages = result["messages"]

                generated = all_messages[
                    len(previous_messages):
                ]

                st.session_state.messages.extend(
                    generated
                )

                # -----------------------------------------
                # TOOL ACTIVITY
                # -----------------------------------------

                tool_messages = [
                    message
                    for message in generated
                    if isinstance(
                        message,
                        ToolMessage
                    )
                ]

                if tool_messages:

                    with st.expander(
                        "🔧 Tool Activity",
                        expanded=False,
                    ):

                        for tool_message in tool_messages:

                            st.markdown(
                                f"**Tool: {tool_message.name}**"
                            )

                            st.code(
                                str(
                                    tool_message.content
                                )[:5000]
                            )

                # -----------------------------------------
                # FINAL AI RESPONSE
                # -----------------------------------------

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

                        final_response = str(
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
                        "but did not return a final response."
                    )

    except Exception as error:

        st.error(
            "The AI Agent could not process this request."
        )

        with st.expander(
            "Technical Details",
            expanded=False,
        ):

            st.code(
                str(error)
            )

        if (
            st.session_state.messages
            and st.session_state.messages[-1]
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
