
import os
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# =====================================================
# PAGE CONFIGURATION
# =====================================================
load_dotenv()

st.set_page_config(
    page_title="Agentic AI Assistant + RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# PROFESSIONAL LIGHT THEME - HIGH CONTRAST
# =====================================================
st.markdown("""
<style>
:root {
    --bg: #F3F6FB;
    --white: #FFFFFF;
    --navy: #17365D;
    --text: #26364A;
    --muted: #52657A;
    --border: #D7E1EE;
    --blue: #2563EB;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #F3F6FB !important;
    color: #26364A !important;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: #F3F6FB !important;
}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #FFFFFF !important;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #D7E1EE !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: #26364A !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #17365D !important;
    font-weight: 750 !important;
    letter-spacing: -0.02em;
}

p, li, label, .stMarkdown {
    color: #26364A !important;
}

[data-testid="stCaptionContainer"] p {
    color: #52657A !important;
}

/* Upload panel */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px solid #D7E1EE !important;
    border-radius: 14px !important;
    padding: 12px !important;
}

[data-testid="stFileUploader"] section {
    background: #F8FAFD !important;
    border: 1px dashed #A9BCD3 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] * {
    color: #26364A !important;
}

/* Text inputs */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #172B4D !important;
    border: 1px solid #B8C8DA !important;
    border-radius: 10px !important;
    caret-color: #2563EB !important;
}

input::placeholder,
textarea::placeholder {
    color: #64748B !important;
    opacity: 1 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1px solid #D7E1EE !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 16px rgba(23, 54, 93, 0.07) !important;
}

[data-testid="stChatInput"] textarea {
    border: none !important;
    box-shadow: none !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #2563EB !important;
    border-radius: 10px !important;
    font-weight: 650 !important;
}

.stButton > button *,
.stFormSubmitButton > button * {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    color: #26364A !important;
    border: 1px solid #D7E1EE !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    box-shadow: 0 3px 12px rgba(23, 54, 93, 0.04);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #26364A !important;
}

/* Alerts and expanders */
[data-testid="stAlert"],
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    color: #26364A !important;
    border: 1px solid #D7E1EE !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #26364A !important;
}

hr {
    border-color: #D7E1EE !important;
}

a {
    color: #1D4ED8 !important;
}

/* Tool cards */
.tool-card {
    background: #F3F7FD;
    border: 1px solid #DFE8F3;
    color: #243B57 !important;
    border-radius: 9px;
    padding: 10px 12px;
    margin: 6px 0;
    font-size: 14px;
}

.hero-subtitle {
    color: #52657A !important;
    font-size: 16px;
}

@media (max-width: 700px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1rem 0.8rem 2rem;
    }
    h1 {
        font-size: 1.8rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DASHBOARD HEADER
# =====================================================
st.title("🤖 Agentic AI Assistant + RAG")
st.markdown(
    '<p class="hero-subtitle">'
    'Groq GPT-OSS-120B · LangChain · LangGraph · Web Search · PDF/TXT Q&A'
    '</p>',
    unsafe_allow_html=True,
)
st.info(
    "Upload a PDF or TXT file from the sidebar, then ask questions about it. "
    "Documents are indexed for the current session only."
)

# =====================================================
# GROQ API KEY
# =====================================================
api_key = None

try:
    api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    api_key = None

api_key = api_key or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. Add it in Streamlit Cloud → "
        "App settings → Secrets."
    )
    st.code('GROQ_API_KEY = "your_real_groq_api_key"')
    st.stop()

# =====================================================
# SESSION STATE
# =====================================================
defaults = {
    "lc_messages": [],
    "display_messages": [],
    "vectorstore": None,
    "doc_fingerprint": None,
    "indexed_files": [],
    "last_tools": [],
}

for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =====================================================
# CALCULATOR TOOLS
# =====================================================
@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


web_search = DuckDuckGoSearchRun()

# =====================================================
# RAG SEARCH TOOL
# =====================================================
def make_rag_tool(vectorstore):
    @tool
    def search_uploaded_documents(query: str) -> str:
        """Search uploaded PDF/TXT files for relevant passages."""
        docs = vectorstore.similarity_search(query, k=4)

        if not docs:
            return "No relevant passages found in the uploaded documents."

        results = []

        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Uploaded document")
            page = doc.metadata.get("page")
            page_info = f", page {page + 1}" if isinstance(page, int) else ""

            results.append(
                f"[Source {i}: {source}{page_info}]\n{doc.page_content}"
            )

        return "\n\n".join(results)

    return search_uploaded_documents

# =====================================================
# BUILD PDF/TXT INDEX
# =====================================================
def build_index(files):
    all_docs = []
    fingerprint_parts = []
    names = []

    for uploaded in files:
        raw = uploaded.getvalue()
        fingerprint_parts.append(uploaded.name.encode("utf-8") + raw)
        suffix = Path(uploaded.name).suffix.lower()

        if suffix not in (".pdf", ".txt"):
            st.warning(f"Skipped {uploaded.name}: PDF/TXT files only.")
            continue

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(raw)
                temp_path = tmp.name

            if suffix == ".pdf":
                docs = PyPDFLoader(temp_path).load()
            else:
                docs = TextLoader(
                    temp_path, encoding="utf-8"
                ).load()

            for doc in docs:
                doc.metadata["source"] = uploaded.name

            all_docs.extend(docs)
            names.append(uploaded.name)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    fingerprint = hashlib.sha256(
        b"".join(fingerprint_parts)
    ).hexdigest()

    if not all_docs:
        st.session_state.vectorstore = None
        st.session_state.doc_fingerprint = fingerprint
        st.session_state.indexed_files = []
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(all_docs)

    if not chunks:
        st.session_state.vectorstore = None
        st.session_state.doc_fingerprint = fingerprint
        st.session_state.indexed_files = []
        return

    with st.spinner("Building document embeddings and FAISS index..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        st.session_state.vectorstore = FAISS.from_documents(
            chunks, embeddings
        )

    st.session_state.doc_fingerprint = fingerprint
    st.session_state.indexed_files = names

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("📚 RAG Knowledge Base")
    st.write("Upload documents, then ask questions about their contents.")

    files = st.file_uploader(
        "Upload PDF/TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if files:
        fingerprint = hashlib.sha256(
            b"".join(
                f.name.encode("utf-8") + f.getvalue()
                for f in files
            )
        ).hexdigest()

        if fingerprint != st.session_state.doc_fingerprint:
            try:
                build_index(files)

                if st.session_state.vectorstore is not None:
                    st.success(
                        f"Indexed {len(st.session_state.indexed_files)} file(s)."
                    )
                else:
                    st.warning("No readable text found in the uploaded files.")

            except Exception as exc:
                st.session_state.vectorstore = None
                st.session_state.doc_fingerprint = fingerprint
                st.session_state.indexed_files = []
                st.error(f"Document indexing failed: {exc}")

    else:
        st.session_state.vectorstore = None
        st.session_state.doc_fingerprint = None
        st.session_state.indexed_files = []

    if st.session_state.indexed_files:
        st.markdown("**Indexed files**")
        for name in st.session_state.indexed_files:
            st.caption(f"📄 {name}")

    st.divider()
    st.subheader("Available Tools")

    st.markdown(
        '<div class="tool-card">➕ Addition calculator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tool-card">✖️ Multiplication calculator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tool-card">🌐 DuckDuckGo web search</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tool-card">📄 PDF/TXT document retrieval</div>',
        unsafe_allow_html=True,
    )

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.lc_messages = []
        st.session_state.display_messages = []
        st.session_state.last_tools = []
        st.rerun()

# =====================================================
# GROQ MODEL + LANGGRAPH
# =====================================================
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=2048,
    api_key=api_key,
)

tools = [add, multiply, web_search]

if st.session_state.vectorstore is not None:
    tools.append(
        make_rag_tool(st.session_state.vectorstore)
    )

llm_with_tools = llm.bind_tools(tools)

system_prompt = """You are a helpful Agentic AI Assistant.
Use calculator tools for calculations and DuckDuckGo for web searches when useful.
When the user asks about uploaded documents, use search_uploaded_documents if available.
Base document answers on retrieved passages and cite source filename and page when available.
If retrieved passages do not answer the question, clearly say the uploaded documents do not contain enough information.
Never claim to have used a tool unless it was actually called."""


def assistant_node(state: MessagesState):
    response = llm_with_tools.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()

# =====================================================
# CHAT HISTORY
# =====================================================
for item in st.session_state.display_messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

prompt = st.chat_input(
    "Ask a question, calculate, search the web, or ask about your files..."
)

if prompt:
    st.session_state.display_messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = graph.invoke({
                    "messages": (
                        st.session_state.lc_messages
                        + [HumanMessage(content=prompt)]
                    )
                })

            st.session_state.lc_messages = result["messages"]

            answer = ""
            used_tools = []

            # Read only the latest AI response for display.
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage):
                    used_tools = [
                        call.get("name", "tool")
                        for call in (message.tool_calls or [])
                    ]

                    if isinstance(message.content, str) and message.content.strip():
                        answer = message.content
                        break

                    if isinstance(message.content, list):
                        blocks = [
                            block.get("text", "")
                            for block in message.content
                            if isinstance(block, dict)
                            and block.get("type") == "text"
                        ]
                        if blocks:
                            answer = "\n".join(blocks)
                            break

            if not answer:
                answer = "I couldn't create a text response. Please try again."

            st.session_state.last_tools = used_tools
            st.markdown(answer)

        st.session_state.display_messages.append({
            "role": "assistant",
            "content": answer,
        })

    except Exception as exc:
        st.error(
            f"Error: {exc}\n\n"
            "Check your Groq API key, model access, dependencies, "
            "and internet connection."
        )

if st.session_state.last_tools:
    with st.expander("🔎 Tool activity"):
        st.write(", ".join(st.session_state.last_tools))

st.caption(
    "PDF/TXT files are indexed for the current session only. "
    "The document index does not persist after the app restarts."
)
