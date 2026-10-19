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

load_dotenv()
st.set_page_config(page_title="Agentic AI + RAG", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.stApp{background:#0b1220;color:#edf2f7}
[data-testid="stSidebar"]{background:#101a2b}
[data-testid="stChatMessage"]{border:1px solid #26364b;border-radius:14px}
</style>
""", unsafe_allow_html=True)
st.title("🤖 Agentic AI Assistant + RAG")
st.caption("Groq GPT-OSS-120B • LangChain • LangGraph • Web Search • PDF/TXT Q&A")

api_key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
api_key = api_key or os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY missing. Add it to .env or Streamlit secrets.")
    st.code('GROQ_API_KEY="your_real_groq_api_key"')
    st.stop()

for key, default in {
    "lc_messages": [], "display_messages": [], "vectorstore": None,
    "doc_fingerprint": None, "indexed_files": [], "last_tools": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b

web_search = DuckDuckGoSearchRun()

def make_rag_tool(vectorstore):
    @tool
    def search_uploaded_documents(query: str) -> str:
        """Search uploaded PDF/TXT files for relevant passages. Use for questions about uploaded documents."""
        docs = vectorstore.similarity_search(query, k=4)
        if not docs:
            return "No relevant passages found in the uploaded documents."
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Uploaded document")
            page = doc.metadata.get("page")
            page_info = f", page {page + 1}" if isinstance(page, int) else ""
            results.append(f"[Source {i}: {source}{page_info}]\n{doc.page_content}")
        return "\n\n".join(results)
    return search_uploaded_documents

def build_index(files):
    all_docs, fingerprint_parts, names = [], [], []
    for uploaded in files:
        raw = uploaded.getvalue()
        fingerprint_parts.append(uploaded.name.encode() + raw)
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in (".pdf", ".txt"):
            st.warning(f"Skipped {uploaded.name}: PDF/TXT only.")
            continue
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw)
                temp_path = tmp.name
            docs = PyPDFLoader(temp_path).load() if suffix == ".pdf" else TextLoader(temp_path, encoding="utf-8").load()
            for doc in docs:
                doc.metadata["source"] = uploaded.name
            all_docs.extend(docs)
            names.append(uploaded.name)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    fingerprint = hashlib.sha256(b"".join(fingerprint_parts)).hexdigest()
    if not all_docs:
        st.session_state.vectorstore = None
        st.session_state.doc_fingerprint = fingerprint
        st.session_state.indexed_files = []
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150, add_start_index=True)
    chunks = splitter.split_documents(all_docs)
    with st.spinner("Building document embeddings and FAISS index..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    st.session_state.doc_fingerprint = fingerprint
    st.session_state.indexed_files = names

with st.sidebar:
    st.header("📚 RAG Knowledge Base")
    st.write("Upload documents, then ask questions about their contents.")
    files = st.file_uploader("Upload PDF/TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if files:
        fingerprint = hashlib.sha256(b"".join(f.name.encode() + f.getvalue() for f in files)).hexdigest()
        if fingerprint != st.session_state.doc_fingerprint:
            try:
                build_index(files)
                if st.session_state.vectorstore:
                    st.success(f"Indexed {len(st.session_state.indexed_files)} file(s).")
                else:
                    st.warning("No readable text found.")
            except Exception as exc:
                st.session_state.vectorstore = None
                st.error(f"Indexing failed: {exc}")
    else:
        st.session_state.vectorstore = None
        st.session_state.doc_fingerprint = None
        st.session_state.indexed_files = []

    if st.session_state.indexed_files:
        st.markdown("**Indexed files**")
        for name in st.session_state.indexed_files:
            st.caption("• " + name)
    st.divider()
    st.markdown("**Tools**")
    st.caption("➕ Addition")
    st.caption("✖️ Multiplication")
    st.caption("🌐 DuckDuckGo search")
    st.caption("📄 RAG document retrieval")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.lc_messages = []
        st.session_state.display_messages = []
        st.session_state.last_tools = []
        st.rerun()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, max_tokens=2048, api_key=api_key)
tools = [add, multiply, web_search]
if st.session_state.vectorstore is not None:
    tools.append(make_rag_tool(st.session_state.vectorstore))
llm_with_tools = llm.bind_tools(tools)

system_prompt = """You are a helpful Agentic AI Assistant.
Use calculator tools for calculations and DuckDuckGo for web searches when useful.
When the user asks about uploaded documents, use search_uploaded_documents if available.
Base document answers on retrieved passages and cite the source filename and page when available.
If retrieved passages do not answer the question, clearly say the uploaded documents do not contain enough information.
Never claim to have used a tool unless it was actually called."""

def assistant_node(state: MessagesState):
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile()

for item in st.session_state.display_messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

prompt = st.chat_input("Ask a question, calculate, search the web, or ask about your files...")
if prompt:
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = graph.invoke({"messages": st.session_state.lc_messages + [HumanMessage(content=prompt)]})
            st.session_state.lc_messages = result["messages"]
            answer, used_tools = "", []
            for message in result["messages"]:
                if isinstance(message, AIMessage):
                    used_tools.extend(call.get("name", "tool") for call in (message.tool_calls or []))
                    if isinstance(message.content, str) and message.content.strip():
                        answer = message.content
                    elif isinstance(message.content, list):
                        blocks = [b.get("text", "") for b in message.content if isinstance(b, dict) and b.get("type") == "text"]
                        if blocks:
                            answer = "\n".join(blocks)
            if not answer:
                answer = "I couldn't create a text response. Please try again."
            st.session_state.last_tools = used_tools
            st.markdown(answer)
        st.session_state.display_messages.append({"role": "assistant", "content": answer})
    except Exception as exc:
        st.error(f"Error: {exc}\n\nCheck your Groq API key, model access, dependencies, and internet connection.")

if st.session_state.last_tools:
    with st.expander("🔎 Tool activity"):
        st.write(", ".join(st.session_state.last_tools))
st.caption("PDF/TXT files are indexed in the current session. This starter version does not persist the vector database after restart.")
