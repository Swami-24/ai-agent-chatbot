import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(page_title='AI Agent Chatbot', page_icon='🤖', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
.stApp{background:#0B1020}[data-testid="stSidebar"]{background:#11182B;border-right:1px solid #27324A}[data-testid="stSidebar"] *{color:#E5E7EB}
.hero{padding:34px 38px;border-radius:24px;margin-bottom:24px;background:linear-gradient(135deg,#172554 0%,#312E81 50%,#581C87 100%);border:1px solid #4C4A91;box-shadow:0 15px 40px rgba(0,0,0,.25)}
.hero-badge{display:inline-block;padding:6px 13px;border-radius:999px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);color:#E0E7FF;font-size:.78rem;font-weight:700;letter-spacing:1px;margin-bottom:12px}.hero h1{color:#FFF!important;font-size:2.45rem;font-weight:800;margin:0 0 10px}.hero p{color:#DDE4FF!important;font-size:1.03rem;line-height:1.7;margin:0;max-width:850px}
.feature-card{background:#11182B;border:1px solid #27324A;border-radius:17px;padding:20px;min-height:145px;height:100%}.feature-card h3{color:#C4B5FD!important;font-size:1rem;margin:0 0 10px}.feature-card p{color:#AEBBD2!important;line-height:1.6;font-size:.91rem;margin:0}.section-label{color:#A5B4FC!important;font-size:.78rem;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:4px}.stButton>button{border-radius:10px!important;min-height:42px;font-weight:650!important}.stChatInput textarea{background:#11182B!important;color:#FFF!important}.footer{text-align:center;color:#64748B;font-size:.78rem;padding:24px 0 8px}
</style>''', unsafe_allow_html=True)

@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b

search_tool = DuckDuckGoSearchRun()
tools = [search_tool, add, multiply]

@st.cache_resource(show_spinner=False)
def build_agent():
    api_key = st.secrets.get('GROQ_API_KEY') or os.getenv('GROQ_API_KEY')
    if not api_key:
        raise RuntimeError('GROQ_API_KEY is missing. Add it to Streamlit Secrets.')
    llm = ChatGroq(model='openai/gpt-oss-120b', api_key=api_key, temperature=0.2, max_tokens=2048)
    llm_with_tools = llm.bind_tools(tools)
    def assistant(state: MessagesState):
        system_message = ('You are a professional AI Agent Assistant. Answer clearly and accurately. '
                          'Use available tools when useful. Use web search for current information, '
                          'specific facts needing online verification, or explicit web-search requests. '
                          'Use calculator tools for arithmetic. After tool results, explain naturally. '
                          'Do not claim to have searched the web if you did not use the search tool.')
        response = llm_with_tools.invoke([('system', system_message)] + state['messages'])
        return {'messages': [response]}
    builder = StateGraph(MessagesState)
    builder.add_node('assistant', assistant)
    builder.add_node('tools', ToolNode(tools))
    builder.add_edge(START, 'assistant')
    builder.add_conditional_edges('assistant', tools_condition)
    builder.add_edge('tools', 'assistant')
    return builder.compile()

if 'messages' not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.markdown('## 🤖 AI Agent Chatbot')
    st.caption('LANGGRAPH • GROQ • AGENTIC AI')
    st.divider()
    st.markdown('### ⚡ How it works')
    st.markdown('1. You ask a question.\n2. GPT-OSS 120B analyzes it.\n3. LangGraph decides whether a tool is needed.\n4. The selected tool executes.\n5. The agent returns the final answer.')
    st.divider()
    st.markdown('### 🧰 Available Tools')
    st.markdown('- 🔎 Web Search — DuckDuckGo\n- ➕ Addition — custom Python tool\n- ✖️ Multiplication — custom Python tool')
    st.divider()
    st.markdown('### 🧠 Model')
    st.code('openai/gpt-oss-120b', language='text')
    st.divider()
    st.markdown('### 🔐 Security')
    st.caption('The Groq API key is loaded from Streamlit Secrets. Never place your API key directly in app.py or GitHub.')
    if st.button('🗑️ Clear Conversation', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown('''<div class="hero"><div class="hero-badge">✦ AGENTIC AI • LANGGRAPH</div><h1>AI Agent Chatbot</h1><p>A professional conversational AI agent powered by OpenAI GPT-OSS 120B through Groq, with LangGraph tool-calling, web search, and custom calculation tools.</p></div>''', unsafe_allow_html=True)

c1,c2,c3=st.columns(3,gap='medium')
for col,title,text in [(c1,'🧠 Agentic Reasoning','The agent can decide whether a normal response or an external tool is required.'),(c2,'🔧 Tool Calling','LangGraph connects the language model with web search and custom Python tools.'),(c3,'⚡ Fast AI Inference','GPT-OSS 120B runs through Groq for fast inference and agentic workflows.')]:
    with col: st.markdown(f'<div class="feature-card"><h3>{title}</h3><p>{text}</p></div>', unsafe_allow_html=True)

st.write(''); st.markdown('<div class="section-label">TRY AN EXAMPLE</div>', unsafe_allow_html=True)
e1,e2,e3=st.columns(3)
examples=[(e1,'🔎 Web Research','Search the web for the latest developments in AI agents.'),(e2,'➗ Calculation','Multiply 125 by 24 and add 500.'),(e3,'💬 General AI','Explain how LangGraph works in simple terms.')]
for col,label,prompt in examples:
    with col:
        if st.button(label,use_container_width=True): st.session_state.pending_prompt=prompt; st.rerun()

for message in st.session_state.messages:
    if isinstance(message,HumanMessage):
        with st.chat_message('user'): st.markdown(message.content)
    elif isinstance(message,AIMessage) and message.content:
        with st.chat_message('assistant'): st.markdown(message.content)

prompt=st.chat_input('Ask the AI agent anything...')
if 'pending_prompt' in st.session_state: prompt=st.session_state.pop('pending_prompt')

if prompt:
    user_message=HumanMessage(content=prompt); st.session_state.messages.append(user_message)
    with st.chat_message('user'): st.markdown(prompt)
    try:
        with st.chat_message('assistant'):
            with st.spinner('🤖 Agent is thinking...'):
                agent=build_agent(); old_count=len(st.session_state.messages)
                result=agent.invoke({'messages':st.session_state.messages})
                generated=result['messages'][old_count:]
                st.session_state.messages.extend(generated)
                tool_messages=[m for m in generated if isinstance(m,ToolMessage)]
                if tool_messages:
                    with st.expander('🧰 Tool activity',expanded=False):
                        for tm in tool_messages:
                            st.write(f'**{tm.name}**'); st.code(str(tm.content)[:4000])
                final_text=next((m.content for m in reversed(generated) if isinstance(m,AIMessage) and m.content), '')
                if final_text: st.markdown(final_text)
                else: st.warning('The agent completed the tool workflow but did not return a final text response. Please try again.')
    except Exception as e:
        st.error('The AI agent could not complete this request.')
        with st.expander('Technical error details'): st.code(str(e))
        if st.session_state.messages and st.session_state.messages[-1] == user_message: st.session_state.messages.pop()

st.markdown('<div class="footer"><strong>AI Agent Chatbot</strong> · LangGraph · LangChain · Groq<br>Built for Agentic AI experimentation and deployment.</div>', unsafe_allow_html=True)
