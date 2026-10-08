# 🤖 AI Agent Chatbot

Professional Agentic AI chatbot built with **LangGraph**, **LangChain**, and **Groq**, using **OpenAI GPT-OSS 120B**.

## Features
- LangGraph state-based agent workflow
- GPT-OSS 120B through Groq
- Tool calling
- DuckDuckGo web search
- Addition and multiplication tools
- Streamlit chat UI
- Conversation history and clear button
- Streamlit Secrets for API key security

## Architecture
User → Streamlit → LangGraph Agent → GPT-OSS 120B → Tool Decision → Tools → Final Answer

## Local setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

For local testing create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_groq_api_key"
```
Never commit that file.

## Streamlit deployment
Connect this GitHub repository to Streamlit Community Cloud and add:
```toml
GROQ_API_KEY = "your_groq_api_key"
```
in the app Secrets settings.

## Model
`openai/gpt-oss-120b`

## Disclaimer
AI-generated responses can contain errors. Verify important information before relying on it.
