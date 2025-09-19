import streamlit as st
from rag_pipeline import build_index, get_chat_engine

st.set_page_config(page_title="RAG PDF Chatbot (Gemini)", layout="centered")
st.title("📚 RAG PDF Chatbot (Gemini + LlamaIndex)")

# Build/load index once
if "chat_engine" not in st.session_state:
    index = build_index("data")
    st.session_state.chat_engine = get_chat_engine(index)
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("Ask something about your PDF:")

if user_input:
    response = st.session_state.chat_engine.chat(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response.response))

# Show conversation
for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {msg}")
