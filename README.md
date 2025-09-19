# 📚 RAG PDF Chatbot with Gemini + LlamaIndex

This is a **Retrieval-Augmented Generation (RAG) chatbot** built using:
- **LlamaIndex** for document loading + indexing
- **Gemini (Google AI)** as the LLM + embeddings
- **ChromaDB** for persistent vector storage
- **Streamlit** for chat UI

## 🚀 Features
- Upload PDFs into the `data/` folder
- Chat with Gemini about your documents
- Persistent vector storage with ChromaDB
- Conversational memory using `chat_engine`

## 🛠️ Setup
```bash
git clone https://github.com/yourusername/rag-pdf-chatbot-gemini.git
cd rag-pdf-chatbot-gemini
pip install -r requirements.txt
```

Set your **Google API Key**:
```bash
export GOOGLE_API_KEY="your_api_key"
```

## ▶️ Run
```bash
streamlit run src/app.py
```

## 📂 Demo
*(Add demo video/gif here)*
