from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Initialize Gemini LLM + Embeddings
llm = Gemini(model="models/gemini-1.5-flash")
embedding_model = GeminiEmbedding(model="models/embedding-001")

def build_index(pdf_dir="data"):
    # Load PDFs
    docs = SimpleDirectoryReader(pdf_dir).load_data()

    # Setup persistent ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("pdf_chatbot")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build Index
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embedding_model
    )
    return index

def get_chat_engine(index):
    return index.as_chat_engine(
        chat_mode="condense_question",
        llm=llm,
        verbose=True
    )
