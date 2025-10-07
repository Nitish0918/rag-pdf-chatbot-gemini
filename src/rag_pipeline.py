import json
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- ✅ Universal Import Block for HuggingFaceEmbedding ---
try:
    # Newer versions (>=0.10.0)
    from llama_index.embeddings.huggingface_base import HuggingFaceEmbedding
    print("✅ Imported HuggingFaceEmbedding from huggingface_base (new version)")
except ModuleNotFoundError:
    try:
        # Older versions (<0.10.0)
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print("✅ Imported HuggingFaceEmbedding from huggingface (old version)")
    except ModuleNotFoundError as e:
        print("❌ Failed to import HuggingFaceEmbedding.")
        print("   Please install dependencies with:")
        print("   pip install -U llama-index llama-index-embeddings-huggingface")
        raise e

# --- Initialize Gemini LLM ---
llm = Gemini(model="models/gemini-2.0-flash")

# --- Initialize Embedding Model with Auto-Fallback ---
embedding_source = "gemini"  # Track current source

try:
    print("🔹 Trying Gemini embeddings...")
    embedding_model = GeminiEmbedding(model="models/embedding-001")

    # Test to detect quota/key issues
    _ = embedding_model.get_text_embedding("test")
    print("✅ Gemini embeddings active.")

except Exception as e:
    print(f"⚠️ Gemini embedding failed: {e}")
    print("🔄 Switching to local HuggingFace embeddings...")
    embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedding_source = "local"
    print("✅ Local embeddings active (HuggingFace).")

# --- Save embedding source status to JSON ---
status_data = {"embedding_source": embedding_source}
with open("embedding_status.json", "w") as f:
    json.dump(status_data, f, indent=2)

print(f"📊 Embedding source in use: {embedding_source}")
print("📝 Status written to embedding_status.json")

# --- Build Index Function ---
def build_index(pdf_dir="data"):
    print("📚 Loading documents...")
    docs = SimpleDirectoryReader(pdf_dir).load_data()

    print("💾 Setting up persistent ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("pdf_chatbot")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"⚙️ Building index using {embedding_source} embeddings...")
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embedding_model
    )
    print("✅ Index built successfully.")
    return index

# --- Chat Engine Setup ---
def get_chat_engine(index):
    print(f"💬 Chat engine ready (using {embedding_source} embeddings).")
    return index.as_chat_engine(
        chat_mode="condense_question",
        llm=llm,
        verbose=True
    )

# --- Helper Function to Check Embedding Source ---
def get_embedding_source():
    """Returns which embedding backend is currently in use."""
    try:
        with open("embedding_status.json", "r") as f:
            status = json.load(f)
        return status.get("embedding_source", "unknown")
    except FileNotFoundError:
        return "unknown"
