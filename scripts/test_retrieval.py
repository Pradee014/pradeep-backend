import asyncio
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings

# Configuration
QUERY = "What are my top 3 technical skills?"
TOP_K = 2

def test_retrieval():
    print(f"🧪 Starting Acid Test...")
    print(f"🔎 Query: '{QUERY}'")
    
    # 1. Initialize Embeddings
    # Must match ingestion dims (512)
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )

    # 2. Initialize Vector Store
    vector_store = PineconeVectorStore(
        index_name="portfolio-rag",
        embedding=embeddings,
        pinecone_api_key=settings.PINECONE_API_KEY.get_secret_value()
    )

    # 3. Execute Search
    print("📡 Querying Pinecone...")
    results = vector_store.similarity_search(QUERY, k=TOP_K)

    # 4. output Results
    if not results:
        print("❌ No results found!")
        return

    print(f"✅ Found {len(results)} matches:\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} (Source: {doc.metadata.get('source', 'unknown')}) ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content Snippet:\n{doc.page_content[:300]}...\n")

if __name__ == "__main__":
    test_retrieval()
