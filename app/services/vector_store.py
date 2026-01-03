from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

def get_vector_store():
    """
    Returns a configured PineconeVectorStore instance.
    Uses OpenAI Embeddings with dimensions=512.
    """
    
    # Initialize Embeddings
    # Note: text-embedding-3-small supports 'dimensions' param to truncate embeddings
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS, 
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )

    # Initialize Vector Store
    # We assume the index already exists as per user confirmation
    vector_store = PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=settings.PINECONE_API_KEY.get_secret_value()
    )
    
    return vector_store

async def search_documents(query: str, k: int = 4):
    """
    Semantic search for documents relevant to the query.
    """
    try:
        store = get_vector_store()
        # similarity_search_with_score returns List[Tuple[Document, float]]
        results = store.similarity_search_with_score(query, k=k)
        return results
    except Exception as e:
        logger.error(f"Error checking vector store: {str(e)}")
        # Return empty list gracefully if Pinecone is down/unconfigured so the app doesn't crash
        return []
