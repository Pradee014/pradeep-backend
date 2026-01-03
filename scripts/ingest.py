import os
import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vector_store import get_vector_store
from app.core.config import settings
from pinecone import Pinecone, ServerlessSpec
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data"


def ensure_index_exists():
    """Create the Pinecone index if it doesn't exist."""
    pc = Pinecone(api_key=settings.PINECONE_API_KEY.get_secret_value())
    
    existing_indexes = pc.list_indexes().names()
    if settings.PINECONE_INDEX_NAME not in existing_indexes:
        logger.info(f"✨ Creating index '{settings.PINECONE_INDEX_NAME}' with dimensions={settings.EMBEDDING_DIMENSIONS}...")
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIMENSIONS, # 512
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(settings.PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)
        logger.info(f"✅ Index '{settings.PINECONE_INDEX_NAME}' is ready.")
    else:
        logger.info(f"✅ Index '{settings.PINECONE_INDEX_NAME}' already exists.")

def ingest_data():
    logger.info(f"🚀 Starting ingestion from {DATA_PATH}...")

    # 1. Pipeline Setup
    ensure_index_exists()

    if not os.path.exists(DATA_PATH):
        logger.error(f"❌ Data directory '{DATA_PATH}' not found.")
        return

    # 2. Load Documents (PDF, MD, TXT)
    documents = []
    
    # PDF Loader
    pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    # Text/Markdown Loader
    # Note: We use TextLoader for both for simplicity, or could separate if needed
    txt_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())
    
    md_loader = DirectoryLoader(DATA_PATH, glob="**/*.md", loader_cls=TextLoader)
    documents.extend(md_loader.load())

    if not documents:
        logger.warning("⚠️ No documents found. Add .pdf, .md, or .txt files to 'data/'")
        return

    logger.info(f"📚 Loaded {len(documents)} documents.")

    # 3. Split Text (Optimized for text-embedding-3-small)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    
    # 4. Metadata Enrichment ("The Secret Sauce")
    for chunk in chunks:
        source = chunk.metadata.get("source", "").lower()
        filename = os.path.basename(source)
        
        # Categorize
        if "resume" in filename or "cv" in filename:
            chunk.metadata["category"] = "professional_background"
            chunk.metadata["priority"] = 1.0 # High priority
        else:
            chunk.metadata["category"] = "general_knowledge"
            chunk.metadata["priority"] = 0.5 # Standard priority

    logger.info(f"✂️ Split into {len(chunks)} enriched chunks.")

    # 5. Embed & Upsert
    try:
        store = get_vector_store()
        logger.info("📡 Connecting to Pinecone...")
        
        # Add documents using the vector store wrapper
        # The store handles the embedding generation using the config in app/core/config.py
        store.add_documents(chunks)
        
        logger.info(f"✅ Successfully ingested {len(chunks)} chunks into Pinecone.")
        
    except Exception as e:
        logger.error(f"❌ Failed to ingest data: {str(e)}")

if __name__ == "__main__":
    ingest_data()
