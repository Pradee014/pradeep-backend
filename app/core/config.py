from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Pradeep's Agentic RAG"
    DEBUG: bool = True
    
    # API Keys & Secrets
    OPENAI_API_KEY: SecretStr
    PINECONE_API_KEY: SecretStr
    OLLAMA_API_KEY: SecretStr
    
    # External Services
    OLLAMA_BASE_URL: str = "https://api.ollamacloud.com"  # Default generic URL, user to override
    PINECONE_ENV: str = "us-east-1" # Example default
    
    # Paths / Constants
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 512

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra fields in .env

settings = Settings()
