from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database settings
    database_url: str = "postgresql://username:password@localhost:5432/robotics_book"

    # Qdrant settings
    qdrant_url: str = "https://your-cluster-url.qdrant.io"
    qdrant_api_key: str = "your_qdrant_api_key"
    qdrant_collection_name: str = "book_embeddings"

    # Google Gemini settings
    gemini_api_key: str = "your_gemini_api_key"

    # Application settings
    secret_key: str = "your-secret-key-here-generate-secure-one"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Application settings
    debug: bool = True
    app_name: str = "AI and Robotics Book Platform"
    version: str = "1.0.0"

    # RAG settings
    rag_threshold: float = 0.7  # Threshold for vector search
    max_search_results: int = 5

    # Performance settings
    response_timeout: int = 30  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()