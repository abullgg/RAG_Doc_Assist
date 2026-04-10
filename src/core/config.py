from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Document Assistant"
    API_V1_STR: str = "" # MVP often uses root or simple prefix
    
    # LLM Settings
    ANTHROPIC_API_KEY: str
    MODEL_NAME: str = "claude-3-haiku-20240307" # Haiku is good for MVP/testing
    
    # Vector DB Settings
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    
    # Ingestion Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
