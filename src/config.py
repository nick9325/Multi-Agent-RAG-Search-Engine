import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Fix Windows pagefile mmap memory issue (OS Error 1455)
os.environ["SAFETENSORS_DISABLE_MMAP"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # LLM Settings (Groq API)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    PRIMARY_LLM_MODEL: str = os.getenv("PRIMARY_LLM_MODEL", "openai/gpt-oss-120b")
    FAST_LLM_MODEL: str = os.getenv("FAST_LLM_MODEL", "openai/gpt-oss-20b")


    # Local Storage & Models
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))
    BM25_PERSIST_PATH: str = os.getenv("BM25_PERSIST_PATH", str(BASE_DIR / "data" / "bm25.pkl"))
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    CROSS_ENCODER_MODEL_NAME: str = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Search & Reranking Hyperparameters
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "4"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    CROSS_ENCODER_THRESHOLD: float = float(os.getenv("CROSS_ENCODER_THRESHOLD", "-5.0"))

    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()

