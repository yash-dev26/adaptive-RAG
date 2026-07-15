from dotenv import load_dotenv
import os
load_dotenv()

config = {
    "qdrant_url": os.getenv("QDRANT_URL"),
    "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
    "qdrant_collection_name": os.getenv("QDRANT_COLLECTION_NAME"),
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "mongodb_uri": os.getenv("MONGODB_URI"),
    "semantic_cache_collection_name": os.getenv("SEMANTIC_CACHE_COLLECTION_NAME"),
    "CORS_ORIGINS": os.getenv("CORS_ORIGINS", "http://localhost:5174,http://localhost:5173,http://localhost:3000"),
    "redis_host": os.getenv("REDIS_HOST", "localhost"),
    "redis_port": os.getenv("REDIS_PORT", "6379"),
    "redis_password": os.getenv("REDIS_PASSWORD"),
    "redis_ssl": os.getenv("REDIS_SSL", "false"),
    "groq_api_key": os.getenv("GROQ_API_KEY"),
    "langsmith_api_key": os.getenv("LANGSMITH_API_KEY"),
    "langsmith_tracing": os.getenv("LANGSMITH_TRACING", "false"),
    "langsmith_project": os.getenv("LANGSMITH_PROJECT", "Adaptive RAG"),
}