from pymongo.asynchronous.mongo_client import AsyncMongoClient
from app.config.server import config

_client: AsyncMongoClient | None = None

APP_DB_NAME = "adaptive_rag_app"


def get_mongo_client() -> AsyncMongoClient:
    global _client
    if _client is None:
        if not config["mongodb_uri"]:
            raise ValueError("MONGODB_URI is not set in the environment variables.")
        _client = AsyncMongoClient(config["mongodb_uri"])
    return _client


def get_app_db():
    return get_mongo_client()[APP_DB_NAME]