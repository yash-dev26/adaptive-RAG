from pymongo import MongoClient
from app.config.server import config

_client: MongoClient | None = None

APP_DB_NAME = "adaptive_rag_app"


def get_mongo_client() -> MongoClient:
    global _client
    if _client is None:
        if not config["mongodb_uri"]:
            raise ValueError("MONGODB_URI is not set in the environment variables.")
        _client = MongoClient(config["mongodb_uri"])
    return _client


def get_app_db():
    return get_mongo_client()[APP_DB_NAME]