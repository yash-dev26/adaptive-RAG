from qdrant_client import AsyncQdrantClient as QdrantClient
from app.config.server import config

qdrant_client = QdrantClient(
    url=config["qdrant_url"],
    api_key=config["qdrant_api_key"],
    cloud_inference=True,
)