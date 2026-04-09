from dotenv import load_dotenv
import os
load_dotenv()

config = {
    "qdrant_url": os.getenv("QDRANT_URL"),
    "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
    "qdrant_collection_name": os.getenv("QDRANT_COLLECTION_NAME"),
}