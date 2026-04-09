from app.config.qdrantConfig import qdrant_client
from app.config.server import config

if config["qdrant_collection_name"] not in [collection.name for collection in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=config["qdrant_collection_name"],
        vectors_config={
            "size": 384,
            "distance": "Cosine"
        }
    )
else:
    print(f"Collection '{config['qdrant_collection_name']}' exists.")

    
async def store_in_qdrant(collection_name: str, data: list[dict], file_id: str, user_id: str):
    qdrant_client.upsert(
        collection_name=collection_name,
        points=data
    )   
    return {"status": "success", "message": f"Data stored in collection '{collection_name}'.", "file_id": file_id}

async def search_data(collection_name: str, query_vector: list[float], top_k: int = 5):
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return search_result




