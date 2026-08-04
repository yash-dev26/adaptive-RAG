import asyncio
import time

from qdrant_client.http.models import FieldCondition, Filter, Range

from app.config.qdrantConfig import qdrant_client
from app.config.server import config


async def cleanup_semantic_cache(interval: int = 600):
    while True:
        try:
            collection_name = config.get("semantic_cache_collection_name")
            if not collection_name:
                print("[cleanup] semantic cache collection not configured; skipping")
                await asyncio.sleep(interval)
                continue

            current_time = int(time.time())

            print("[cleanup] running semantic cache cleanup...")

            await qdrant_client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="expires_at",
                            range=Range(lt=current_time),
                        )
                    ]
                ),
            )

            print("[cleanup] done")
        except Exception as e:
            print(f"[cleanup] error: {e}")

        await asyncio.sleep(interval)