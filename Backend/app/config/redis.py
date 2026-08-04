import redis
from app.config.server import config

redis_client = redis.asyncio.Redis(
    host=config["redis_host"],
    port=int(config["redis_port"]),
    password=config["redis_password"],
    ssl=True,
    ssl_cert_reqs=None,
    decode_responses=True,
)
