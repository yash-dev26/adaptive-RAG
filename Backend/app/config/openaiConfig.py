from openai import OpenAI
from app.config.server import config

openai_client = OpenAI(
    api_key=config["openai_api_key"]
)