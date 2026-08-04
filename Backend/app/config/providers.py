from openai import AsyncOpenAI
from groq import AsyncGroq
from langsmith.wrappers import wrap_openai


def get_openai_client(api_key: str) -> AsyncOpenAI:
    client = AsyncOpenAI(api_key=api_key)
    return wrap_openai(client)


def get_groq_client(api_key: str) -> AsyncGroq:
    return AsyncGroq(api_key=api_key)
