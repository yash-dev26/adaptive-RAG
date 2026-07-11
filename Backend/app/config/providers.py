from openai import OpenAI
from groq import Groq


def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)
