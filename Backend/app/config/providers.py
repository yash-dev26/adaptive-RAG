from openai import OpenAI
from groq import Groq
from langsmith.wrappers import wrap_openai


def get_openai_client(api_key: str) -> OpenAI:
    client = OpenAI(api_key=api_key)
    return wrap_openai(client)


def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)
