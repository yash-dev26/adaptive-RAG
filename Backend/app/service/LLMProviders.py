from app.config.providers import get_openai_client, get_groq_client
from app.config.models import OPENAI_DEFAULT_MODEL, GROQ_FAST_MODEL

def generate_completion(
    provider: str,
    messages: list,
    openai_api_key: str,
    groq_api_key: str | None = None,
    model: str | None = None,
    temperature: float = 0,
    response_format=None,
):
    if not openai_api_key:
        raise ValueError("openai_api_key is required (used as the fallback provider too)")

    # No Groq key supplied -> don't even attempt Groq, go straight to OpenAI.
    if provider == "groq" and not groq_api_key:
        # `model` here (if set) was a Groq-hosted model id
        # Drop it so we fall through to OPENAI_DEFAULT_MODEL 
        provider = "openai"
        model = None

    if provider == "openai":
        client = get_openai_client(openai_api_key)
        response = client.chat.completions.create(
            model=model or OPENAI_DEFAULT_MODEL,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        return response.choices[0].message.content

    elif provider == "groq":
        try:
            client = get_groq_client(groq_api_key)
            response = client.chat.completions.create(
                model=model or GROQ_FAST_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"[groq fallback] {e}")
            client = get_openai_client(openai_api_key)
            response = client.chat.completions.create(
                model=OPENAI_DEFAULT_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content

    raise ValueError(f"Unsupported provider: {provider}")