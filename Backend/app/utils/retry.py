"""
Shared retry utilities for external API calls.

Use this for any outbound call to a third-party service that can fail
transiently: OpenAI (embeddings, chat), Cohere (rerank), Groq (chat),
Qdrant (vector store), etc.

Usage:
    from app.utils.retry import external_api_retry

    @external_api_retry()
    def _embed_batch(batch_texts, openai_api_key):
        ...

    @external_api_retry(max_attempts=2)  # override defaults per-call-site
    async def rerank_chunks(...):
        ...
"""

import logging
from typing import Callable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Import guards: not every service's SDK is installed in every module,
# so we resolve exception types lazily and fall back gracefully.
def _collect_retryable_exceptions() -> tuple:
    exc_types = []

    try:
        from openai import RateLimitError as OpenAIRateLimitError
        from openai import APIConnectionError as OpenAIAPIConnectionError
        from openai import APIStatusError as OpenAIAPIStatusError
        exc_types += [OpenAIRateLimitError, OpenAIAPIConnectionError, OpenAIAPIStatusError]
    except ImportError:
        pass

    try:
        from cohere.errors import (
            TooManyRequestsError as CohereTooManyRequestsError,
            InternalServerError as CohereInternalServerError,
            ServiceUnavailableError as CohereServiceUnavailableError,
        )
        exc_types += [
            CohereTooManyRequestsError,
            CohereInternalServerError,
            CohereServiceUnavailableError,
        ]
    except ImportError:
        pass

    try:
        from qdrant_client.http.exceptions import (
            ResponseHandlingException as QdrantResponseHandlingException,
            UnexpectedResponse as QdrantUnexpectedResponse,
        )
        exc_types += [QdrantResponseHandlingException, QdrantUnexpectedResponse]
    except ImportError:
        pass

    try:
        import httpx
        exc_types += [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError]
    except ImportError:
        pass

    return tuple(exc_types)


_RETRYABLE_EXCEPTIONS = _collect_retryable_exceptions()


def _is_retryable(exc: BaseException) -> bool:
    """
    Decide whether an exception from an external API call is worth retrying.

    Retry: rate limits, connection errors, 5xx server errors, timeouts.
    Don't retry: 4xx client errors (bad key, bad request, not found) —
    these won't fix themselves and just burn the user's rate-limit budget
    and add latency to a request that's going to fail anyway.
    """
    # Anything with a status_code attribute (OpenAI/Cohere style): only
    # retry on 5xx or 429. 429s are usually already their own exception
    # type, but some SDKs surface them as a generic status error too.
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code >= 500 or status_code == 429

    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True

    return False


def external_api_retry(
    max_attempts: int = 3,
    min_wait: float = 1,
    max_wait: float = 8,
) -> Callable:
    """
    Decorator factory for retrying external API calls with exponential
    backoff. Works on both sync and async functions (tenacity handles both
    transparently as of tenacity>=8).

    Args:
        max_attempts: total attempts including the first (default 3)
        min_wait: minimum backoff in seconds (default 1)
        max_wait: maximum backoff in seconds (default 8)
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )