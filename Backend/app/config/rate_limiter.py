from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request


def session_or_ip_key(request: Request) -> str:
    """
    Now that there's no JWT `sub` to key off of, rate limit per BYOK
    session id when the client sends one (X-Session-Id), falling back to
    IP for requests that arrive without it (e.g. malformed clients).
    """
    session_id = request.headers.get("X-Session-Id")
    return session_id or get_remote_address(request)


# Centralized limiter instance to avoid circular imports
limiter = Limiter(key_func=session_or_ip_key)
