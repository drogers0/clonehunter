def log(func):
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


def retry(n):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for _ in range(n):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    pass
            return None

        return wrapper

    return decorator


@log
@retry(3)
async def fetch(url: str) -> bytes:
    return b""


@log
async def post(url: str, data: bytes) -> int:
    """Post data to url, return status code."""
    return 200
