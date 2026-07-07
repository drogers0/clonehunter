import asyncio


async def fetch_data(url: str) -> dict:
    await asyncio.sleep(0)
    return {"url": url, "data": None}


async def process(items):
    results = []
    for item in items:
        result = await fetch_data(item)
        results.append(result)
    return results


async def gather_results(urls: list[str]) -> list[dict]:
    """Fetch all URLs concurrently."""
    tasks = [fetch_data(url) for url in urls]
    return await asyncio.gather(*tasks)
