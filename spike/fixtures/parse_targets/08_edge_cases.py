def process_walrus(data: list[int]) -> list[int]:
    return [y for x in data if (y := x * 2) > 0]


def process_match(status: str) -> str:
    match status:
        case "ok":
            return "success"
        case "err":
            return "failure"
        case _:
            return "unknown"


def complex_signature(
    first: str,
    second: int = 0,
    *args: float,
    keyword_only: bool = False,
    **kwargs: str,
) -> dict[str, int]:
    return {"first": len(first), "second": second}


def nested_walrus(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [item for item in items if (item not in seen) and not seen.add(item)]
