# Пример с юникодом / Example with unicode
def вычислить(значение: float, коэффициент: float = 1.0) -> float:
    """Вычислить результат."""
    return значение * коэффициент


def calculate(value: float) -> float:
    # Arithmetic operation → result
    return value * 2.0


def normalize_text(text: str) -> str:
    """Normalize unicode text for comparison."""
    import unicodedata

    return unicodedata.normalize("NFC", text.strip().lower())
