from __future__ import annotations

from typing import cast


def build_base_overrides(
    *,
    engine_name: str | None = None,
    embedder: str | None = None,
    index: str | None = None,
    device: str | None = None,
    env_embedder: str | None = None,
) -> dict[str, object]:
    overrides: dict[str, object] = {}
    embedder_overrides: dict[str, object] = {}

    if engine_name:
        overrides["engine"] = engine_name
    if embedder:
        embedder_overrides["name"] = embedder
    if index:
        overrides["index"] = {"name": index}
    if device:
        embedder_overrides["device"] = device
    if (env_embedder or "").strip().lower() == "stub" and "name" not in embedder_overrides:
        embedder_overrides["name"] = "stub"
    if embedder_overrides:
        overrides["embedder"] = embedder_overrides
    return overrides


def clean_overrides(overrides: dict[str, object]) -> dict[str, object]:
    cleaned: dict[str, object] = {}
    for key, value in overrides.items():
        if isinstance(value, dict):
            value_dict = cast(dict[str, object], value)
            filtered: dict[str, object] = {k: v for k, v in value_dict.items() if v is not None}
            if filtered:
                cleaned[key] = filtered
        elif value is not None:
            cleaned[key] = value
    return cleaned
