from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass
from typing import Any

from clonehunter.core.types import Embedding, SnippetRef


@dataclass(frozen=True, slots=True)
class CodeBertConfig:
    model_name: str
    revision: str
    max_length: int
    batch_size: int
    device: str


class CodeBertEmbedder:
    def __init__(self, config: CodeBertConfig) -> None:
        self._config = config
        import os

        import torch
        import transformers as _transformers

        self._torch: Any = torch
        transformers: Any = _transformers
        with contextlib.suppress(Exception):
            logging_module: Any = getattr(transformers, "logging", None)
            if logging_module is not None:
                logging_module.set_verbosity_error()
        with contextlib.suppress(Exception):
            logging_utils = importlib.import_module("transformers.utils.logging")
            raw_disable = getattr(logging_utils, "disable_progress_bar", None)
            if callable(raw_disable):
                raw_disable()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        with contextlib.suppress(Exception):
            import huggingface_hub as _hub

            hub_logging: Any = getattr(_hub, "logging", None)
            if hub_logging is not None:
                hub_logging.set_verbosity_error()
        auto_tokenizer = transformers.AutoTokenizer
        auto_model = transformers.AutoModel
        self._tokenizer: Any = auto_tokenizer.from_pretrained(
            config.model_name,
            revision=config.revision,
            use_fast=True,
        )
        self._model: Any = auto_model.from_pretrained(
            config.model_name,
            revision=config.revision,
        ).to(config.device)
        self._model.eval()

    @property
    def dim(self) -> int:
        return int(self._model.config.hidden_size)

    def embed(self, snippets: list[SnippetRef]) -> list[Embedding]:
        vectors: list[Embedding] = []
        if not snippets:
            return vectors

        with self._torch.no_grad():
            for start in range(0, len(snippets), self._config.batch_size):
                batch = snippets[start : start + self._config.batch_size]
                inputs: Any = self._tokenizer(
                    [snip.text for snip in batch],
                    padding=True,
                    truncation=True,
                    max_length=self._config.max_length,
                    return_tensors="pt",
                ).to(self._config.device)
                outputs: Any = self._model(**inputs)
                last_hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                masked = last_hidden * mask
                summed = masked.sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                pooled = summed / counts
                for row in pooled.cpu().tolist():
                    vectors.append(Embedding(vector=row, dim=len(row)))
        return vectors
