from __future__ import annotations

import logging
from typing import Any

from huggingface_hub import snapshot_download
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def resolve_embedding_model_path(model_name: str) -> str:
    """Resolve an embedding model to a local snapshot directory.

    This helper tries two passes:
    - a cache-only lookup first, to avoid unnecessary network calls
    - a network-enabled lookup second, if the model is not already cached

    It raises a ``RuntimeError`` only after both attempts fail.
    """
    last_error: Exception | None = None

    for local_files_only in (True, False):
        try:
            return snapshot_download(
                repo_id=model_name,
                local_files_only=local_files_only,
                max_workers=1,
            )
        except Exception as exc:  # pragma: no cover - exercised against real environment/network
            last_error = exc

    raise RuntimeError(f"Could not resolve embedding model '{model_name}'.") from last_error


def build_huggingface_embeddings(
    model_name: str,
    *,
    device: str = "cpu",
    normalize_embeddings: bool = True,
    strict: bool = True,
) -> HuggingFaceEmbeddings | None:
    """Build a Hugging Face embedding wrapper for dense retrieval.

    This helper resolves the model to a local snapshot path first, then builds
    ``HuggingFaceEmbeddings`` with ``local_files_only=True`` so later use stays
    on disk.

    When ``strict`` is False, an embedding-load failure does not raise. Instead
    it logs a warning and returns ``None`` so inference can fall back to
    sparse-only retrieval.
    """
    try:
        model_path = resolve_embedding_model_path(model_name)
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={
                "device": device,
                "local_files_only": True,
            },
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )
    except Exception as exc:  # pragma: no cover - exercised against real environment/network
        if strict:
            raise
        logger.warning("Falling back to sparse-only retrieval because embeddings could not be loaded: %s", exc)
        return None
