"""Unit tests for embedding-loader behavior.

Purpose:
    This module verifies cached model resolution, network fallback behavior,
    strict error handling, and sparse-only fallback behavior for
    `llm_rag/iii_vector_db/embedding_loader.py`.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch


SOURCE_DIR = Path(__file__).resolve().parents[2] / "iii_vector_db"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

fake_huggingface_hub = types.ModuleType("huggingface_hub")
fake_huggingface_hub.snapshot_download = lambda *args, **kwargs: "fake-model-path"
sys.modules.setdefault("huggingface_hub", fake_huggingface_hub)

fake_langchain_huggingface = types.ModuleType("langchain_huggingface")


class FakeHuggingFaceEmbeddings:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


fake_langchain_huggingface.HuggingFaceEmbeddings = FakeHuggingFaceEmbeddings
sys.modules.setdefault("langchain_huggingface", fake_langchain_huggingface)

sys.modules.pop("embedding_loader", None)
el = importlib.import_module("embedding_loader")


def test_resolve_embedding_model_path_prefers_cached_snapshot_first():
    with patch.object(el, "snapshot_download", side_effect=["/cached/model"]) as mock_snapshot:
        result = el.resolve_embedding_model_path("BAAI/bge-m3")

    assert result == "/cached/model"
    assert mock_snapshot.call_count == 1
    assert mock_snapshot.call_args.kwargs["repo_id"] == "BAAI/bge-m3"
    assert mock_snapshot.call_args.kwargs["local_files_only"] is True


def test_resolve_embedding_model_path_falls_back_to_network_lookup():
    with patch.object(
        el,
        "snapshot_download",
        side_effect=[RuntimeError("cache miss"), "/downloaded/model"],
    ) as mock_snapshot:
        result = el.resolve_embedding_model_path("BAAI/bge-m3")

    assert result == "/downloaded/model"
    assert mock_snapshot.call_count == 2
    assert mock_snapshot.call_args_list[0].kwargs["local_files_only"] is True
    assert mock_snapshot.call_args_list[1].kwargs["local_files_only"] is False


def test_resolve_embedding_model_path_raises_after_both_attempts_fail():
    with patch.object(
        el,
        "snapshot_download",
        side_effect=[RuntimeError("cache miss"), RuntimeError("network miss")],
    ):
        try:
            el.resolve_embedding_model_path("BAAI/bge-m3")
        except RuntimeError as exc:
            assert "Could not resolve embedding model 'BAAI/bge-m3'." in str(exc)
        else:
            raise AssertionError("Expected resolve_embedding_model_path to raise RuntimeError.")


def test_build_huggingface_embeddings_builds_wrapper_from_resolved_path():
    with patch.object(el, "resolve_embedding_model_path", return_value="/resolved/model") as mock_resolve:
        with patch.object(el, "HuggingFaceEmbeddings", FakeHuggingFaceEmbeddings):
            embeddings = el.build_huggingface_embeddings(
                "BAAI/bge-m3",
                device="cpu",
                normalize_embeddings=False,
                strict=True,
            )

    assert isinstance(embeddings, FakeHuggingFaceEmbeddings)
    assert embeddings.kwargs["model_name"] == "/resolved/model"
    assert embeddings.kwargs["model_kwargs"] == {"device": "cpu", "local_files_only": True}
    assert embeddings.kwargs["encode_kwargs"] == {"normalize_embeddings": False}
    mock_resolve.assert_called_once_with("BAAI/bge-m3")


def test_build_huggingface_embeddings_reraises_when_strict():
    with patch.object(el, "resolve_embedding_model_path", side_effect=RuntimeError("boom")):
        try:
            el.build_huggingface_embeddings("BAAI/bge-m3", strict=True)
        except RuntimeError as exc:
            assert str(exc) == "boom"
        else:
            raise AssertionError("Expected build_huggingface_embeddings to re-raise in strict mode.")


def test_build_huggingface_embeddings_returns_none_and_logs_when_not_strict():
    with patch.object(el, "resolve_embedding_model_path", side_effect=RuntimeError("boom")):
        with patch.object(el.logger, "warning") as mock_warning:
            embeddings = el.build_huggingface_embeddings("BAAI/bge-m3", strict=False)

    assert embeddings is None
    mock_warning.assert_called_once()
    assert "sparse-only retrieval" in mock_warning.call_args.args[0]
