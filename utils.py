"""
General-purpose helpers for the Playlist Generator project.

Provides:
  - CLAP checkpoint downloading and model loading
  - Audio file discovery
  - Embedding I/O (save / load .npz)
  - L2 normalisation
  - Cosine k-NN search
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import laion_clap


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_EXTS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".aiff", ".wma"}
)

DEFAULT_CHECKPOINT_URL = (
    "https://huggingface.co/lukewys/laion_clap/resolve/main/"
    "music_audioset_epoch_15_esc_90.14.pt"
)
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/music_audioset_epoch_15_esc_90.14.pt")


# ---------------------------------------------------------------------------
# CLAP helpers
# ---------------------------------------------------------------------------


def ensure_checkpoint(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    checkpoint_url: str = DEFAULT_CHECKPOINT_URL,
) -> Path:
    """Download the CLAP checkpoint if it is not already on disk."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        print(f"Downloading checkpoint → {checkpoint_path} …")

        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                pct = min(block_num * block_size / total_size * 100, 100)
                print(f"\r  {pct:5.1f}%", end="", flush=True)

        urllib.request.urlretrieve(checkpoint_url, checkpoint_path, reporthook=_progress)
        print()
    else:
        print(f"Checkpoint already present: {checkpoint_path}")
    return checkpoint_path


def load_clap_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    checkpoint_url: str = DEFAULT_CHECKPOINT_URL,
    device: str | None = None,
) -> laion_clap.CLAP_Module:
    """Load and return a ready-to-use CLAP music model.

    Downloads the checkpoint first if needed.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = ensure_checkpoint(checkpoint_path, checkpoint_url)
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    model.load_ckpt(str(ckpt))
    print(f"CLAP model loaded on {device}")
    return model


# ---------------------------------------------------------------------------
# Audio file helpers
# ---------------------------------------------------------------------------


def collect_audio_files(root: Path | str) -> list[Path]:
    """Return all audio files under *root*, sorted by path."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    return files


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise *x* along the last axis."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def save_embeddings(path: Path | str, file_list: Sequence[str], embeddings: np.ndarray) -> None:
    """Persist *file_list* and *embeddings* to a compressed .npz archive.

    Args:
        path:        Output file path (e.g. ``data/embeddings.npz``).
        file_list:   Ordered list of audio file paths (strings).
        embeddings:  2-D float32 array of shape ``(N, D)``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        paths=np.array(file_list, dtype=object),
        embeddings=embeddings.astype(np.float32),
    )
    print(f"Saved {len(file_list)} embeddings → {path}")


def load_embeddings(path: Path | str) -> tuple[list[str], np.ndarray]:
    """Load embeddings saved by :func:`save_embeddings`.

    Returns:
        A ``(file_list, embeddings)`` tuple where *file_list* is a list of
        audio path strings and *embeddings* is a float32 array of shape
        ``(N, D)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    data = np.load(path, allow_pickle=True)
    file_list = [str(p) for p in data["paths"]]
    embeddings = data["embeddings"].astype(np.float32)
    print(f"Loaded {len(file_list)} embeddings from {path}  shape={embeddings.shape}")
    return file_list, embeddings


# ---------------------------------------------------------------------------
# k-NN search
# ---------------------------------------------------------------------------


def knn_search(
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine nearest-neighbour search (both inputs must be L2-normalised).

    Args:
        query_emb:   1-D array of shape ``(D,)`` — the query vector.
        corpus_embs: 2-D array of shape ``(N, D)`` — the corpus.
        k:           Number of neighbours to return.

    Returns:
        ``(indices, scores)`` — arrays of length ``min(k, N)``, sorted by
        descending cosine similarity.
    """
    scores: np.ndarray = corpus_embs @ query_emb
    k = min(k, len(scores))
    top_idx: np.ndarray = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx, scores[top_idx]
