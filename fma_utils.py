"""FMA (Free Music Archive) metadata and genre-cluster analysis utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# FMA Medium includes all FMA Small tracks; Large includes both.
_FMA_SUBSETS: dict[str, tuple[str, ...]] = {
    "small":  ("small",),
    "medium": ("small", "medium"),
    "large":  ("small", "medium", "large"),
}


def load_fma_genre_map(
    fma_metadata_dir: Path | str,
    fma_size: str = "small",
) -> dict[int, str]:
    """Return ``{track_id: genre_name}`` for the given FMA subset.

    Reads the multi-level ``tracks.csv`` and filters by ``(set, subset)``.
    Track-ids are integers matching the MP3 stem (``000002.mp3`` → ``2``).
    """
    fma_size = fma_size.lower()
    if fma_size not in _FMA_SUBSETS:
        raise ValueError(
            f"fma_size must be one of {list(_FMA_SUBSETS)}; got {fma_size!r}"
        )

    tracks_csv = Path(fma_metadata_dir) / "tracks.csv"
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    subset_col = ("set", "subset")
    genre_col = ("track", "genre_top")

    valid_subsets = _FMA_SUBSETS[fma_size]
    mask = tracks[subset_col].isin(valid_subsets)
    filtered = tracks.loc[mask, genre_col].dropna()
    filtered = filtered[filtered != ""]

    return {int(tid): str(genre) for tid, genre in filtered.items()}


def _track_id_from_path(path: str | Path) -> int | None:
    """Extract the integer track-id from an FMA path (``000002.mp3`` → ``2``)."""
    try:
        return int(Path(path).stem)
    except ValueError:
        return None


def compute_genre_text_audio_alignment(
    model,
    genre_map: dict[int, str],
    audio_paths: list[str],
    audio_embs: np.ndarray,
    device: str,
) -> dict[str, float]:
    """Cosine similarity between each genre's text embedding and its mean audio embedding.

    Since only the text encoder changes during LoRA fine-tuning, comparing this
    metric before and after training shows per-genre alignment improvement.

    Returns ``{genre_name: cosine_similarity}``.
    """
    audio_embs = np.asarray(audio_embs, dtype=np.float32)

    genre_indices: dict[str, list[int]] = {}
    for i, path in enumerate(audio_paths):
        tid = _track_id_from_path(path)
        if tid is None:
            continue
        genre = genre_map.get(tid)
        if genre is None:
            continue
        genre_indices.setdefault(genre, []).append(i)

    if not genre_indices:
        return {}

    genres = sorted(genre_indices.keys())
    text_embs = model.get_text_embedding(genres, use_tensor=False)
    text_embs = np.asarray(text_embs, dtype=np.float32)
    norms = np.linalg.norm(text_embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    text_embs = text_embs / norms  # (G, D)

    results: dict[str, float] = {}
    for g_idx, genre in enumerate(genres):
        indices = genre_indices[genre]
        mean_audio = audio_embs[indices].mean(axis=0)
        norm = np.linalg.norm(mean_audio)
        if norm < 1e-9:
            continue
        mean_audio /= norm
        cosine_sim = float(text_embs[g_idx] @ mean_audio)
        results[genre] = cosine_sim

    return results


def compute_intra_genre_cosine_similarity(
    genre_map: dict[int, str],
    audio_paths: list[str],
    audio_embs: np.ndarray,
    max_per_genre: int = 500,
) -> dict[str, float]:
    """Mean pairwise cosine similarity between all tracks of the same genre.

    Pure audio-space clustering metric, independent of the text encoder.
    *max_per_genre* caps the sample size to keep the O(n²) cost manageable.

    Returns ``{genre_name: mean_pairwise_cosine_sim}``.
    """
    audio_embs = np.asarray(audio_embs, dtype=np.float32)

    genre_indices: dict[str, list[int]] = {}
    for i, path in enumerate(audio_paths):
        tid = _track_id_from_path(path)
        if tid is None:
            continue
        genre = genre_map.get(tid)
        if genre is None:
            continue
        genre_indices.setdefault(genre, []).append(i)

    results: dict[str, float] = {}
    rng = np.random.default_rng(seed=42)
    for genre, indices in sorted(genre_indices.items()):
        if len(indices) < 2:
            continue
        if len(indices) > max_per_genre:
            indices = rng.choice(indices, size=max_per_genre, replace=False).tolist()
        embs = audio_embs[indices]  # (n, D)
        gram = embs @ embs.T        # cosine similarities (embeddings are unit-norm)
        n = len(embs)
        mean_sim = (gram.sum() - np.trace(gram)) / (n * (n - 1))  # mean off-diagonal
        results[genre] = float(mean_sim)

    return results
