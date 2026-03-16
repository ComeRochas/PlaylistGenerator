"""FMA (Free Music Archive) metadata and genre-cluster analysis utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

# Tracks whose (set, subset) value belongs to each logical FMA split.
# FMA Medium includes all tracks present in FMA Small.
_FMA_SUBSETS: dict[str, tuple[str, ...]] = {
    "small":  ("small",),
    "medium": ("small", "medium"),
    "large":  ("small", "medium", "large"),
}


def load_fma_genre_map(
    fma_metadata_dir: Path | str,
    fma_size: str = "small",
) -> dict[int, str]:
    """Return a mapping from FMA track-id to top-level genre.

    The ``tracks.csv`` shipped with the FMA metadata archive uses a two-level
    column header.  The columns of interest are ``('set', 'subset')`` (which
    FMA split the track belongs to) and ``('track', 'genre_top')`` (the single
    top-level genre label).

    Args:
        fma_metadata_dir: Directory containing ``tracks.csv``
            (e.g. ``data/fma_metadata``).
        fma_size: One of ``'small'``, ``'medium'``, or ``'large'``.
            FMA Medium includes all FMA Small tracks; FMA Large includes
            both.

    Returns:
        ``{track_id: genre_name}`` for tracks that have a non-empty
        ``genre_top`` value.  Track-ids are plain integers (matching the
        numeric part of the MP3 filename, e.g. ``000002.mp3`` → ``2``).
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
    """Extract the integer track-id from an FMA MP3 path.

    FMA stores files as ``<dir>/<zero-padded-6-digit-id>.mp3``, e.g.
    ``data/fma_small/000/000002.mp3`` → ``2``.
    """
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
    """For each genre, measure how well CLAP text embeddings align with audio.

    The metric is the cosine similarity between the text embedding of the
    genre name and the **mean** L2-normalised audio embedding of all tracks
    belonging to that genre.  Because only the text encoder is updated during
    LoRA fine-tuning, running this function before and after training reveals
    whether alignment improved.

    Args:
        model: A loaded CLAP model (base or LoRA-adapted).
        genre_map: ``{track_id: genre_name}`` from :func:`load_fma_genre_map`.
        audio_paths: Ordered list of file paths corresponding to rows of
            *audio_embs*.
        audio_embs: Float32 array of shape ``(N, D)``, L2-normalised audio
            embeddings.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        ``{genre_name: cosine_similarity}`` for genres with at least one
        track present in *audio_paths*.
    """
    audio_embs = np.asarray(audio_embs, dtype=np.float32)

    # Map each corpus path to its genre (skip paths without a known id/genre)
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

    # Compute a single text embedding per genre name
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
        mean_audio /= norm  # unit vector
        cosine_sim = float(text_embs[g_idx] @ mean_audio)
        results[genre] = cosine_sim

    return results


def compute_intra_genre_cosine_similarity(
    genre_map: dict[int, str],
    audio_paths: list[str],
    audio_embs: np.ndarray,
    max_per_genre: int = 500,
) -> dict[str, float]:
    """Mean pairwise cosine similarity of tracks within the same genre.

    This is a pure audio-space clustering metric (independent of the text
    encoder).  A higher value means genre tracks are tightly clustered in the
    CLAP audio embedding space.

    Args:
        genre_map: ``{track_id: genre_name}`` from :func:`load_fma_genre_map`.
        audio_paths: Ordered list of file paths corresponding to rows of
            *audio_embs*.
        audio_embs: Float32 array of shape ``(N, D)``, L2-normalised.
        max_per_genre: Cap on the number of tracks sampled per genre to keep
            computation tractable (pairwise cost is O(n²)).

    Returns:
        ``{genre_name: mean_pairwise_cosine_sim}`` for genres with at least
        two tracks in the corpus.
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
        # Gram matrix: cosine similarities (embeddings already unit-norm)
        gram = embs @ embs.T  # (n, n)
        n = len(embs)
        # Mean of off-diagonal entries
        mean_sim = (gram.sum() - np.trace(gram)) / (n * (n - 1))
        results[genre] = float(mean_sim)

    return results
