"""Playlist generation utilities: embedding randomisation, dual-anchor KNN, and export."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np


def scramble_embedding(emb: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Add isotropic Gaussian noise to an L2-normalised embedding, then re-normalise.

    ``std=0.05`` is a mild perturbation that keeps the direction close to the
    original while making repeated queries return slightly different playlists.
    """
    noise = np.random.normal(0.0, std, emb.shape).astype(emb.dtype)
    perturbed = emb + noise
    norm = np.linalg.norm(perturbed)
    if norm < 1e-9:
        return emb.copy()
    return perturbed / norm


def dual_anchor_knn(
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """K-nearest-neighbour search biased by two corpus anchors.

    Scores each track as ``10·cos(track, best_match) + 1·cos(track, worst_match)``
    where best/worst are the corpus tracks most and least similar to the query.
    This keeps results relevant while injecting mild diversity.

    Both inputs must be L2-normalised so dot products equal cosine similarities.

    Returns:
        ``(indices, scores)`` sorted best-first, length ``min(k, N)``.
    """
    query_emb = np.asarray(query_emb, dtype=np.float32)
    corpus_embs = np.asarray(corpus_embs, dtype=np.float32)

    query_scores = corpus_embs @ query_emb  # (N,)

    i_max = int(np.argmax(query_scores))
    i_min = int(np.argmin(query_scores))

    sim_to_max = corpus_embs @ corpus_embs[i_max]  # (N,)
    sim_to_min = corpus_embs @ corpus_embs[i_min]  # (N,)
    blended = 10.0 * sim_to_max + 1.0 * sim_to_min  # (N,)

    k = min(k, len(blended))
    top_idx = np.argpartition(-blended, k - 1)[:k]
    order = np.argsort(-blended[top_idx])
    sorted_idx = top_idx[order]

    return sorted_idx, blended[sorted_idx]


def _prompt_slug(prompt: str, max_len: int = 48) -> str:
    """Turn a free-text prompt into a safe directory name."""
    slug = prompt.lower().strip()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    slug = slug[:max_len].rstrip("_")
    return slug or "playlist"


def _unique_dir(root: Path, slug: str) -> Path:
    """Return ``root/slug``, or ``root/slug_2``, ``root/slug_3``, … if taken."""
    candidate = root / slug
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = root / f"{slug}_{n}"
        if not candidate.exists():
            return candidate
        n += 1


def save_playlist(
    prompt: str,
    track_paths: list[str],
    output_root: Path | str = Path("playlists"),
    cover_image=None,
) -> Path:
    """Copy playlist tracks (and optionally a cover image) into a new folder.

    The folder is created under *output_root* with a name derived from the
    prompt (spaces → underscores, special characters stripped).  If a folder
    with that name already exists a numeric suffix is appended (``_2``, ``_3``…).

    Track filenames are prefixed with their rank so they sort correctly in any
    file manager (``01_track.mp3``, ``02_track.mp3``, …).

    Args:
        prompt:       The text prompt used to generate the playlist.
        track_paths:  Ordered list of source audio file paths (best-first).
        output_root:  Parent directory for all playlists (default: ``playlists/``).
        cover_image:  Either a ``PIL.Image`` or a ``Path`` to an image file.
                      Saved as ``cover.jpg`` inside the playlist folder.

    Returns:
        Path to the newly created playlist folder.
    """
    output_root = Path(output_root)
    playlist_dir = _unique_dir(output_root, _prompt_slug(prompt))
    playlist_dir.mkdir(parents=True)

    width = len(str(len(track_paths)))
    for rank, src in enumerate(track_paths, start=1):
        dst = playlist_dir / f"{rank:0{width}d}_{Path(src).name}"
        shutil.copy2(src, dst)

    if cover_image is not None:
        if isinstance(cover_image, Path):
            shutil.copy2(cover_image, playlist_dir / cover_image.name)
        else:
            cover_image.save(playlist_dir / "cover.jpg")

    print(f"Playlist saved → {playlist_dir}  ({len(track_paths)} tracks)")
    return playlist_dir
