from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np


def scramble_embedding(emb: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Add small Gaussian noise to an L2-normalised embedding, then re-normalise."""
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
    """KNN biased by two corpus anchors (best and worst match) for mild diversity."""
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
    slug = prompt.lower().strip()
    slug = re.sub(r"[^\w\s]", "", slug)
    slug = re.sub(r"\s+", "_", slug)
    slug = slug[:max_len].rstrip("_")
    return slug or "playlist"


def _unique_dir(root: Path, slug: str) -> Path:
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
    """Copy tracks into a new folder under output_root named after the prompt."""
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
