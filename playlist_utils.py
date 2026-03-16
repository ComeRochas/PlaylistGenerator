"""Playlist generation utilities: embedding randomisation and dual-anchor KNN."""

from __future__ import annotations

import numpy as np


def scramble_embedding(emb: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Add isotropic Gaussian noise to an L2-normalised embedding vector.

    The perturbed vector is re-normalised to the unit sphere so it remains
    compatible with cosine-similarity search.

    Args:
        emb: 1-D float32 unit vector of shape ``(D,)``.
        std: Standard deviation of the Gaussian noise.  Larger values produce
            more randomisation; ``0.05`` is a mild perturbation that keeps the
            direction close to the original.

    Returns:
        Perturbed unit vector of the same shape as *emb*.
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
    """K-nearest-neighbour search using two corpus anchors as diversity probes.

    Instead of ranking by direct cosine similarity between the query and each
    corpus track, this function:

    1. Computes cosine similarities between the query and all corpus tracks.
    2. Identifies the **max-anchor** (track most similar to the query) and the
       **min-anchor** (track least similar).
    3. Scores every corpus track as a weighted combination::

           score_i = 10 * cos(track_i, max_anchor) + 1 * cos(track_i, min_anchor)

       The heavy weight on the max-anchor keeps results relevant; the small
       weight on the min-anchor injects mild diversity.
    4. Returns the top-K tracks by this blended score.

    Both *query_emb* and *corpus_embs* must be L2-normalised (unit vectors)
    so that dot products equal cosine similarities.

    Args:
        query_emb: 1-D float32 unit vector of shape ``(D,)``.
        corpus_embs: 2-D float32 array of shape ``(N, D)``, L2-normalised.
        k: Number of tracks to return.

    Returns:
        ``(indices, scores)`` where *indices* is an int array of length *k*
        (sorted best-first) and *scores* are the corresponding blended scores.
    """
    query_emb = np.asarray(query_emb, dtype=np.float32)
    corpus_embs = np.asarray(corpus_embs, dtype=np.float32)

    # Step 1 – cosine similarities to query
    query_scores = corpus_embs @ query_emb  # (N,)

    # Step 2 – anchors
    i_max = int(np.argmax(query_scores))
    i_min = int(np.argmin(query_scores))

    # Step 3 – blended scores using anchor embeddings as probes
    sim_to_max = corpus_embs @ corpus_embs[i_max]  # (N,)
    sim_to_min = corpus_embs @ corpus_embs[i_min]  # (N,)
    blended = 10.0 * sim_to_max + 1.0 * sim_to_min  # (N,)

    # Step 4 – top-K selection
    k = min(k, len(blended))
    top_idx = np.argpartition(-blended, k - 1)[:k]
    order = np.argsort(-blended[top_idx])
    sorted_idx = top_idx[order]

    return sorted_idx, blended[sorted_idx]
