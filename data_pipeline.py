"""Data pipeline: download FMA data and compute / cache CLAP audio embeddings.

This module is the single entry-point for all data-preparation work so that
notebooks never need to worry about whether the data or embeddings exist:

    audio_paths, audio_embs = data_pipeline.ensure_embeddings()

If ``data/fma_small/`` is missing the audio is downloaded first; if
``data/embeddings.npz`` is missing (or has gaps) the embeddings are computed
and saved.  Repeated calls are cheap — all steps are skipped when the data
are already in place.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

import utils

# ---------------------------------------------------------------------------
# FMA constants
# ---------------------------------------------------------------------------

FMA_SMALL_URL   = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_META_URL    = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

FMA_SMALL_SHA1  = "ade154f733639d52e35e32f5593efe5be76c6d70"
FMA_META_SHA1   = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"

FMA_SMALL_TRACK_COUNT = 8_000

FMA_META_FILES = ["tracks.csv", "genres.csv", "features.csv", "echonest.csv"]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _sha1_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    print(f"Downloading  {url}")
    print(f"  → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            gib = total_size / (1 << 30)
            print(f"\r  {pct:5.1f}%  ({gib:.2f} GiB total)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _verify(path: Path, expected_sha1: str) -> bool:
    print(f"Verifying checksum for {path.name} …", end=" ", flush=True)
    digest = _sha1_file(path)
    ok = digest == expected_sha1
    print("OK" if ok else f"MISMATCH (got {digest})")
    return ok


def _ensure_zip(zip_path: Path, url: str, sha1: str) -> None:
    if zip_path.exists():
        print(f"Found cached zip: {zip_path}")
        if not _verify(zip_path, sha1):
            print("Checksum failed — re-downloading …")
            zip_path.unlink()
            _download(url, zip_path)
            assert _verify(zip_path, sha1), "Checksum still failed after re-download!"
    else:
        _download(url, zip_path)
        assert _verify(zip_path, sha1), "Checksum failed after download!"


def _extract_zip(zip_path: Path, dest: Path) -> None:
    print(f"Extracting {zip_path.name} → {dest} …")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print("Extraction complete.")


def _count_mp3s(directory: Path) -> int:
    return sum(1 for _ in directory.rglob("*.mp3"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_fma_data(
    songs_dir: Path | str = utils.DEFAULT_SONGS_DIR,
    metadata_dir: Path | str = utils.DEFAULT_METADATA_DIR,
    temp_dir: Path | str = Path("temp"),
) -> None:
    """Download FMA Small audio and metadata if not already on disk.

    Both archives are SHA-1 verified; partial downloads are detected and
    re-fetched.  Zip files are deleted after successful extraction.
    """
    songs_dir    = Path(songs_dir)
    metadata_dir = Path(metadata_dir)
    temp_dir     = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    audio_zip = temp_dir / "fma_small.zip"
    meta_zip  = temp_dir / "fma_metadata.zip"

    # ── Audio ─────────────────────────────────────────────────────────────────
    mp3_count = _count_mp3s(songs_dir) if songs_dir.exists() else 0
    print(f"Audio tracks on disk: {mp3_count} / {FMA_SMALL_TRACK_COUNT}")

    if mp3_count >= FMA_SMALL_TRACK_COUNT:
        print("All tracks present — skipping audio download.")
    else:
        _ensure_zip(audio_zip, FMA_SMALL_URL, FMA_SMALL_SHA1)
        _extract_zip(audio_zip, songs_dir.parent)
        print(f"Tracks after extraction: {_count_mp3s(songs_dir)}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta_present = all((metadata_dir / f).exists() for f in FMA_META_FILES)

    if meta_present:
        print("Metadata already present — skipping metadata download.")
    else:
        _ensure_zip(meta_zip, FMA_META_URL, FMA_META_SHA1)
        _extract_zip(meta_zip, metadata_dir.parent)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    for zf in (audio_zip, meta_zip):
        if zf.exists():
            zf.unlink()


def compute_audio_embeddings(
    songs_dir: Path | str = utils.DEFAULT_SONGS_DIR,
    emb_out: Path | str = utils.DEFAULT_EMB_PATH,
    batch_size: int = 8,
    device: str | None = None,
    checkpoint_path: Path = utils.DEFAULT_CHECKPOINT_PATH,
    checkpoint_url: str = utils.DEFAULT_CHECKPOINT_URL,
) -> tuple[list[str], np.ndarray]:
    """Compute CLAP audio embeddings for all files in *songs_dir*, incrementally.

    Tracks already present in *emb_out* are skipped.  Each file is processed
    individually so a corrupt MP3 only skips itself.  Results are merged and
    saved back to *emb_out*.

    Returns:
        ``(audio_paths, audio_embs)`` — the complete merged embedding index.
    """
    import torch

    songs_dir = Path(songs_dir)
    emb_out   = Path(emb_out)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    all_files = utils.collect_audio_files(songs_dir)
    print(f"Found {len(all_files)} audio file(s) under {songs_dir}")

    if emb_out.exists():
        known_paths, known_embs = utils.load_embeddings(emb_out)
        known_set = set(known_paths)
    else:
        known_paths, known_embs = [], np.empty((0, 0), dtype=np.float32)
        known_set = set()
        print("No existing embeddings — computing from scratch.")

    new_files = [p for p in all_files if str(p) not in known_set]
    print(f"Already embedded : {len(known_paths)}")
    print(f"New to embed     : {len(new_files)}")

    if not new_files:
        print("Embeddings are up to date.")
        return known_paths, known_embs

    # Only load the model when there is actual work to do
    model = utils.load_clap_model(checkpoint_path, checkpoint_url, device=device)

    new_embs_chunks: list[np.ndarray] = []
    new_good_paths:  list[str]        = []
    skipped:         list[str]        = []
    total = len(new_files)

    for i, path in enumerate(new_files):
        pct = (i + 1) / total * 100
        print(f"\r  Embedding {i + 1}/{total}  ({pct:.0f}%)", end="", flush=True)
        try:
            emb = model.get_audio_embedding_from_filelist(x=[str(path)], use_tensor=False)
            emb = np.asarray(emb, dtype=np.float32)
            if emb.shape[0] == 0:
                raise ValueError("Empty embedding returned")
            new_embs_chunks.append(emb)
            new_good_paths.append(str(path))
        except Exception as exc:
            skipped.append(str(path))
            print(f"\n  [SKIP] {Path(path).name}  — {type(exc).__name__}: {exc}")

    print()

    if skipped:
        print(f"Skipped {len(skipped)} file(s) due to load errors.")

    if not new_embs_chunks:
        print("No new embeddings produced.")
        return known_paths, known_embs

    new_embs = utils.l2_normalize(np.vstack(new_embs_chunks))
    print(f"New embeddings shape: {new_embs.shape}")

    if known_paths:
        all_paths: list[str]  = known_paths + new_good_paths
        all_embs: np.ndarray  = np.vstack([known_embs, new_embs])
    else:
        all_paths = new_good_paths
        all_embs  = new_embs

    utils.save_embeddings(emb_out, all_paths, all_embs)
    return all_paths, all_embs


def ensure_embeddings(
    emb_path: Path | str = utils.DEFAULT_EMB_PATH,
    songs_dir: Path | str = utils.DEFAULT_SONGS_DIR,
    batch_size: int = 8,
    device: str | None = None,
    checkpoint_path: Path = utils.DEFAULT_CHECKPOINT_PATH,
    checkpoint_url: str = utils.DEFAULT_CHECKPOINT_URL,
    download_fma: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Ensure FMA data and embeddings are ready, then return them.

    Runs the full pipeline on first use; every step is skipped on subsequent
    calls if its output already exists.  Typical usage::

        audio_paths, audio_embs = data_pipeline.ensure_embeddings()

    Set ``download_fma=False`` if you manage the FMA data separately.

    Returns:
        ``(audio_paths, audio_embs)`` — complete embedding index ready for k-NN.
    """
    emb_path  = Path(emb_path)
    songs_dir = Path(songs_dir)

    if download_fma:
        ensure_fma_data(songs_dir=songs_dir)

    return compute_audio_embeddings(
        songs_dir=songs_dir,
        emb_out=emb_path,
        batch_size=batch_size,
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_url=checkpoint_url,
    )
