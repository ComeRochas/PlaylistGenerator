"""
General-purpose helpers for the Playlist Generator project.

Provides:
  - CLAP checkpoint downloading and model loading
  - Audio file discovery
  - Embedding I/O (save / load .npz)
  - L2 normalisation
  - Cosine k-NN search
  - Symmetric contrastive loss for fine-tuning
  - MusicCapsDataset for LoRA fine-tuning
  - LoRA injection into the CLAP text encoder
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
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


# ---------------------------------------------------------------------------
# Fine-tuning: contrastive loss
# ---------------------------------------------------------------------------


def symmetric_contrastive_loss(
    audio_embs: torch.Tensor,
    text_embs: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE / contrastive loss (as in CLIP / CLAP training).

    Args:
        audio_embs:  Float tensor of shape ``(N, D)``, L2-normalised.
        text_embs:   Float tensor of shape ``(N, D)``, L2-normalised.
        logit_scale: Scalar tensor — learned temperature (applied as ``exp(logit_scale)``).

    Returns:
        Scalar loss tensor.
    """
    audio_embs = F.normalize(audio_embs, p=2, dim=-1)
    text_embs = F.normalize(text_embs, p=2, dim=-1)

    scale = logit_scale.exp()
    logits_per_audio = scale * audio_embs @ text_embs.t()
    logits_per_text = logits_per_audio.t()

    labels = torch.arange(len(logits_per_audio), device=audio_embs.device)
    loss_a = F.cross_entropy(logits_per_audio, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_a + loss_t) / 2


# ---------------------------------------------------------------------------
# Fine-tuning: MusicCaps dataset
# ---------------------------------------------------------------------------


class MusicCapsDataset(Dataset):
    """PyTorch Dataset for LoRA fine-tuning on MusicCaps.

    Expects:
    - A pre-computed embeddings file (npz) produced by the fine-tuning
      notebook's pre-compute step (paths = ytid keys, embeddings = audio vecs).
    - A captions CSV with at least ``ytid`` and ``caption`` columns.

    ``__getitem__`` returns ``(audio_emb_tensor, caption_str)``.
    """

    def __init__(self, embeddings_path: Path | str, captions_csv: Path | str) -> None:
        embeddings_path = Path(embeddings_path)
        captions_csv = Path(captions_csv)

        data = np.load(embeddings_path, allow_pickle=True)
        self.embeddings: np.ndarray = data["embeddings"].astype(np.float32)
        ytids: list[str] = [str(p) for p in data["paths"]]

        captions_df = pd.read_csv(captions_csv).set_index("ytid")

        # Keep only rows that have both an embedding and a caption
        self.samples: list[tuple[int, str]] = []
        for i, ytid in enumerate(ytids):
            if ytid in captions_df.index:
                caption = str(captions_df.at[ytid, "caption"])
                self.samples.append((i, caption))

        print(
            f"MusicCapsDataset: {len(self.samples)} paired samples "
            f"(embeddings: {len(ytids)}, captions: {len(captions_df)})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        emb_idx, caption = self.samples[idx]
        audio_emb = torch.from_numpy(self.embeddings[emb_idx])
        return audio_emb, caption


# ---------------------------------------------------------------------------
# Fine-tuning: LoRA injection
# ---------------------------------------------------------------------------


def eval_audio_text_cosine_similarity(
    model: laion_clap.CLAP_Module,
    dataset: "MusicCapsDataset",
    sample_indices: list[int],
    device: str = "cpu",
    batch_size: int = 32,
) -> float:
    """Mean cosine similarity between paired audio and text embeddings.

    For each sample, the audio embedding is taken from the pre-computed cache
    and the text embedding is produced on-the-fly by the (optionally LoRA-adapted)
    text encoder.  Both are L2-normalised before the dot product, so the result
    is in [-1, 1] — higher is better.

    Args:
        model:          A :class:`laion_clap.CLAP_Module` (may have LoRA applied).
        dataset:        A :class:`MusicCapsDataset` instance.
        sample_indices: Indices into *dataset* to evaluate.
        device:         Torch device string.
        batch_size:     How many samples to process at once.

    Returns:
        Scalar mean cosine similarity over all *sample_indices*.
    """
    model.eval()
    all_sims: list[float] = []

    with torch.no_grad():
        for start in range(0, len(sample_indices), batch_size):
            idx_batch = sample_indices[start : start + batch_size]
            audio_embs = torch.stack([dataset[i][0] for i in idx_batch]).to(device)
            captions   = [dataset[i][1] for i in idx_batch]

            text_embs_np = model.get_text_embedding(captions, use_tensor=False)
            text_embs = torch.tensor(text_embs_np, dtype=torch.float32, device=device)

            # Both inputs normalised → dot product = cosine similarity
            a = F.normalize(audio_embs, p=2, dim=-1)
            t = F.normalize(text_embs,  p=2, dim=-1)
            sims = (a * t).sum(dim=-1)
            all_sims.extend(sims.cpu().tolist())

    return float(np.mean(all_sims))


def load_clap_model_with_lora(
    lora_ckpt_dir: Path | str,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    checkpoint_url: str = DEFAULT_CHECKPOINT_URL,
    device: str | None = None,
) -> laion_clap.CLAP_Module:
    """Load a CLAP model and apply saved LoRA adapter weights to its text encoder.

    Args:
        lora_ckpt_dir:   Directory written by ``model.model.text_branch.save_pretrained()``.
        checkpoint_path: Base CLAP checkpoint (default music model).
        checkpoint_url:  URL used to download the checkpoint if absent.
        device:          Torch device string; auto-detected if *None*.

    Returns:
        A :class:`laion_clap.CLAP_Module` whose text branch carries the LoRA adapters.
    """
    from peft import PeftModel

    model = load_clap_model(checkpoint_path, checkpoint_url, device=device)

    lora_ckpt_dir = Path(lora_ckpt_dir)
    if not lora_ckpt_dir.exists():
        raise FileNotFoundError(
            f"LoRA checkpoint directory not found: {lora_ckpt_dir}\n"
            "Run finetune_clap.ipynb first to generate it."
        )

    model.model.text_branch = PeftModel.from_pretrained(
        model.model.text_branch,
        str(lora_ckpt_dir),
    )
    model.model.text_branch.eval()
    print(f"LoRA adapter loaded from {lora_ckpt_dir}")
    return model


def inject_lora_text_encoder(
    model: laion_clap.CLAP_Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> None:
    """Freeze the CLAP audio encoder and inject LoRA into the text encoder.

    Modifies *model* in-place: replaces ``model.model.text_branch`` with a
    PEFT LoRA model and sets ``requires_grad=False`` on all audio-branch
    parameters.  Only LoRA adapter weights and ``logit_scale`` remain
    trainable.

    Args:
        model:          A loaded :class:`laion_clap.CLAP_Module`.
        r:              LoRA rank.
        lora_alpha:     LoRA alpha (scaling factor).
        lora_dropout:   Dropout applied inside LoRA layers.
        target_modules: Attention projection names to target.
                        Defaults to ``["query", "value"]`` (RoBERTa naming).
    """
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["query", "value"]

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad_(False)

    # Apply LoRA to the text branch
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    model.model.text_branch = get_peft_model(model.model.text_branch, lora_config)

    # Unfreeze text_projection — it maps text_branch output into the shared
    # embedding space and its pooler weights are randomly initialized in this
    # checkpoint (marked MISSING in the load report), so it benefits from tuning.
    if hasattr(model.model, "text_projection"):
        for param in model.model.text_projection.parameters():
            param.requires_grad_(True)

    # Keep logit_scale trainable
    if hasattr(model.model, "logit_scale_a"):
        model.model.logit_scale_a.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA injected — trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
