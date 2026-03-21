from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import laion_clap

AUDIO_EXTS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".aiff", ".wma"}
)

DEFAULT_CHECKPOINT_URL = (
    "https://huggingface.co/lukewys/laion_clap/resolve/main/"
    "music_audioset_epoch_15_esc_90.14.pt"
)
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/music_audioset_epoch_15_esc_90.14.pt")
DEFAULT_SONGS_DIR = Path("data/fma_small")
DEFAULT_METADATA_DIR = Path("data/fma_metadata")
DEFAULT_EMB_PATH = Path("data/embeddings.npz")


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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = ensure_checkpoint(checkpoint_path, checkpoint_url)
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
    model.load_ckpt(str(ckpt))
    print(f"CLAP model loaded on {device}")
    return model


def collect_audio_files(root: Path | str) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    return files


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def save_embeddings(path: Path | str, file_list: Sequence[str], embeddings: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        paths=np.array(file_list, dtype=object),
        embeddings=embeddings.astype(np.float32),
    )
    print(f"Saved {len(file_list)} embeddings → {path}")


def load_embeddings(path: Path | str) -> tuple[list[str], np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    data = np.load(path, allow_pickle=True)
    file_list = [str(p) for p in data["paths"]]
    embeddings = data["embeddings"].astype(np.float32)
    print(f"Loaded {len(file_list)} embeddings from {path}  shape={embeddings.shape}")
    return file_list, embeddings


def knn_search(
    query_emb: np.ndarray,
    corpus_embs: np.ndarray,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    scores: np.ndarray = corpus_embs @ query_emb
    k = min(k, len(scores))
    top_idx: np.ndarray = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx, scores[top_idx]


def symmetric_contrastive_loss(
    audio_embs: torch.Tensor,
    text_embs: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    audio_embs = F.normalize(audio_embs, p=2, dim=-1)
    text_embs = F.normalize(text_embs, p=2, dim=-1)

    scale = logit_scale.exp()
    logits_per_audio = scale * audio_embs @ text_embs.t()
    logits_per_text = logits_per_audio.t()

    labels = torch.arange(len(logits_per_audio), device=audio_embs.device)
    loss_a = F.cross_entropy(logits_per_audio, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_a + loss_t) / 2


class MusicCapsDataset(Dataset):

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


def eval_audio_text_cosine_similarity(
    model: laion_clap.CLAP_Module,
    dataset: "MusicCapsDataset",
    sample_indices: list[int],
    device: str = "cpu",
    batch_size: int = 32,
) -> float:
    model.eval()
    all_sims: list[float] = []

    with torch.no_grad():
        for start in range(0, len(sample_indices), batch_size):
            idx_batch = sample_indices[start : start + batch_size]
            audio_embs = torch.stack([dataset[i][0] for i in idx_batch]).to(device)
            captions = [dataset[i][1] for i in idx_batch]

            text_embs_np = model.get_text_embedding(captions, use_tensor=False)
            text_embs = torch.tensor(text_embs_np, dtype=torch.float32, device=device)

            a = F.normalize(audio_embs, p=2, dim=-1)
            t = F.normalize(text_embs, p=2, dim=-1)
            sims = (a * t).sum(dim=-1)
            all_sims.extend(sims.cpu().tolist())

    return float(np.mean(all_sims))


def load_clap_model_with_lora(
    lora_ckpt_dir: Path | str,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    checkpoint_url: str = DEFAULT_CHECKPOINT_URL,
    device: str | None = None,
) -> laion_clap.CLAP_Module:
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


class MusicCapsRawDataset(Dataset):
    """Dataset returning (waveform, caption) pairs for audio-encoder fine-tuning.

    Audio is resampled to 48 kHz, mono, and padded/trimmed to 10 s.
    """

    TARGET_SR = 48_000
    TARGET_SAMPLES = 480_000  # 10 s @ 48 kHz

    def __init__(self, wav_dir: Path | str, captions_csv: Path | str) -> None:
        wav_dir = Path(wav_dir)
        captions_df = pd.read_csv(captions_csv)

        self.samples: list[tuple[Path, str]] = []
        for _, row in captions_df.iterrows():
            p = wav_dir / f"{row['ytid']}.wav"
            if p.exists():
                self.samples.append((p, str(row["caption"])))

        print(f"MusicCapsRawDataset: {len(self.samples)} WAV clips found in {wav_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path, caption = self.samples[idx]
        waveform, sr = torchaudio.load(str(path))

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (n_samples,)

        if sr != self.TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, self.TARGET_SR)

        n = waveform.shape[0]
        if n < self.TARGET_SAMPLES:
            waveform = F.pad(waveform, (0, self.TARGET_SAMPLES - n))
        else:
            waveform = waveform[: self.TARGET_SAMPLES]

        return waveform, caption


def unfreeze_audio_encoder(
    model: laion_clap.CLAP_Module,
    top_n_stages: int = 1,
) -> None:
    """Unfreeze the last top_n_stages Swin stages + norm + audio projection heads."""
    for p in model.model.text_branch.parameters():
        p.requires_grad_(False)

    for p in model.model.audio_branch.parameters():
        p.requires_grad_(False)

    stages = model.model.audio_branch.layers
    for stage in stages[-top_n_stages:]:
        for p in stage.parameters():
            p.requires_grad_(True)

    for p in model.model.audio_branch.norm.parameters():
        p.requires_grad_(True)
    for p in model.model.audio_transform.parameters():
        p.requires_grad_(True)
    for p in model.model.audio_projection.parameters():
        p.requires_grad_(True)

    if hasattr(model.model, "logit_scale_a"):
        model.model.logit_scale_a.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Audio encoder unfrozen (last {top_n_stages} Swin stage(s) + norm + projection) — "
        f"trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
    )


def inject_lora_text_encoder(
    model: laion_clap.CLAP_Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> None:
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["query", "value"]

    for param in model.parameters():
        param.requires_grad_(False)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    model.model.text_branch = get_peft_model(model.model.text_branch, lora_config)

    if hasattr(model.model, "text_projection"):
        for param in model.model.text_projection.parameters():
            param.requires_grad_(True)

    if hasattr(model.model, "logit_scale_a"):
        model.model.logit_scale_a.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA injected — trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
