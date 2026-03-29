from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

import data_pipeline
import image_gen
import playlist_utils
import utils

APP_TITLE = "Playlist Generator Demo"
MAX_K = 50
DEFAULT_K = 20
DEFAULT_VLM_ID = "Qwen/Qwen2-VL-2B-Instruct"

_CACHE: dict[str, Any] = {
    "audio_paths": None,
    "audio_embs": None,
    "clap_model": None,
    "clap_mode": None,
    "vlm": None,
    "vlm_processor": None,
    "vlm_id": None,
    "device": None,
}


def _device() -> str:
    if _CACHE["device"] is None:
        _CACHE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return str(_CACHE["device"])


def _ensure_index(download_fma: bool) -> tuple[list[str], np.ndarray]:
    if _CACHE["audio_paths"] is not None and _CACHE["audio_embs"] is not None:
        return _CACHE["audio_paths"], _CACHE["audio_embs"]

    audio_paths, audio_embs = data_pipeline.ensure_embeddings(
        emb_path=utils.DEFAULT_EMB_PATH,
        songs_dir=utils.DEFAULT_SONGS_DIR,
        device=_device(),
        download_fma=download_fma,
    )
    _CACHE["audio_paths"] = audio_paths
    _CACHE["audio_embs"] = audio_embs
    return audio_paths, audio_embs


def _load_clap(use_lora: bool) -> Any:
    target_mode = "lora" if use_lora else "base"
    if _CACHE["clap_model"] is not None and _CACHE["clap_mode"] == target_mode:
        return _CACHE["clap_model"]

    if use_lora:
        model = utils.load_clap_model_with_lora(
            Path("checkpoints/clap_lora"),
            checkpoint_path=utils.DEFAULT_CHECKPOINT_PATH,
            checkpoint_url=utils.DEFAULT_CHECKPOINT_URL,
            device=_device(),
        )
    else:
        model = utils.load_clap_model(
            checkpoint_path=utils.DEFAULT_CHECKPOINT_PATH,
            checkpoint_url=utils.DEFAULT_CHECKPOINT_URL,
            device=_device(),
        )

    _CACHE["clap_model"] = model
    _CACHE["clap_mode"] = target_mode
    return model


def _load_vlm(vlm_id: str) -> tuple[Any, Any]:
    if _CACHE["vlm"] is not None and _CACHE["vlm_processor"] is not None and _CACHE["vlm_id"] == vlm_id:
        return _CACHE["vlm"], _CACHE["vlm_processor"]

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    dtype = torch.float16 if _device() == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(vlm_id)
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        vlm_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(_device())
    vlm.eval()

    _CACHE["vlm"] = (vlm, process_vision_info)
    _CACHE["vlm_processor"] = processor
    _CACHE["vlm_id"] = vlm_id
    return _CACHE["vlm"], processor


def _image_to_prompt(image: Image.Image, vlm_id: str) -> str:
    vlm_bundle, processor = _load_vlm(vlm_id)
    vlm, process_vision_info = vlm_bundle

    prompt = (
        "Describe this image and its emotions in rich detail. "
        "Focus on mood, atmosphere, feelings, colors, energy level, and narrative. "
        "Describe what this image would sound like as a music playlist."
    )

    rgb = image.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rgb},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    model_inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_device())

    with torch.no_grad():
        output_ids = vlm.generate(**model_inputs, max_new_tokens=250, do_sample=False)

    generated_ids = [out[len(inp) :] for inp, out in zip(model_inputs.input_ids, output_ids)]
    description = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return description


def _retrieve(
    text_prompt: str,
    k: int,
    use_lora: bool,
    prompt_scramble: bool,
    scramble_std: float,
    dual_knn: bool,
    download_fma: bool,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    audio_paths, audio_embs = _ensure_index(download_fma=download_fma)
    model = _load_clap(use_lora=use_lora)

    text_emb = model.get_text_embedding([text_prompt], use_tensor=False)
    text_emb = utils.l2_normalize(np.asarray(text_emb, dtype=np.float32))[0]

    if prompt_scramble:
        text_emb = playlist_utils.scramble_embedding(text_emb, std=scramble_std)

    if dual_knn:
        indices, scores = playlist_utils.dual_anchor_knn(text_emb, audio_embs, k=k)
    else:
        indices, scores = utils.knn_search(text_emb, audio_embs, k=k)

    return audio_paths, indices, scores


def _tracks_table(audio_paths: list[str], indices: np.ndarray, scores: np.ndarray) -> str:
    lines = ["| Rang | Score | Morceau |", "|---:|---:|---|"]
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        lines.append(f"| {rank} | {float(score):.4f} | {Path(audio_paths[int(idx)]).name} |")
    return "\n".join(lines)


def _as_file_list(audio_paths: list[str], indices: np.ndarray) -> list[str]:
    return [audio_paths[int(i)] for i in indices]


def run_demo(
    mode: str,
    text_prompt: str,
    image_input: Image.Image | None,
    k: int,
 ) -> tuple[str, list[str], Image.Image | None]:
    if mode == "Texte":
        prompt = (text_prompt or "").strip()
        if not prompt:
            raise gr.Error("Entre un prompt texte.")
    else:
        if image_input is None:
            raise gr.Error("Ajoute une image pour le mode image.")
        prompt = _image_to_prompt(image_input, DEFAULT_VLM_ID)

    audio_paths, indices, scores = _retrieve(
        text_prompt=prompt,
        k=int(k),
        use_lora=False,
        prompt_scramble=False,
        scramble_std=0.05,
        dual_knn=False,
        download_fma=False,
    )

    table_md = _tracks_table(audio_paths, indices, scores)
    tracks_files = _as_file_list(audio_paths, indices)

    cover_preview = None
    if mode == "Texte":
        cover_preview = image_gen.generate_playlist_image(prompt, device=_device())

    return table_md, tracks_files, cover_preview


def _theme_css() -> str:
    return """
    :root {
        --bg-a: #f4efe6;
        --bg-b: #d9e7f0;
        --ink: #1b1f23;
        --accent: #c4482f;
        --accent-2: #2f6b7a;
        --card: rgba(255, 255, 255, 0.85);
        --border: rgba(27, 31, 35, 0.12);
    }

    .gradio-container {
        font-family: 'Avenir Next', 'Segoe UI', sans-serif;
        background:
            radial-gradient(circle at 15% 20%, #f6d8a8 0%, transparent 35%),
            radial-gradient(circle at 85% 10%, #a8d0e6 0%, transparent 30%),
            linear-gradient(135deg, var(--bg-a), var(--bg-b));
        color: var(--ink);
    }

    .app-shell {
        border: 1px solid var(--border);
        border-radius: 20px;
        background: var(--card);
        backdrop-filter: blur(6px);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.08);
        padding: 8px;
    }

    .app-title {
        letter-spacing: 0.4px;
        margin-bottom: 0;
    }

    .app-sub {
        margin-top: 0;
        color: #3f4b52;
    }

    button.primary {
        background: linear-gradient(110deg, var(--accent), #e16a4f) !important;
        border: none !important;
    }

    button.secondary {
        border-color: var(--accent-2) !important;
        color: var(--accent-2) !important;
    }
    """


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE, css=_theme_css()) as demo:
        gr.Markdown(
            """
            <div class='app-shell'>
            <h1 class='app-title'>Playlist Generator</h1>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                mode = gr.Radio(["Texte", "Image"], value="Texte", label="Entrée")
                text_prompt = gr.Textbox(
                    label="Prompt texte",
                    placeholder="Ex: acid techno nocturne, brut, industriel, 140 bpm",
                    lines=3,
                )
                image_input = gr.Image(label="Image source", type="pil")

                k = gr.Slider(5, MAX_K, value=DEFAULT_K, step=1, label="Nombre de morceaux")

                run_btn = gr.Button("Générer la playlist", variant="primary")

            with gr.Column(scale=7):
                tracks_md = gr.Markdown(label="Résultats")
                tracks_files = gr.Files(label="Morceaux (téléchargeables)")
                cover_preview = gr.Image(label="Cover générée (mode texte)", type="pil")

        run_btn.click(
            fn=run_demo,
            inputs=[
                mode,
                text_prompt,
                image_input,
                k,
            ],
            outputs=[
                tracks_md,
                tracks_files,
                cover_preview,
            ],
            show_progress=True,
        )

        def _toggle_inputs(selected_mode: str):
            is_text = selected_mode == "Texte"
            return (
                gr.update(visible=is_text),
                gr.update(visible=not is_text),
                gr.update(visible=is_text),
            )

        mode.change(
            fn=_toggle_inputs,
            inputs=mode,
            outputs=[text_prompt, image_input, cover_preview],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
