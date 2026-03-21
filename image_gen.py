from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

_pipeline = None


def _get_pipeline(device: str):
    global _pipeline
    if _pipeline is None:
        from diffusers import StableDiffusionPipeline  # type: ignore[import]
        import torch

        dtype = torch.float16 if device == "cuda" else torch.float32
        _pipeline = StableDiffusionPipeline.from_pretrained(
            "segmind/small-sd",
            torch_dtype=dtype,
        )
        _pipeline = _pipeline.to(device)
        _pipeline.set_progress_bar_config(disable=True)
    return _pipeline


def generate_playlist_image(
    prompt: str,
    output_path: Path | None = None,
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    device: str | None = None,
) -> "Image":
    """Generate a playlist cover image from a text prompt."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = _get_pipeline(device)

    visual_prompt = (
        f"Album cover art for a music playlist: {prompt}, "
        "vibrant colors, artistic, detailed, professional album artwork, 4k"
    )
    negative_prompt = "text, watermark, blurry, low quality, distorted"

    result = pipe(
        visual_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )
    image = result.images[0]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    return image
