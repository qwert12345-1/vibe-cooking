"""Text-to-image rendering of the dish described by an LLM-generated recipe.

Backend: HuggingFace `diffusers`. Default model is **SDXS-512-DreamShaper**
(~1.5 GB, *1-step* inference, 5-15 s/image on CPU) — the smallest usable
text-to-image model today. Upgrade to Stable Diffusion 1.5 (~4 GB, 15-step,
45-120 s/image, much better quality) by setting `RR_T2I_REPO`.

Install once:

    pip install diffusers accelerate

Knobs:
    RR_T2I_REPO    (default: "IDKiro/sdxs-512-dreamshaper")
    RR_T2I_STEPS   (overrides per-model default)
    RR_T2I_SIZE    (default: 512, pixel width/height)
    RR_NO_IMAGE    (set to "1" to disable image generation entirely)

Upgrade recipe:
    set RR_T2I_REPO=stable-diffusion-v1-5/stable-diffusion-v1-5
    set RR_T2I_STEPS=15
"""
from __future__ import annotations

import os

import base64
import re
import threading
from dataclasses import dataclass
from io import BytesIO
from typing import Optional


DEFAULT_REPO = os.environ.get("RR_T2I_REPO", "IDKiro/sdxs-512-dreamshaper")
# SDXS is a 1-step distilled model — completely different inference config
# than "classic" SD 1.5. Auto-pick sensible defaults from the repo name.
_IS_SDXS = "sdxs" in DEFAULT_REPO.lower()
DEFAULT_STEPS = int(os.environ.get(
    "RR_T2I_STEPS", "1" if _IS_SDXS else "15"
))
DEFAULT_GUIDANCE = 0.0 if _IS_SDXS else 7.0
DEFAULT_SIZE = int(os.environ.get("RR_T2I_SIZE", "512"))
DISABLED = os.environ.get("RR_NO_IMAGE", "0") == "1"


@dataclass
class ImageResult:
    ok: bool
    image_data_uri: str = ""
    error: str = ""
    model: str = ""


# --------------------------------------------------------------------------
# Pipeline singleton (lazy)
# --------------------------------------------------------------------------
_pipe = None
_pipe_lock = threading.Lock()
_pipe_error: Optional[str] = None


def _load_pipeline():
    global _pipe, _pipe_error
    if DISABLED:
        _pipe_error = "Image generation disabled via RR_NO_IMAGE=1."
        return None
    if _pipe is not None:
        return _pipe
    if _pipe_error is not None:
        return None
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        if _pipe_error is not None:
            return None

        try:
            import torch
        except ImportError:
            _pipe_error = "torch not installed"
            return None
        try:
            from diffusers import (
                StableDiffusionPipeline,
                DPMSolverMultistepScheduler,
            )
        except ImportError:
            _pipe_error = ("diffusers not installed. Run: "
                           "pip install diffusers accelerate")
            return None

        try:
            approx_mb = "~1.5 GB" if _IS_SDXS else "~4 GB"
            print(
                f"[image] Loading SD pipeline {DEFAULT_REPO!r} "
                f"(first run downloads {approx_mb})...",
                flush=True,
            )
            # Prefer the cached copy (local_files_only=True) so we skip the
            # "is my cache still up to date?" HEAD request against HF. By bypassing
            # the network, the image generator module will boot instantaneously.
            # Fall back to a normal (online) fetch only if the cache really is empty.
            _common_kwargs = dict(
                torch_dtype=torch.float32,       # fp16 is unreliable on CPU
                safety_checker=None,             # skip NSFW filter for speed
                requires_safety_checker=False,
                # Disable the accelerate "memory efficient" loading path —
                # incompatible between diffusers' call signature and the newer
                # accelerate API in some version combinations.
                low_cpu_mem_usage=False,
            )
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    DEFAULT_REPO, local_files_only=True, **_common_kwargs,
                )
            except Exception:
                pipe = StableDiffusionPipeline.from_pretrained(
                    DEFAULT_REPO, **_common_kwargs,
                )
            # For classic SD, DPM-Solver++ converges faster than DDPM.
            # SDXS uses its own 1-step scheduler — don't touch it.
            if not _IS_SDXS:
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config
                )
            pipe.to("cpu")
            pipe.set_progress_bar_config(disable=False)
            _pipe = pipe
            print("[image] SD pipeline ready.", flush=True)
        except Exception as e:  # network failure, wrong repo, disk full, etc.
            _pipe_error = f"Failed to load pipeline: {e}"
            print(f"[image] {_pipe_error}", flush=True)
            return None
    return _pipe


def check_image_model() -> tuple[bool, str]:
    """Non-blocking health check for the UI."""
    if DISABLED:
        return False, "Image generation disabled (RR_NO_IMAGE=1)"
    try:
        import diffusers  # noqa: F401
    except ImportError:
        return False, ("diffusers not installed — run: "
                       "pip install diffusers accelerate")
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch not installed"
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(
            repo_id=DEFAULT_REPO, filename="model_index.json"
        )
        approx = "~1.5 GB" if _IS_SDXS else "~4 GB"
        status = "cached" if cached else f"will download ({approx}) on first use"
    except Exception:
        status = "on demand"
    return True, f"Ready · {DEFAULT_REPO} · {status}"


# --------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------
def _extract_dish_name(recipe_md: str) -> str:
    """Pick the '# Dish Name' line from the LLM's markdown output.
    Fall back to 'a plated dish' if the header isn't found."""
    m = re.search(r"^#\s+([^\n]+)", recipe_md, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return "a plated dish"


def _build_prompts(dish_name: str, ingredients: list[str]) -> tuple[str, str]:
    """Food-photography-style positive + negative prompts."""
    ings = ", ".join(ingredients[:6]) if ingredients else ""
    pos = (
        f"professional food photography of {dish_name}"
        + (f", featuring {ings}" if ings else "")
        + ", plated beautifully on a ceramic plate, "
          "soft natural lighting, shallow depth of field, "
          "appetizing, high detail, 4k"
    )
    neg = (
        "text, watermark, logo, blurry, low quality, deformed, bad anatomy, "
        "oversaturated, ugly, plastic, fake, people, hands"
    )
    return pos, neg


# --------------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------------
def _pil_to_data_uri(img) -> str:
    """PIL Image → inline data-URI (works inside a Gradio Markdown component)."""
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def generate_dish_image(
    recipe_md: str,
    ingredients: list[str],
    *,
    steps: int = DEFAULT_STEPS,
    size: int = DEFAULT_SIZE,
) -> ImageResult:
    if DISABLED:
        return ImageResult(ok=False, error="Image generation disabled.")
    if not recipe_md or len(recipe_md.strip()) < 5:
        return ImageResult(ok=False, error="Empty recipe text.")
    pipe = _load_pipeline()
    if pipe is None:
        return ImageResult(ok=False, error=f"Pipeline unavailable: {_pipe_error}")

    dish_name = _extract_dish_name(recipe_md)
    pos, neg = _build_prompts(dish_name, ingredients)

    try:
        out = pipe(
            prompt=pos,
            negative_prompt=neg,
            num_inference_steps=steps,
            guidance_scale=DEFAULT_GUIDANCE,
            height=size,
            width=size,
        )
        image = out.images[0]
    except Exception as e:  # OOM, scheduler issues, tokenizer failures, etc.
        return ImageResult(ok=False, error=f"Generation failed: {e}")

    return ImageResult(
        ok=True,
        image_data_uri=_pil_to_data_uri(image),
        model=DEFAULT_REPO,
    )
