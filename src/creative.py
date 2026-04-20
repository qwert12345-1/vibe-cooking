"""Creative recipe generation via a local small LLM loaded with
HuggingFace `transformers` (CPU, float32). No llama.cpp / GGUF / DLL
dependencies — slower than llama.cpp (≈10-20 tok/s for a 0.5 B model on
CPU) but far more robust across Windows / Linux / macOS / pip / conda.

Default model: Qwen2.5-0.5B-Instruct (safetensors, ~1 GB).
Upgrade to the 1.5 B or 3 B variant by setting `RR_LLM_REPO`.
"""
from __future__ import annotations

import os
# HF mirror for restricted networks; setdefault respects an outer override.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get("RR_LLM_REPO", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_MAX_TOKENS = int(os.environ.get("RR_LLM_MAX_TOKENS", "600"))


SYSTEM_PROMPT_EN = """\
You are a creative but realistic recipe writer. Given ingredients the user has
on hand, invent ONE dish and write it using the exact format below. Do not add
preamble, explanations, or notes outside the template.

Rules:
- Use MOSTLY the user's ingredients. You may add common pantry items (salt,
  pepper, oil, water, sugar, flour) but no other surprise ingredients.
- Keep the recipe practical — real home cooking.
- Steps must be short and actionable.

Format (use these exact markdown headers):

# <Dish name>
**Cuisine**: <cuisine>  ·  **Serves**: <n>  ·  **Time**: <total minutes>

## Ingredients
- <quantity> <ingredient>
- ...

## Steps
1. <short step>
2. <short step>
3. ...

## Tip
<one short tip for flavor or technique>
"""

SYSTEM_PROMPT_ZH = """\
你是一位有创意但贴近实际的菜谱作者。根据用户手上的食材，创造 一道 菜，
严格按下面的模板输出，不要写多余的前言或解释。

规则：
- 主要使用用户提供的食材，可以补充常见调料（盐、胡椒、油、水、糖、面粉），
  不要出现其他意外食材。
- 菜谱要实际可操作。
- 步骤要简洁，直接给出动作。

格式（必须使用以下的 markdown 标题）：

# <菜名>
**菜系**: <cuisine>  ·  **份数**: <n>  ·  **总用时**: <分钟>

## 所需食材
- <用量> <食材>
- ...

## 步骤
1. <简短步骤>
2. <简短步骤>
3. ...

## 小贴士
<一条简短的调味或技巧提示>
"""


@dataclass
class CreativeResult:
    ok: bool
    text: str
    model: str = ""


# --------------------------------------------------------------------------
# Language / prompt helpers (unchanged from the llama-cpp version)
# --------------------------------------------------------------------------
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _has_chinese(s: str) -> bool:
    return bool(_CJK_RE.search(s))


def _pick_language(ingredients: list[str], hint: str) -> str:
    if hint in ("en", "zh"):
        return hint
    return "zh" if _has_chinese(" ".join(ingredients)) else "en"


def _build_user_prompt(
    ingredients: list[str],
    *,
    cuisine: str | None,
    diet: list[str] | None,
    excluded: list[str] | None,
    meal_type: list[str] | None,
    max_calories: float | None,
    max_time_minutes: float | None,
    lang: str,
) -> str:
    if lang == "zh":
        parts = [f"用户手上的食材: {', '.join(ingredients)} 。"]
        if cuisine:
            parts.append(f"偏好菜系: {cuisine}。")
        if diet:
            parts.append(f"饮食要求: {', '.join(diet)}。")
        if meal_type:
            parts.append(f"餐类: {', '.join(meal_type)}。")
        if max_calories and max_calories > 0:
            parts.append(f"每份热量不要超过 {int(max_calories)} 千卡。")
        if max_time_minutes and max_time_minutes > 0:
            parts.append(f"总烹饪时间不要超过 {int(max_time_minutes)} 分钟。")
        if excluded:
            parts.append(f"绝对不要包含: {', '.join(excluded)}。")
        parts.append("现在按模板创作一道创意菜。")
        return " ".join(parts)

    parts = [f"Ingredients the user has: {', '.join(ingredients)}."]
    if cuisine:
        parts.append(f"Preferred cuisine: {cuisine}.")
    if diet:
        parts.append(f"Dietary requirements: {', '.join(diet)}.")
    if meal_type:
        parts.append(f"Meal type: {', '.join(meal_type)}.")
    if max_calories and max_calories > 0:
        parts.append(f"Keep calories per serving under {int(max_calories)} kcal.")
    if max_time_minutes and max_time_minutes > 0:
        parts.append(f"Total cooking time must be under {int(max_time_minutes)} minutes.")
    if excluded:
        parts.append(f"Must NOT contain: {', '.join(excluded)}.")
    parts.append("Now write ONE creative dish following the template.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# transformers model singleton — lazy, thread-safe
# --------------------------------------------------------------------------
_bundle = None  # (tokenizer, model)
_bundle_lock = threading.Lock()
_load_error: Optional[str] = None


def _cached_snapshot_dir(repo_id: str) -> str | None:
    """Resolve a repo id to its newest local HF snapshot directory, if any."""
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if not hub_cache:
        return None

    repo_dir = Path(hub_cache) / f"models--{repo_id.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = repo_dir / "snapshots" / revision
        if snapshot_dir.is_dir():
            return str(snapshot_dir)

    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _load_llm():
    """Load the tokenizer + CausalLM once. Thread-safe and idempotent.

    Prefers the cached copy (`local_files_only=True`) so a restricted-network
    machine where huggingface.co times out still starts up as long as the
    model is already on disk under ./models/ (populated by the bundler)."""
    global _bundle, _load_error
    if _bundle is not None:
        return _bundle
    if _load_error is not None:
        return None
    with _bundle_lock:
        if _bundle is not None:
            return _bundle
        if _load_error is not None:
            return None
        try:
            import torch  # noqa: F401 — side effect: init torch before transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            _load_error = f"transformers / torch import failed: {e}"
            return None

        def _load(local_only: bool):
            # Qwen2.5 is natively supported by transformers (Qwen2ForCausalLM);
            # we explicitly do NOT pass trust_remote_code=True. Enabling it
            # makes transformers do an extra hub lookup to check for custom
            # code updates, which network-times-out even when
            # local_files_only=True is set (the flag only skips weight
            # downloads, not remote-code metadata checks).
            source = DEFAULT_MODEL
            if local_only:
                source = _cached_snapshot_dir(DEFAULT_MODEL) or DEFAULT_MODEL
            tok = AutoTokenizer.from_pretrained(
                source,
                local_files_only=local_only,
            )
            mdl = AutoModelForCausalLM.from_pretrained(
                source,
                local_files_only=local_only,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )
            mdl.eval()
            return tok, mdl

        try:
            print(f"[creative] Loading {DEFAULT_MODEL} (transformers, CPU)...",
                  flush=True)
            # Diagnostic: show where HF thinks the cache is. If this prints a
            # different directory than where the model actually lives,
            # someone's env var is overriding ours.
            print(f"[creative]   HF_HOME={os.environ.get('HF_HOME')!r}", flush=True)
            print(f"[creative]   HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')!r}", flush=True)
            local_snapshot = _cached_snapshot_dir(DEFAULT_MODEL)
            if local_snapshot:
                print(f"[creative]   local snapshot={local_snapshot!r}", flush=True)
            try:
                _bundle = _load(local_only=True)
                print("[creative] Loaded from local cache (no network).", flush=True)
            except Exception as _local_err:
                print(
                    f"[creative] Cache miss ({type(_local_err).__name__}: "
                    f"{_local_err}) — fetching from HF ...",
                    flush=True,
                )
                _bundle = _load(local_only=False)
                print("[creative] Fetched + loaded.", flush=True)
        except Exception as e:  # pragma: no cover — defensive
            _load_error = f"Failed to load model: {e}"
            print(f"[creative] {_load_error}", flush=True)
            return None
    return _bundle


def check_llm() -> tuple[bool, str]:
    """Non-blocking UI health check."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False, "transformers not installed — run: pip install transformers"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch not installed"
    return True, f"Ready · {DEFAULT_MODEL} (loads on first Create!)"


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def generate_recipe(
    ingredients: list[str],
    *,
    cuisine: str | None = None,
    diet: list[str] | None = None,
    excluded: list[str] | None = None,
    meal_type: list[str] | None = None,
    max_calories: float | None = None,
    max_time_minutes: float | None = None,
    language: str = "auto",
    temperature: float = 0.75,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> CreativeResult:
    if not ingredients:
        return CreativeResult(ok=False, text="No ingredients — type something first.")

    bundle = _load_llm()
    if bundle is None:
        return CreativeResult(ok=False, text=f"⚠️ LLM unavailable: {_load_error}")

    tokenizer, model = bundle

    lang = _pick_language(ingredients, language)
    system_prompt = SYSTEM_PROMPT_ZH if lang == "zh" else SYSTEM_PROMPT_EN
    user_prompt = _build_user_prompt(
        ingredients,
        cuisine=cuisine, diet=diet, excluded=excluded,
        meal_type=meal_type,
        max_calories=max_calories,
        max_time_minutes=max_time_minutes,
        lang=lang,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        import torch
        # Qwen's tokenizer ships a chat template — render the conversation
        # and let the tokenizer append the assistant prefix for us.
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Strip the prompt tokens so we only keep the newly generated portion.
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    except Exception as e:
        return CreativeResult(ok=False, text=f"⚠️ Inference failed: {e}")

    if not text:
        return CreativeResult(ok=False, text="⚠️ Model returned empty output.")
    return CreativeResult(ok=True, text=text, model=DEFAULT_MODEL)
