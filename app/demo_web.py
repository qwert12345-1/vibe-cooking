"""Gradio web frontend — application entry point.

Run:

    python -m app.demo_web
"""
from __future__ import annotations

import os
from pathlib import Path
import socket

# ---- Project-local cache redirection ---------------------------------------
# Everything the app downloads at runtime (SBERT, Qwen2.5 GGUF, SDXS, NLTK
# corpora) is pointed at directories INSIDE the project tree so the whole
# thing zips up self-contained. If these env vars are already set externally
# (e.g. by a CI runner), we respect that — setdefault only fills in blanks.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"
_NLTK_DIR = _PROJECT_ROOT / "nltk_data"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)
_NLTK_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face caches — force everything under ./models/hub/. We explicitly
# OVERWRITE the env vars (not setdefault) because a lingering outer HF_HOME
# from an older sibling project (e.g. a renamed recipe_recommender/) would
# otherwise make HF look for cached files in the wrong absolute path and
# report a "Cache miss" even when the files are physically in this project
# tree. Hardcoding project-local paths keeps the zip self-contained.
os.environ["HF_HOME"] = str(_MODELS_DIR)
os.environ["HF_HUB_CACHE"] = str(_MODELS_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(_MODELS_DIR / "hub")
# NLTK searches NLTK_DATA for corpora before falling back to ~/nltk_data.
os.environ["NLTK_DATA"] = str(_NLTK_DIR)

# Allow both torch's and llama-cpp-python's OMP runtimes to coexist on Windows.
# Without this, whichever lib imports later fails with "WinError 127" loading
# shm.dll (or similar), because they each bundle their own OpenMP DLL.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# (Former llama-cpp DLL workaround removed — we now load the LLM through the
# transformers stack, which doesn't depend on native ggml/llama DLLs.)

# Import torch first (if available) so its OMP runtime wins the load race
# before llama-cpp-python brings its own in. No-op if torch isn't installed.
try:
    import torch  # noqa: F401
except Exception:
    pass

import sys

# Make `src` importable when invoked as `python -m app.demo_web` from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from app.components import (  # noqa: E402
    render_low_confidence_prompt,
    render_normalization_chips,
    render_result_cards,
    render_single_recipe_card,
)
from app.visualizer import build_cluster_figure  # noqa: E402
from src.constants import CUISINES, DIET_TAGS, MEAL_TYPES  # noqa: E402
from src.creative import generate_recipe  # noqa: E402
from src.seq2seq_backend import generate_seq2seq_recipe  # noqa: E402
from src.normalizer import flatten_resolved  # noqa: E402
from src.recommender import Filters, RecipeRecommender  # noqa: E402
from src.visualize_dish import generate_dish_image  # noqa: E402


def _candidate_local_addresses() -> list[str]:
    addrs: list[str] = []
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
    except Exception:
        infos = []
    for info in infos:
        ip = info[4][0]
        if ip.startswith("127.") or ip == "0.0.0.0" or ip in addrs:
            continue
        addrs.append(ip)
    return addrs


def _hf_auth_sources() -> list[str]:
    sources: list[str] = []
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        if os.environ.get(env_name):
            sources.append(f"env:{env_name}")

    token_candidates = [
        Path(os.environ.get("HF_HOME", "")) / "token" if os.environ.get("HF_HOME") else None,
        Path(os.environ.get("HF_HOME", "")) / "stored_tokens" if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".cache" / "huggingface" / "stored_tokens",
        Path.home() / ".config" / "huggingface" / "token",
        Path.home() / ".config" / "huggingface" / "stored_tokens",
        Path.home() / ".huggingface" / "token",
    ]
    for candidate in token_candidates:
        if candidate and candidate.exists():
            sources.append(f"file:{candidate}")
    return sources


def _print_runtime_diagnostics(server_name: str, server_port: int) -> None:
    hostname = socket.gethostname()
    print("[startup] Machine diagnostics:", flush=True)
    print(f"[startup]   hostname={hostname}", flush=True)
    print(f"[startup]   bind={server_name}:{server_port}", flush=True)
    print(f"[startup]   localhost=http://127.0.0.1:{server_port}", flush=True)

    lan_addrs = _candidate_local_addresses()
    if server_name in ("127.0.0.1", "localhost"):
        print("[startup]   network_access=localhost-only", flush=True)
    else:
        print("[startup]   network_access=LAN-visible", flush=True)
        for addr in lan_addrs:
            print(f"[startup]   lan_url=http://{addr}:{server_port}", flush=True)

    auth_sources = _hf_auth_sources()
    if auth_sources:
        print("[startup]   hf_auth_sources=", flush=True)
        for source in auth_sources:
            print(f"[startup]     - {source}", flush=True)
    else:
        print("[startup]   hf_auth_sources=none detected", flush=True)


# --------------------------------------------------------------------
# Engine bootstrap — build or load from cache at startup.
# --------------------------------------------------------------------
print("Bootstrapping RecipeRecommender...")
_RECIPE_LIMIT = int(os.environ.get("RR_RECIPE_LIMIT", "0")) or None
_ENGINE = RecipeRecommender.build_or_load(
    recipe_limit=_RECIPE_LIMIT,
    use_semantic_normalizer=os.environ.get("RR_NO_SBERT", "0") != "1",
)
print(f"Engine ready: {len(_ENGINE.recipes)} recipes, {len(_ENGINE.vocabulary)} vocab tokens")

# Pre-download + pre-load the creative LLM at startup so the first "✨ Creative
# recipe" click is fast. Set RR_NO_LLM_PRELOAD=1 to skip and keep startup fast
# (LLM will then load lazily on first button click, as before).
if os.environ.get("RR_NO_LLM_PRELOAD", "0") != "1":
    from src.creative import _load_llm  # noqa: E402
    print("Preloading local creative LLM...")
    _load_llm()
    print("Creative LLM preload finished.")

# Same treatment for the SEQ2SEQ seq2seq title model. Small (~30 MB checkpoint),
# so this is fast even on the first run.
if (os.environ.get("RR_NO_SEQ2SEQ_PRELOAD", "0") != "1"
        and os.environ.get("RR_NO_SEQ2SEQ", "0") != "1"):
    from src.seq2seq_backend import _load_bundle as _load_seq2seq  # noqa: E402
    print("Preloading SEQ2SEQ seq2seq model...")
    _load_seq2seq()
    print("SEQ2SEQ preload finished.")

# Same treatment for the text-to-image pipeline. First run still incurs the
# ~1.5 GB (SDXS) or ~4 GB (SD 1.5) download; after that startup is quick.
# Set RR_NO_IMAGE_PRELOAD=1 to skip, or RR_NO_IMAGE=1 to disable T2I entirely.
if (os.environ.get("RR_NO_IMAGE_PRELOAD", "0") != "1"
        and os.environ.get("RR_NO_IMAGE", "0") != "1"):
    from src.visualize_dish import _load_pipeline as _load_t2i  # noqa: E402
    print("Preloading local text-to-image pipeline...")
    _load_t2i()
    print("Text-to-image preload finished.")


# --------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------
def _split_ingredients(text: str) -> list[str]:
    if not text:
        return []
    # Accept comma, newline, and Chinese comma as separators.
    parts = []
    for raw in text.replace("，", ",").replace("\n", ",").split(","):
        p = raw.strip()
        if p:
            parts.append(p)
    return parts


# High-contrast status pill (works in both Gradio light and dark themes).
# Bold text on a saturated background — doesn't rely on the theme's default
# foreground color being readable.
_SPINNER_HTML = """\
<div style="display:inline-flex;align-items:center;gap:12px;padding:12px 18px;
    background:#1d3557;border:1px solid #0f203a;border-radius:10px;
    font-size:19px;color:#ffffff;box-shadow:0 2px 6px rgba(0,0,0,0.12);">
  <span style="width:20px;height:20px;border-radius:50%;
       border:3px solid rgba(255,255,255,0.35);border-top-color:#ffffff;
       animation:rr-spin 0.9s linear infinite;display:inline-block;"></span>
  <span style="font-weight:700;color:#ffffff;letter-spacing:0.2px;">{label}</span>
</div>
<style>@keyframes rr-spin{{to{{transform:rotate(360deg);}}}}</style>
"""


# Image-sized placeholder with a big centered spinner so users see where the
# picture will show up and that work is in progress.
_IMAGE_PLACEHOLDER_HTML = """\
<div style="width:100%;max-width:512px;aspect-ratio:1/1;border-radius:12px;
    background:linear-gradient(135deg,#1d3557 0%,#2b4876 100%);
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    gap:18px;color:#ffffff;box-shadow:0 4px 16px rgba(15,23,42,0.25);">
  <span style="width:56px;height:56px;border-radius:50%;
       border:6px solid rgba(255,255,255,0.25);border-top-color:#ffffff;
       animation:rr-spin 1s linear infinite;display:inline-block;"></span>
  <div style="font-size:15px;font-weight:700;color:#ffffff;
       text-align:center;padding:0 24px;line-height:1.4;
       text-shadow:0 1px 2px rgba(0,0,0,0.25);">
    {label}
  </div>
</div>
<style>@keyframes rr-spin{{to{{transform:rotate(360deg);}}}}</style>
"""


def _progress_markdown(label: str, detail: str = "") -> str:
    """Spinner + short label + optional detail paragraph for Gradio Markdown."""
    html = _SPINNER_HTML.format(label=label)
    if detail:
        return html + f"\n\n{detail}"
    return html


def _image_placeholder(label: str) -> str:
    """Large image-shaped placeholder that sits in the slot where the final
    generated picture will appear — tells the user exactly where to look."""
    return _IMAGE_PLACEHOLDER_HTML.format(label=label)


def _cluster_dropdown_choices() -> list[tuple[str, int]]:
    """Label→cluster_id pairs for the cluster browser dropdown.

    Labels combine the semantic topic (if known) with the top distinctive
    ingredients: 'cluster 3 — stir-fries · chicken, soy sauce, ginger'.
    """
    cids = sorted(set(int(c) for c in _ENGINE.cluster_labels.tolist()))
    return [
        (f"cluster {cid}  —  {_ENGINE.cluster_label_guess(cid)}", cid)
        for cid in cids
    ]


def _representative_recipe_ids(cluster_id: int, n: int = 12) -> list[int]:
    """Return the `n` recipes closest to the centroid of the given cluster
    (in the 2-D projection space)."""
    import numpy as np
    mask = _ENGINE.cluster_labels == int(cluster_id)
    if not mask.any():
        return []
    pts = _ENGINE.projection_2d[mask]
    center = pts.mean(axis=0)
    idx = np.where(mask)[0]
    dists = np.linalg.norm(pts - center, axis=1)
    order = idx[np.argsort(dists)]
    return [int(i) for i in order[:n]]


def on_cluster_browse(cluster_id):
    """Dropdown handler — render a grid of representative recipes for the
    chosen cluster. This replaces the plot-click interaction, which newer
    Gradio versions don't support."""
    if cluster_id is None or cluster_id == "":
        return "<em>Pick a cluster above to see its recipes.</em>"
    try:
        cid = int(cluster_id)
    except (TypeError, ValueError):
        return "<em>Bad cluster id.</em>"
    ids = _representative_recipe_ids(cid, n=12)
    if not ids:
        return "<em>That cluster has no recipes.</em>"
    topic = _ENGINE.cluster_topics.get(cid)
    cards = [
        render_single_recipe_card(_ENGINE.recipes[i], cluster_id=cid, topic=topic)
        for i in ids
    ]
    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));'
        'gap:16px;align-items:stretch;">'
        + "".join(cards)
        + "</div>"
    )


def _build_fig_and_mapping(*, highlight_indices=None, user_position=None):
    """Build the cluster figure AND a {(trace, point): recipe_id} lookup,
    so click events on the Plot component can resolve back to a recipe."""
    fig = build_cluster_figure(
        _ENGINE,
        highlight_indices=highlight_indices,
        user_position=user_position,
    )
    mapping: dict[tuple[int, int], int] = {}
    for t_idx, trace in enumerate(fig.data):
        cd = getattr(trace, "customdata", None)
        if cd is None:
            continue
        for p_idx, val in enumerate(cd):
            try:
                mapping[(t_idx, p_idx)] = int(val)
            except (TypeError, ValueError):
                continue
    return fig, mapping


def _dish_image_markdown(image_result) -> str:
    return (
        "### Dish preview\n\n"
        f'<img src="{image_result.image_data_uri}" '
        'style="width:100%;max-width:512px;border-radius:10px;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.15);" alt="dish preview" />\n\n'
        f'<div class="rr-llm-header"><sub>Generated by `{image_result.model}`.</sub></div>'
    )


def _image_md_for(recipe_text: str, resolved: list[str]) -> str:
    """Run the T2I generator on an already-produced recipe and return the
    markdown to drop into that generator's image slot. Keeps the two
    generator branches (LLM / SEQ2SEQ) symmetric so `on_create` stays readable."""
    image_result = generate_dish_image(recipe_text, resolved)
    if not image_result.ok:
        return (
            "### Dish preview\n\n"
            f"_⚠️ Image generation skipped: {image_result.error}_"
        )
    return _dish_image_markdown(image_result)


def on_create(
    ingredient_text: str,
    cuisine: list[str],
    diet: list[str],
    meal_type: list[str],
    excluded_text: str,
    max_calories: float,
    max_time: float,
    top_k: int,
    want_image: bool,
    generators: list[str],
):
    """Unified "Create" handler.

    Yields tuples that match the UI's `outputs` list:
        ( chips_html, confirm_html,
          llm_section_visibility, llm_header_md, llm_recipe_accordion,
          llm_recipe_md, llm_image_md,
          seq2seq_section_visibility, seq2seq_header_md, seq2seq_recipe_md, seq2seq_image_md,
          retrieval_cards_html, cluster_plot, plot_mapping_state,
          clicked_recipe_html )
    """
    empty_fig_update = gr.update()
    empty_state: dict = {}

    def payload(
        chips: str = "",
        confirm: str = "",
        *,
        llm_visible: bool = True,
        llm_header: str = "",
        llm_recipe_open: bool = False,
        llm_body: str = "",
        llm_image: str = "",
        seq2seq_visible: bool = True,
        seq2seq_recipe_open: bool = False,
        seq2seq_header: str = "",
        seq2seq_body: str = "",
        seq2seq_image: str = "",
        retrieval: str = "",
        fig=empty_fig_update,
        mapping: dict = empty_state,
        clicked: str = gr.update(),
        toast_signal: str = ""
    ):
        return (
            chips, confirm,
            gr.update(visible=llm_visible),
            llm_header,
            gr.update(
                open=llm_recipe_open,
                label="🤖 Wanna see how AI would cook these ingredients?",
            ),
            llm_body,
            llm_image,
            gr.update(visible=seq2seq_visible),
            gr.update(
                open=seq2seq_recipe_open,
                label="🧠 Wanna see the dataset-grounded drafted recipe?",
            ),
            seq2seq_header, seq2seq_body, seq2seq_image,
            retrieval, fig, mapping, clicked,
            f'<div id="rr-toast-signal" data-signal="{toast_signal}"></div>',
        )

    want_llm = "llm" in (generators or [])
    want_seq2seq = "seq2seq" in (generators or [])

    if not want_llm and not want_seq2seq:
        warning = "⚠️ Select at least one generator (LLM or SEQ2SEQ) to run."
        yield payload(
            chips="<em>Pick LLM or SEQ2SEQ (or both) above before hitting Create.</em>",
            llm_recipe_open=True,
            llm_body=warning, seq2seq_body=warning,
        )
        return

    raw = _split_ingredients(ingredient_text)
    if not raw:
        yield payload(
            chips="<em>Enter at least one ingredient above.</em>",
            llm_visible=want_llm,
            seq2seq_visible=want_seq2seq,
            llm_recipe_open=False,
            seq2seq_recipe_open=False,
            llm_body="_Type at least one ingredient above and hit the button._",
            seq2seq_body="_Type at least one ingredient above and hit the button._",
        )
        return

    # ---- Stage 1: normalize + show chips (instant) ----------------------
    tokens = _ENGINE.normalize(raw)
    resolved = flatten_resolved(tokens)
    chips_val = render_normalization_chips(tokens)
    confirm_val = render_low_confidence_prompt(tokens)
    if not resolved:
        yield payload(
            chips=chips_val, confirm=confirm_val,
            llm_visible=want_llm, seq2seq_visible=want_seq2seq,
            llm_recipe_open=want_llm,
            llm_body="⚠️ Couldn't resolve any of your ingredients.",
            seq2seq_body="⚠️ Couldn't resolve any of your ingredients.",
        )
        return

    preview = ", ".join(resolved[:10]) + (" …" if len(resolved) > 10 else "")

    # Initial spinner: show a "waiting on retrieval" state in whichever
    # sections are active.
    init_body = _progress_markdown(
        "Waiting on retrieval before thinking up a creative dish…",
        f"**Canonical ingredients:** _{preview}_",
    )
    yield payload(
        chips=chips_val, confirm=confirm_val,
        llm_visible=want_llm, seq2seq_visible=want_seq2seq,
        llm_recipe_open=False,
        llm_body=init_body if want_llm else "",
        seq2seq_body=init_body if want_seq2seq else "",
        retrieval=_progress_markdown("Finding similar real recipes in the dataset…"),
        toast_signal="loading",
    )

    # ---- Stage 2: retrieval + cluster plot update (fast) ----------------
    filters = Filters(
        cuisine=cuisine or [],
        diet=diet or [],
        meal_type=meal_type or [],
        excluded_ingredients=_split_ingredients(excluded_text),
        max_calories=max_calories if max_calories and max_calories > 0 else None,
        max_time_minutes=max_time if max_time and max_time > 0 else None,
    )
    ranked = _ENGINE.recommend(tokens, top_k=int(top_k), filters=filters)
    retrieval_html = render_result_cards(ranked)
    highlight = [r.recipe.id for r in ranked]
    user_pos = _ENGINE.project_user_position(tokens)
    fig, plot_mapping = _build_fig_and_mapping(
        highlight_indices=highlight, user_position=user_pos
    )

    # Auto-populate the "Selected recipe" panel with the top-ranked real
    # recipe so the user always sees something there right after Create — no
    # need to first click a point on the scatter.
    if ranked:
        top = ranked[0].recipe
        top_cid = int(_ENGINE.cluster_labels[top.id])
        top_topic = _ENGINE.cluster_topics.get(top_cid)
        top_card_html = render_single_recipe_card(top, cluster_id=top_cid, topic=top_topic)
    else:
        top_card_html = "<em>No recipes matched your filters.</em>"

    llm_body = seq2seq_body = ""
    llm_header = seq2seq_header = ""
    llm_image = seq2seq_image = ""
    llm_ok = False

    # ---- Stage 3: SEQ2SEQ branch ---------------------------------------------
    if want_seq2seq:
        yield payload(
            chips=chips_val, confirm=confirm_val,
            llm_visible=want_llm, seq2seq_visible=want_seq2seq,
            llm_recipe_open=False, seq2seq_recipe_open=False,
            seq2seq_body=_progress_markdown(
                "Running SEQ2SEQ seq2seq + RAG draft — ~2 to 5 s…",
                f"**Canonical ingredients:** _{preview}_",
            ),
            llm_body=(_progress_markdown("Queued — starts after SEQ2SEQ…", f"**Canonical ingredients:** _{preview}_")
                     if want_llm else ""),
            retrieval=retrieval_html, fig=fig, mapping=plot_mapping, clicked=top_card_html,
        )

        seq2seq_result = generate_seq2seq_recipe(resolved, engine=_ENGINE)
        if not seq2seq_result.ok:
            seq2seq_body = seq2seq_result.text
            seq2seq_header = ""
        else:
            seq2seq_header = (
                "_Generated by the hand-trained SEQ2SEQ seq2seq + retrieval-"
                "augmented draft composer — title predicted by the model, "
                "ingredients/steps grounded in the top-5 retrieval matches._"
            )
            seq2seq_body = seq2seq_result.text

        if want_image and seq2seq_result.ok:
            yield payload(
                chips=chips_val, confirm=confirm_val,
                llm_visible=want_llm, seq2seq_visible=want_seq2seq,
                llm_recipe_open=False, seq2seq_recipe_open=False,
                seq2seq_header=seq2seq_header, seq2seq_body=seq2seq_body,
                seq2seq_image="### Dish preview\n\n" + _image_placeholder(
                    "Rendering SEQ2SEQ dish image — ~5–15 s on CPU with SDXS…"
                ),
                llm_body=(_progress_markdown("Queued — starts after SEQ2SEQ image…", f"**Canonical ingredients:** _{preview}_")
                         if want_llm else ""),
                retrieval=retrieval_html, fig=fig, mapping=plot_mapping, clicked=top_card_html,
            )
            seq2seq_image = _image_md_for(seq2seq_result.text, resolved)

        yield payload(
            chips=chips_val, confirm=confirm_val,
            llm_visible=want_llm, seq2seq_visible=want_seq2seq,
            llm_recipe_open=False, seq2seq_recipe_open=False,
            seq2seq_header=seq2seq_header, seq2seq_body=seq2seq_body,
            seq2seq_image=seq2seq_image,
            llm_body=(_progress_markdown("Starting LLM…", f"**Canonical ingredients:** _{preview}_")
                     if want_llm else ""),
            retrieval=retrieval_html, fig=fig, mapping=plot_mapping, clicked=top_card_html,
        )

    # ---- Stage 4: LLM branch ---------------------------------------------
    if want_llm:
        yield payload(
            chips=chips_val, confirm=confirm_val,
            llm_visible=want_llm, seq2seq_visible=want_seq2seq,
            llm_recipe_open=False, seq2seq_recipe_open=False,
            seq2seq_header=seq2seq_header, seq2seq_body=seq2seq_body, seq2seq_image=seq2seq_image,
            llm_body=_progress_markdown(
                "Generating with Qwen2.5-0.5B on CPU (transformers) — ~30 to 90 s…",
                f"**Canonical ingredients:** _{preview}_",
            ),
            retrieval=retrieval_html, fig=fig, mapping=plot_mapping, clicked=top_card_html,
        )

        chosen_cuisine = cuisine[0] if cuisine else None
        
        import re
        lang_override = "zh" if re.search(r"[\u4e00-\u9fff]", ingredient_text) else "en"
        
        llm_result = generate_recipe(
            ingredients=resolved,
            cuisine=chosen_cuisine,
            diet=diet or None,
            excluded=_split_ingredients(excluded_text) or None,
            meal_type=meal_type or None,
            max_calories=(max_calories if max_calories and max_calories > 0 else None),
            max_time_minutes=(max_time if max_time and max_time > 0 else None),
            language=lang_override,
        )
        if not llm_result.ok:
            llm_body = llm_result.text
            llm_header = ""
        else:
            llm_ok = True
            llm_header = (
                f"_Generated locally by `{llm_result.model}` via transformers "
                "— creative, not from the dataset._"
            )
            llm_body = llm_result.text

        if want_image and llm_result.ok:
            yield payload(
                chips=chips_val, confirm=confirm_val,
                llm_visible=want_llm, seq2seq_visible=want_seq2seq,
                llm_header=llm_header,
                llm_recipe_open=not llm_ok, seq2seq_recipe_open=False,
                llm_body=llm_body,
                llm_image="### Dish preview\n\n" + _image_placeholder(
                    "Rendering LLM dish image — ~5–15 s on CPU with SDXS…"
                ),
                seq2seq_header=seq2seq_header, seq2seq_body=seq2seq_body, seq2seq_image=seq2seq_image,
                retrieval=retrieval_html, fig=fig, mapping=plot_mapping, clicked=top_card_html,
            )
            llm_image = _image_md_for(llm_result.text, resolved)

        yield payload(
            chips=chips_val, confirm=confirm_val,
            llm_visible=want_llm, seq2seq_visible=want_seq2seq,
            llm_header=llm_header,
            llm_recipe_open=not llm_ok, seq2seq_recipe_open=False,
            llm_body=llm_body,
            llm_image=llm_image,
            seq2seq_header=seq2seq_header, seq2seq_body=seq2seq_body, seq2seq_image=seq2seq_image,
            retrieval=retrieval_html, fig=fig, mapping=plot_mapping, clicked=top_card_html,
        )

    # ---- Final ----------------------------------------------------------
    yield payload(
        chips=chips_val, confirm=confirm_val,
        llm_visible=want_llm, seq2seq_visible=want_seq2seq,
        llm_header=llm_header,
        llm_recipe_open=not llm_ok, seq2seq_recipe_open=not seq2seq_ok if 'seq2seq_ok' in locals() else False,
        llm_body=llm_body,
        llm_image=llm_image,
        seq2seq_header=seq2seq_header, seq2seq_body=seq2seq_body, seq2seq_image=seq2seq_image,
        retrieval=retrieval_html, fig=fig, mapping=plot_mapping,
        clicked=top_card_html,
        toast_signal="done",
    )


def on_plot_click_relay(payload_json: str):
    """Called when the JS bridge writes a Plotly click payload into the
    hidden textbox. Parses {trace, point, customdata} and renders the
    recipe card if customdata is a valid recipe id."""
    import json as _json
    if not payload_json or not payload_json.strip():
        return "<em>Click a point on the map above.</em>"
    try:
        data = _json.loads(payload_json)
    except Exception as e:
        return f"<em>Bad click payload: {e}</em>"
    cd = data.get("customdata")
    # Plotly can wrap customdata in an array. Unwrap.
    if isinstance(cd, list) and cd:
        cd = cd[0]
    if cd is None:
        return (
            f"<em>That point isn't a recipe — probably the "
            f"matching-region outline or the user marker "
            f"(trace={data.get('trace')}, point={data.get('point')}).</em>"
        )
    try:
        recipe_id = int(cd)
    except (TypeError, ValueError):
        return f"<em>Couldn't parse customdata: {cd!r}</em>"
    if not (0 <= recipe_id < len(_ENGINE.recipes)):
        return f"<em>Recipe id out of range: {recipe_id}</em>"
    recipe = _ENGINE.recipes[recipe_id]
    cid = int(_ENGINE.cluster_labels[recipe_id])
    topic = _ENGINE.cluster_topics.get(cid)
    return render_single_recipe_card(recipe, cluster_id=cid, topic=topic)


def on_cluster_click(mapping: dict, evt: gr.SelectData):
    """When a point on the cluster scatter is clicked, render the recipe
    at that position as a standalone card. `mapping` comes from the State
    that was populated when the plot was last built: it maps each
    (trace_index, point_index) pair to the corresponding recipe id."""
    # Diagnostic logging — helpful when debugging why a click doesn't work.
    # Remove or lower when everything is stable.
    print(
        f"[click] index={getattr(evt, 'index', None)} "
        f"value={getattr(evt, 'value', None)} "
        f"mapping_size={len(mapping) if mapping else 0}",
        flush=True,
    )
    if not mapping:
        return (
            "<em>Click happens after you've created a recipe — "
            "the map is empty right now.</em>"
        )
    idx = getattr(evt, "index", None)
    # Gradio's SelectData for Plotly plots has varied shapes across versions:
    #   - [trace_idx, point_idx]                 (most common)
    #   - just point_idx (single trace plots)
    #   - {"point_index": ..., "curve_number": ...} (older Gradio builds)
    t_idx = p_idx = None
    if isinstance(idx, (list, tuple)) and len(idx) >= 2:
        try:
            t_idx, p_idx = int(idx[0]), int(idx[1])
        except (TypeError, ValueError):
            pass
    elif isinstance(idx, dict):
        try:
            t_idx = int(idx.get("curve_number", 0))
            p_idx = int(idx["point_index"])
        except (KeyError, TypeError, ValueError):
            pass
    if t_idx is None or p_idx is None:
        return (
            "<em>Couldn't read click coordinates — "
            f"got index={idx!r}. Check the terminal for a [click] debug line.</em>"
        )
    # Mapping keys may come back as strings after JSON round-trip via gr.State.
    recipe_id = mapping.get((t_idx, p_idx))
    if recipe_id is None:
        recipe_id = mapping.get(f"{t_idx},{p_idx}")
    if recipe_id is None:
        # Also tolerate str->str key form (some Gradio State backends).
        recipe_id = mapping.get(f"({t_idx}, {p_idx})")
    if recipe_id is None:
        return (
            f"<em>That point isn't a recipe (trace={t_idx}, pt={p_idx}) — "
            "probably a user/highlight marker.</em>"
        )
    recipe = _ENGINE.recipes[int(recipe_id)]
    cid = int(_ENGINE.cluster_labels[int(recipe_id)])
    topic = _ENGINE.cluster_topics.get(cid)
    return render_single_recipe_card(recipe, cluster_id=cid, topic=topic)


# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
# JS bridge for Plotly clicks. Injected into the page <head> so it runs once
# on initial load, BEFORE Gradio has mounted any components. A MutationObserver
# then waits for the .js-plotly-plot div to appear and attaches a plotly_click
# handler that relays the clicked point into our hidden relay Textbox.
_PLOT_CLICK_SCRIPT = """
<script>
(function() {
    console.log('[rr-click-bridge] script loaded');
    const RELAY_ID = "rr-plot-click-relay";
    const attach = () => {
        // Gradio replaces the .js-plotly-plot element whenever the plot
        // value changes (e.g. after Create!). Scan all divs every time and
        // attach to any that doesn't yet have our flag.
        const plotDivs = document.querySelectorAll('.js-plotly-plot');
        const relay =
            document.querySelector('#' + RELAY_ID + ' textarea')
            || document.querySelector('#' + RELAY_ID + ' input')
            || document.querySelector('textarea#' + RELAY_ID)
            || document.querySelector('input#' + RELAY_ID);
        if (!relay || plotDivs.length === 0) return;
        plotDivs.forEach((plotDiv) => {
            if (plotDiv.__rr_click_attached) return;
            plotDiv.__rr_click_attached = true;
            console.log('[rr-click-bridge] ATTACHED to', plotDiv);
            plotDiv.on('plotly_click', function(evt) {
                if (!evt || !evt.points || !evt.points[0]) return;
                const p = evt.points[0];
                const cd = (p.customdata === undefined) ? null : p.customdata;
                const payload = JSON.stringify({
                    trace: p.curveNumber,
                    point: p.pointIndex,
                    customdata: cd
                });
                const proto = (relay.tagName === 'TEXTAREA')
                    ? window.HTMLTextAreaElement.prototype
                    : window.HTMLInputElement.prototype;
                const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
                setter.call(relay, payload);
                relay.dispatchEvent(new Event('input', {bubbles: true}));
                console.log('[rr-click-bridge] posted payload', payload);
            });
        });
    };
    const setup = () => {
        attach();
        // Keep observer alive forever — Gradio may swap the .js-plotly-plot
        // element on every plot update, and each new element needs its own
        // click listener. attach() bails early on divs it already tagged,
        // so re-running it on every mutation is cheap.
        const obs = new MutationObserver(() => attach());
        obs.observe(document.documentElement, {childList: true, subtree: true});
    };
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setup);
    } else {
        setup();
    }
})();

// ---- Toast notification bridge ----------------------------------------
(function() {
    var lastSignal = '';
    var loadingToast = null;
    var doneToastTimeout = null;

    function mkToast(text, cls) {
        var t = document.createElement('div');
        t.className = 'rr-toast ' + cls;
        t.textContent = text;
        document.body.appendChild(t);
        requestAnimationFrame(function() {
            requestAnimationFrame(function() { t.classList.add('rr-toast-show'); });
        });
        return t;
    }

    function rmToast(t) {
        if (!t || !t.parentNode) return;
        t.classList.remove('rr-toast-show');
        setTimeout(function() { if (t.parentNode) t.parentNode.removeChild(t); }, 400);
    }

    function checkSignal() {
        var el = document.getElementById('rr-toast-signal');
        if (!el) return;
        var sig = el.dataset.signal || '';
        if (sig === lastSignal) return;
        lastSignal = sig;
        if (sig === 'loading') {
            if (loadingToast) { rmToast(loadingToast); }
            if (doneToastTimeout) { clearTimeout(doneToastTimeout); doneToastTimeout = null; }
            loadingToast = mkToast(
                "feel free to browse around this universe and we'll drag you back when the meal is done (hypothetically, at least))",
                'rr-toast-loading'
            );
        } else if (sig === 'done') {
            if (loadingToast) { rmToast(loadingToast); loadingToast = null; }
            var done = mkToast('food is ready... hypothetically! please scroll back to see the results!', 'rr-toast-done');
            doneToastTimeout = setTimeout(function() { rmToast(done); }, 4000);
        }
    }

    var toastObs = new MutationObserver(checkSignal);
    function setupToast() {
        toastObs.observe(document.documentElement, {childList: true, subtree: true});
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupToast);
    } else {
        setupToast();
    }
})();
</script>
"""


# CSS to hide the JS-click-relay Textbox off-screen without removing it from
# the DOM (display:none would also kick it out of Gradio's interactive state).
_APP_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    font-size: 19px !important;
}

.rr-hidden, .rr-hidden * {
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    opacity: 0 !important;
    pointer-events: none !important;
    overflow: hidden !important;
}
/* Kill Gradio Markdown's internal max-height/overflow so the generated
   dish image on the right isn't boxed inside a scrollbar. */
.rr-no-scroll,
.rr-no-scroll * {
    max-height: none !important;
    overflow: visible !important;
}
/* Give each creative-recipe section its own padded card so the content
   doesn't slam against the page edge. Using a 3-layer selector because
   Gradio 5 wraps `gr.Group` in 2–3 nested containers, and without the
   broader selector only a narrow inner strip picks up the styling. */
.rr-recipe-section,
.rr-recipe-section > div,
.rr-recipe-section > .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.rr-recipe-section {
    padding: 24px 30px !important;
    margin: 18px 0 !important;
    border-radius: 20px !important;
    background: rgba(30, 41, 59, 0.4) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease !important;
}
.rr-recipe-section:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 45px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}
.rr-collapse-toggle {
    margin-top: 10px !important;
}
.rr-collapse-toggle > .label-wrap button,
.rr-collapse-toggle > .block > .label-wrap button {
    font-size: 20px !important;
}
.rr-picky-accordion > .label-wrap button,
.rr-picky-accordion > .block > .label-wrap button {
    font-size: 19px !important;
}
.rr-create-btn button {
    font-size: 22px !important;
}
/* ---- Floating toast notifications ---- */
.rr-toast {
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 99999;
    max-width: 380px;
    padding: 14px 20px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 500;
    line-height: 1.55;
    opacity: 0;
    transform: translateY(16px);
    transition: opacity 0.35s ease, transform 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    pointer-events: none;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.rr-toast.rr-toast-show {
    opacity: 1;
    transform: translateY(0);
}
.rr-toast-loading {
    background: rgba(20, 30, 50, 0.96);
    border: 1px solid rgba(99, 102, 241, 0.55);
    color: #e2e8f0;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}
.rr-toast-done {
    background: rgba(10, 28, 20, 0.96);
    border: 1px solid rgba(52, 211, 153, 0.55);
    color: #a7f3d0;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}
.rr-ingredients-box label,
.rr-ingredients-box .gr-block-label,
.rr-ingredients-box .gradio-textbox label,
.rr-ingredients-box .gradio-textbox .gr-block-label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 24px !important;
    -webkit-font-smoothing: antialiased;
}
.rr-ingredients-box textarea {
    font-size: 22px !important;
}
.rr-llm-header,
.rr-llm-header p,
.rr-llm-header em,
.rr-llm-header span {
    color: #dbeafe !important;
}
.rr-llm-header code {
    color: #fef3c7 !important;
    background: rgba(15, 23, 42, 0.75) !important;
    border: 1px solid rgba(251, 191, 36, 0.28);
}
"""


# Gradio 5.x accepts theme in Blocks(); 6.0 moved it to launch(). Pass both
# defensively so the theme is picked up regardless of version.
_THEME = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="fuchsia",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Fira Code"), "ui-monospace", "Consolas", "monospace"],
).set(
    body_background_fill="linear-gradient(135deg, #09090b 0%, #1e1b4b 100%)",
    body_text_color="#f8fafc",
    background_fill_primary="rgba(15, 23, 42, 0.6)",
    background_fill_secondary="rgba(30, 41, 59, 0.4)",
    border_color_accent_subdued="#6366f1",
    border_color_accent="#8b5cf6",
    color_accent_soft="#4f46e5",
    block_background_fill="rgba(30, 41, 59, 0.4)",
    block_border_width="1px",
    block_border_color="rgba(251, 191, 36, 0.10)",
    block_radius="16px",
    block_shadow="0 8px 32px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05)",
    input_background_fill="rgba(15, 23, 42, 0.8)",
    input_border_color="rgba(148, 163, 184, 0.2)",
    input_border_color_focus="#f59e0b",
    input_radius="12px",
    button_primary_background_fill="linear-gradient(90deg, #4f46e5 0%, #d946ef 100%)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4338ca 0%, #c026d3 100%)",
    button_primary_border_color="#4f46e5",
    button_primary_text_color="white",
    button_secondary_background_fill="rgba(30, 41, 59, 0.6)",
    button_secondary_border_color="rgba(148, 163, 184, 0.2)",
    button_secondary_text_color="#f8fafc",
    button_secondary_background_fill_hover="rgba(51, 65, 85, 0.8)",
)
_blocks_kwargs = {
    "title": "🧑🍳 Let's Vibe Cook!",
    "head": _PLOT_CLICK_SCRIPT,
    "css": _APP_CSS + "\n#rr-intro-md, #rr-intro-md > * { border-radius: 0 !important; }\n#rr-intro-md { text-align: center !important; }\n#rr-intro-md h1 { font-size: 3rem !important; }",
}

try:
    import warnings
    # Suppress the noisy Gradio 6.0 future warnings so they don't clog the terminal
    warnings.filterwarnings("ignore", message=".*parameter in the Blocks constructor will be removed.*")
    _blocks_kwargs["theme"] = _THEME
except Exception:
    pass

with gr.Blocks(**_blocks_kwargs) as demo:
    gr.Markdown(
        "# 🧑‍🍳 Let's Vibe Cook!\n"
        "Throw in whatever you have — we’ll make something up and show you the vibes.\n"
        "If it starts getting a little… concerning, we’ve got real recipes below.\n"
        "Or just press buttons and see what happens — this is basically a food personality test.",
        elem_id="rr-intro-md",
    )

    # Three short (≤3-ingredient) example sets designed around the creative-
    # cooking pitch. The first is a safe-ish template for a pleasant first
    # impression; the third deliberately pushes into truly "dark cuisine"
    # territory to showcase when the retrieval fallback earns its keep.
    _EXAMPLE_COZY = "chicken, garlic, lemon"
    _EXAMPLE_BOLD = "chocolate, bacon, chili"
    _EXAMPLE_DARK = "watermelon, mayonnaise, wasabi"

    # ---------------- Input card (full-width, centered) ---------------
    with gr.Group():
        ingredient_tb = gr.Textbox(
            label="Ingredients (comma- or newline-separated)",
            placeholder="chicken, onion, tomatoes, garlic, miso",
            lines=5,
            elem_classes=["rr-ingredients-box"],
        )
        with gr.Row(equal_height=True):
            ex_cozy_btn = gr.Button("🌿 Cozy classic", size="lg")
            ex_bold_btn = gr.Button("🔥 Bold mashup", size="lg")
            ex_dark_btn = gr.Button("💀 Dark cuisine", size="lg")
        gr.HTML("<div style='margin-bottom:4px;'></div>")
        ex_cozy_btn.click(lambda: _EXAMPLE_COZY, outputs=ingredient_tb)
        ex_bold_btn.click(lambda: _EXAMPLE_BOLD, outputs=ingredient_tb)
        ex_dark_btn.click(lambda: _EXAMPLE_DARK, outputs=ingredient_tb)

        with gr.Accordion("Picky? (or not, whatever)", open=False, elem_classes=["rr-picky-accordion"]):
            gr.HTML(
                "<div style='color:#94a3b8;font-size:17px;font-style:italic;margin:0 0 14px 0;'>"
                "These filters shape the real recipe suggestions below — the AI chefs ignore them and freelance anyway.</div>"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    cuisine_cb = gr.CheckboxGroup(choices=CUISINES, label="Cuisine")
                    meal_cb = gr.CheckboxGroup(choices=MEAL_TYPES, label="Meal type")
                with gr.Column(scale=1):
                    diet_cb = gr.CheckboxGroup(choices=DIET_TAGS, label="Diet / Health")
                with gr.Column(scale=1):
                    excluded_tb = gr.Textbox(
                        label="Excluded ingredients (comma-separated)",
                        placeholder="beef, peanut",
                    )
                    max_cal_sl = gr.Slider(0, 2000, value=0, step=50,
                                           label="Max calories (0 = no limit)")
                    max_time_sl = gr.Slider(0, 240, value=0, step=10,
                                            label="Max time (min; 0 = no limit)")

        with gr.Row():
            top_k_sl = gr.Slider(
                3, 30, value=10, step=1, scale=3,
                label="Emergency Exit: Real Recipes (how many)",
            )
            want_image_cb = gr.Checkbox(
                value=True, scale=2,
                label="Want visuals? ",
            )

        # Which creative-recipe generators to run. At least one must be
        # selected — the handler falls back to a friendly error otherwise.
        generators_cbg = gr.CheckboxGroup(
            choices=[
                ("🤖 Creative Chef", "llm"),
                ("🧠 Evidence-Based Chef", "seq2seq"),
            ],
            value=["llm", "seq2seq"],
            label="Choose Your Theoretical Chef(s) (proof of edibility not included)",
        )

        gr.HTML(
            "<div style='color:#94a3b8;font-size:17px;margin:4px 0 8px 2px;'>"
            "🤖 Creative Chef freestyles off-script. 🧠 Evidence-Based Chef stays grounded in the dataset. "
            "Running both and comparing is half the fun.</div>"
        )
        create_btn = gr.Button(
            "🍳 Create!", variant="primary", size="lg",
            elem_classes=["rr-create-btn"],
        )

    # Chips / confirmations shown between the input card and the hero section,
    # full width — cleaner than a tall skinny right column.
    chips_html = gr.HTML()
    confirm_html = gr.HTML()

    toast_signal_html = gr.HTML('<div id="rr-toast-signal"></div>', visible=True)


    # -------- Creative dishes (HERO) — one section per generator ---------
    # The LLM section and the SEQ2SEQ section render the same "text-left,
    # image-right" layout, but each has its own header / body / image slots
    # so they can update independently. If the user deselects one generator,
    # its section becomes visible=False (hidden entirely).
    gr.Markdown("## ✨ What You Just Invented")

    with gr.Row():
        # --- SEQ2SEQ section ---
        with gr.Column():
            seq2seq_section = gr.Group(visible=True)
            with seq2seq_section:
                with gr.Accordion(
                    "🧠 Wanna see Evidence-Based Chef's recipe?",
                    open=False,
                    elem_classes=["rr-recipe-section", "rr-collapse-toggle"],
                ) as seq2seq_recipe_accordion:
                    # gr.Markdown("### 🧠 Evidence-Based Chef")
                    seq2seq_header_md = gr.Markdown("")
                    seq2seq_recipe_md = gr.Markdown(
                        "_Hit **Create!** with **Evidence-Based Chef** selected above to predict a "
                        "title with the trained seq2seq and compose a grounded "
                        "recipe draft from it._"
                    )
                    seq2seq_image_md = gr.Markdown("", elem_classes=["rr-no-scroll"])

        # --- LLM section ---
        with gr.Column():
            llm_section = gr.Group(visible=True)
            with llm_section:
                with gr.Accordion(
                    "🤖 What if we just… let it cook? ← 🔥 best?",
                    open=False,
                    elem_classes=["rr-recipe-section", "rr-collapse-toggle"],
                ) as llm_recipe_accordion:
                    # gr.Markdown("### 🤖 LLM — Qwen2.5-0.5B")
                    llm_header_md = gr.Markdown("", elem_classes=["rr-llm-header"])
                    llm_recipe_md = gr.Markdown(
                        "Hit **Create!** with **LLM** selected to obtain a theoretical recipe (proof omitted)."
                    )
                    llm_image_md = gr.Markdown("", elem_classes=["rr-no-scroll"])

    # -------- Fallback: similar real recipes -----------------------------
    gr.Markdown(
        "## 📖 Emergency Exit\n"
        "_Recipes that have been tested by society — just in case._"
    )
    retrieval_cards_html = gr.HTML("")

    # -------- Recipe universe (interactive via JS bridge) ---------------
    gr.Markdown(
        "## 🗺️ Principal Craving Analysis\n"
        "We performed Principal Craving Analysis™ on recipe embeddings, "
        "clustered them in flavor space, and gently flattened the universe into 2D.\n"
        "Click any dot to read that recipe — your search lights up the nearest flavour neighbourhood.\n"
        "Hit **Create!** first to see your ingredients plotted on the map 💞."
    )
    cluster_plot = gr.Plot(label="K-Means clusters")
    # `gr.Plot` (used for Plotly figures) does not expose a .select / .click
    # event in Gradio — only Altair-backed gr.ScatterPlot/LinePlot do. To keep
    # our rich Plotly figure AND still handle clicks, we use a JS bridge:
    #  - a hidden Textbox (below) acts as the relay channel
    #  - a MutationObserver-based snippet (registered in demo.load) attaches
    #    a plotly_click listener to the rendered plot and writes the clicked
    #    point's {trace, point, customdata} payload into the Textbox
    #  - the Textbox's .change() fires on_plot_click_relay to render the card
    # Keep this in the DOM (not visible=False, which strips it) but hide it
    # via CSS so the JS bridge can find and update it. See the .rr-hidden
    # rule in the Blocks `css=` below.
    plot_click_relay_tb = gr.Textbox(
        elem_id="rr-plot-click-relay",
        elem_classes=["rr-hidden"],
        label="",
        show_label=False,
    )
    # `plot_mapping_state` is no longer used (the JS bridge reads customdata
    # directly from the clicked point), but kept as an empty State for the
    # `on_create` yield tuple length — removing it would mean editing every
    # yield. Cheap to keep, effectively a no-op.
    plot_mapping_state = gr.State({})
    gr.Markdown("### Selected recipe")
    clicked_recipe_html = gr.HTML(
        "<em>Hit <strong>Create!</strong> first — the top result loads here automatically. "
        "Then click any dot on the map to explore the full 39k recipe universe.</em>"
    )

    # Initial plot so the component isn't empty on load.
    demo.load(lambda: build_cluster_figure(_ENGINE), inputs=None, outputs=cluster_plot)

    # JS click bridge is injected via `head=` on gr.Blocks above — no
    # per-session setup needed here. The script waits for the Plotly div
    # to mount, then attaches a plotly_click listener that relays the
    # clicked point into the hidden `rr-plot-click-relay` Textbox.

    create_btn.click(
        on_create,
        inputs=[
            ingredient_tb, cuisine_cb, diet_cb, meal_cb, excluded_tb,
            max_cal_sl, max_time_sl, top_k_sl, want_image_cb, generators_cbg,
        ],
        outputs=[
            chips_html, confirm_html,
            llm_section, llm_header_md, llm_recipe_accordion, llm_recipe_md, llm_image_md,
            seq2seq_section, seq2seq_recipe_accordion, seq2seq_header_md, seq2seq_recipe_md, seq2seq_image_md,
            retrieval_cards_html,
            cluster_plot, plot_mapping_state,
            clicked_recipe_html,
            toast_signal_html,
        ],
    )

    # The hidden relay Textbox fires .change() whenever the JS bridge above
    # writes a new payload into it. That handler turns the payload into a
    # rendered recipe card.
    plot_click_relay_tb.change(
        on_plot_click_relay,
        inputs=[plot_click_relay_tb],
        outputs=[clicked_recipe_html],
    )


if __name__ == "__main__":
    _server_name = os.environ.get("RR_SERVER_NAME", "127.0.0.1")
    _server_port = int(os.environ.get("RR_PORT", "7860"))
    _print_runtime_diagnostics(_server_name, _server_port)
    _launch_kwargs: dict = {
        "server_name": _server_name,
        "server_port": _server_port,
        "inbrowser": True,
    }
    
    # Gradio 6.0+: theme, head, css move to launch(). Keep for forward-compat.
    try:
        import inspect
        if "theme" in inspect.signature(demo.launch).parameters:
            _launch_kwargs["theme"] = _THEME
    except Exception:
        pass

    demo.launch(**_launch_kwargs)
