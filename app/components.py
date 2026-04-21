"""Pure-rendering helpers for the Gradio UI (chips, confirmation prompts, result cards)."""
from __future__ import annotations

from html import escape

from src.data_loader import Recipe
from src.normalizer import NormalizedToken
from src.recommender import RankedRecipe


def render_normalization_chips(tokens: list[NormalizedToken]) -> str:
    """Render resolved tokens as colored chips; flags low-confidence matches for review."""
    if not tokens:
        return "<em>No ingredients parsed yet.</em>"
    parts = ['<div style="display:flex;flex-wrap:wrap;gap:6px;">']
    for t in tokens:
        stage = t.stage
        color = {
            "exact": "#2a9d8f",
            "lemmatize": "#2a9d8f",
            "alias": "#457b9d",
            "superordinate": "#6a994e",
            "fuzzy": "#e9c46a",
            "semantic": "#f4a261",
            "unresolved": "#e63946",
            "empty": "#adb5bd",
        }.get(stage, "#6c757d")
        label = ", ".join(t.canonical) if t.canonical else "(unresolved)"
        tooltip = f"{stage} · confidence {t.confidence:.2f}"
        mark = " ⚠" if t.needs_confirmation or not t.canonical else ""
        parts.append(
            f'<span title="{escape(tooltip)}" '
            f'style="background:{color};color:white;padding:4px 10px;'
            f'border-radius:14px;font-size:12px;">'
            f'{escape(t.original)} → {escape(label)}{mark}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


def render_low_confidence_prompt(tokens: list[NormalizedToken]) -> str:
    """HTML list describing ambiguous matches the user should confirm."""
    ambiguous = [t for t in tokens if t.needs_confirmation or not t.canonical]
    if not ambiguous:
        return ""
    rows = []
    for t in ambiguous:
        if not t.canonical:
            rows.append(
                f"<li><b>{escape(t.original)}</b>: could not resolve. "
                f"Try a more specific word or remove it.</li>"
            )
            continue
        alts = ", ".join(f"<code>{escape(a)}</code> ({s:.2f})" for a, s in t.candidates[:4])
        rows.append(
            f"<li><b>{escape(t.original)}</b> → <code>{escape(t.canonical[0])}</code> "
            f"(stage: {t.stage}, confidence {t.confidence:.2f}). "
            f"Alternatives: {alts}</li>"
        )
    return (
        "<div style='background:rgba(234, 179, 8, 0.1);padding:10px;border-radius:6px;"
        "border:1px solid rgba(234, 179, 8, 0.3);color:#fde047;'>"
        "<b>Please confirm these low-confidence matches:</b>"
        f"<ul style='margin:6px 0 0 20px;'>{''.join(rows)}</ul></div>"
    )


def _chip(text: str, bg: str, fg: str = "#ffffff",
          border: str = "transparent", symbol: str = "") -> str:
    """Small colored pill. `symbol` is rendered as a prefix (e.g. '✓ carrot')."""
    sym_html = f'<span style="margin-right:4px;font-weight:700;">{symbol}</span>' if symbol else ""
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'padding:3px 9px;border-radius:12px;font-size:12px;margin:2px 3px 2px 0;'
        f'border:1px solid {border};">{sym_html}{escape(text)}</span>'
    )


def _tag_chip(label: str, kind: str) -> str:
    """Cuisine / diet / meal type tag. Different colors per kind."""
    palette = {
        "cuisine": ("rgba(59, 130, 246, 0.2)", "#93c5fd", "rgba(59, 130, 246, 0.4)"),
        "diet":    ("rgba(34, 197, 94, 0.2)", "#86efac", "rgba(34, 197, 94, 0.4)"),
        "meal":    ("rgba(245, 158, 11, 0.2)", "#fcd34d", "rgba(245, 158, 11, 0.4)"),
        "meta":    ("rgba(148, 163, 184, 0.15)", "#cbd5e1", "rgba(148, 163, 184, 0.3)"),
    }
    bg, fg, border = palette.get(kind, palette["meta"])
    return _chip(label, bg=bg, fg=fg, border=border)


def render_result_cards(ranked: list[RankedRecipe]) -> str:
    """Square recipe cards laid out in a responsive grid.

    Colour legend for the ingredient section:
      green   — you have this
      orange  — you're missing this (non-pantry, flagged for the user)
      gray    — pantry staple the recipe calls for (assumed on hand)
      blue    — an ingredient you typed that this recipe doesn't use
    """
    if not ranked:
        return "<em>No recipes matched. Try relaxing filters or adding more ingredients.</em>"

    cards = []
    for i, r in enumerate(ranked):
        rec = r.recipe
        name = escape(rec.name)
        url = escape(rec.url) if rec.url else "#"

        # Square image on top, fixed 4:3 aspect with a subtle placeholder fallback.
        img_html = (
            f'<div style="position:relative;width:100%;aspect-ratio:4/3;'
            f'background:rgba(15,23,42,0.8);border-radius:10px 10px 0 0;overflow:hidden;">'
            f'<img src="{escape(rec.image)}" '
            f'style="width:100%;height:100%;object-fit:cover;display:block;" />'
            f'<div style="position:absolute;top:8px;left:8px;background:rgba(15,23,42,0.8);'
            f'color:white;font-weight:700;font-size:12px;padding:3px 9px;border-radius:12px;'
            f'letter-spacing:0.3px;backdrop-filter:blur(4px);">#{i+1}</div>'
            f'<div style="position:absolute;top:8px;right:8px;background:rgba(30,41,59,0.8);'
            f'color:#f8fafc;font-weight:700;font-size:12px;padding:3px 9px;border-radius:12px;'
            f'box-shadow:0 1px 3px rgba(0,0,0,0.3);backdrop-filter:blur(4px);">score {r.score:.3f}</div>'
            f'</div>'
            if rec.image else
            f'<div style="position:relative;width:100%;aspect-ratio:4/3;'
            f'background:linear-gradient(135deg,rgba(30,41,59,0.8) 0%,rgba(15,23,42,0.8) 100%);'
            f'border-radius:10px 10px 0 0;display:flex;align-items:center;justify-content:center;'
            f'color:#94a3b8;font-size:13px;">no image</div>'
        )

        # Tag chips (cuisine / diet / meal / meta)
        tag_chips: list[str] = []
        for c in rec.cuisine[:3]:
            tag_chips.append(_tag_chip(c.title(), "cuisine"))
        for d in rec.diet_labels[:2]:
            tag_chips.append(_tag_chip(d, "diet"))
        for d in rec.health_labels[:3]:
            tag_chips.append(_tag_chip(d, "diet"))
        for m in rec.meal_type[:2]:
            tag_chips.append(_tag_chip(m, "meal"))
        tag_chips.append(_tag_chip(f"{int(rec.calories)} kcal", "meta"))
        if rec.total_time and rec.total_time > 0:
            tag_chips.append(_tag_chip(f"{int(rec.total_time)} min", "meta"))
        tag_chips.append(_tag_chip(f"cluster {r.cluster}", "meta"))
        tags_html = "".join(tag_chips)

        # Ingredients: have (green), missing non-pantry (orange), pantry (gray)
        ing_chips: list[str] = []
        for ing in r.have:
            ing_chips.append(_chip(ing, bg="#2a9d8f", symbol="✓"))
        for ing in r.missing:
            ing_chips.append(_chip(ing, bg="#ff8c42", symbol="✗"))
        for ing in r.missing_pantry:
            ing_chips.append(_chip(ing, bg="rgba(148,163,184,0.1)", fg="#94a3b8", border="rgba(148,163,184,0.3)"))

        unused_html = ""
        if r.unused_from_user:
            unused_chips = "".join(
                _chip(ing, bg="rgba(99,102,241,0.1)", fg="#818cf8", border="rgba(99,102,241,0.3)")
                for ing in r.unused_from_user
            )
            unused_html = (
                '<div style="margin-top:10px;padding-top:10px;border-top:1px dashed rgba(255,255,255,0.1);">'
                '<div style="font-size:11px;color:#818cf8;font-weight:700;'
                'text-transform:uppercase;letter-spacing:0.4px;margin-bottom:4px;">'
                'Not used by this recipe</div>'
                f'{unused_chips}</div>'
            )

        status_badge = (
            '<span style="color:#2a9d8f;font-weight:700;">Got everything (non-pantry)</span>'
            if not r.missing else
            f'<span style="color:#d35400;font-weight:700;">'
            f'Missing {len(r.missing)}</span>'
        )

        cards.append(
            f"""
<div style="display:flex;flex-direction:column;background:rgba(30,41,59,0.4);
    border:1px solid rgba(255,255,255,0.08);border-radius:12px;overflow:hidden;
    box-shadow:0 8px 32px rgba(0,0,0,0.2);color:#f8fafc;height:100%;backdrop-filter:blur(10px);">
  {img_html}
  <div style="padding:14px;display:flex;flex-direction:column;gap:10px;flex:1;">

    <a href="{url}" target="_blank"
       style="color:#f8fafc;text-decoration:none;font-size:16px;font-weight:700;
       line-height:1.3;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
       overflow:hidden;transition:color 0.2s;" onmouseover="this.style.color='#818cf8'" onmouseout="this.style.color='#f8fafc'">{name}</a>

    <div>{tags_html}</div>

    <div>
      <div style="font-size:11px;color:#94a3b8;font-weight:700;text-transform:uppercase;
           letter-spacing:0.4px;margin-bottom:4px;">
        Ingredients
        <span style="color:#94a3b8;font-weight:500;margin-left:6px;text-transform:none;letter-spacing:0;">
          <span style="color:#34d399;">✓ have</span> ·
          <span style="color:#fbbf24;">✗ missing</span> ·
          <span style="color:#64748b;">pantry</span>
        </span>
      </div>
      <div>{"".join(ing_chips)}</div>
    </div>

    {unused_html}

    <div style="margin-top:auto;padding-top:10px;border-top:1px solid rgba(255,255,255,0.05);
         font-size:12px;display:flex;flex-wrap:wrap;gap:8px;
         align-items:baseline;justify-content:space-between;">
      {status_badge}
      <span style="color:#64748b;font-size:11px;">
        cos {r.cosine:.2f} · jac {r.jaccard:.2f} ·
        cov {r.coverage:.2f} · match {r.match_ratio:.2f}
      </span>
    </div>
  </div>
</div>
"""
        )

    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));'
        'gap:16px;align-items:stretch;">'
        + "".join(cards) +
        "</div>"
    )


def render_single_recipe_card(
    recipe: Recipe,
    cluster_id: int | None = None,
    topic: str | None = None,
) -> str:
    """Render ONE recipe as a standalone horizontal card (image on the left,
    full text on the right). Used by the click-to-explore interaction on the
    cluster scatter plot — occupies the full row so there's space for the
    complete ingredient list with quantities and a link to the source page
    (the dataset doesn't carry cooking steps, so we deep-link to the original
    recipe URL for those)."""
    name = escape(recipe.name)
    url_val = recipe.url or ""

    # Left: square image panel
    if recipe.image:
        img_html = (
            f'<img src="{escape(recipe.image)}" alt="{name}" '
            'style="width:100%;height:100%;object-fit:cover;display:block;" />'
        )
    else:
        img_html = (
            '<div style="width:100%;height:100%;'
            'background:linear-gradient(135deg,rgba(30,41,59,0.8) 0%,rgba(15,23,42,0.8) 100%);'
            'display:flex;align-items:center;justify-content:center;'
            'color:#94a3b8;font-size:14px;">no image</div>'
        )

    # Right: tag chips
    tag_chips: list[str] = []
    for c in recipe.cuisine[:3]:
        tag_chips.append(_tag_chip(c.title(), "cuisine"))
    for d in recipe.diet_labels[:2]:
        tag_chips.append(_tag_chip(d, "diet"))
    for d in recipe.health_labels[:3]:
        tag_chips.append(_tag_chip(d, "diet"))
    for m in recipe.meal_type[:2]:
        tag_chips.append(_tag_chip(m, "meal"))
    if cluster_id is not None:
        label = f"cluster {cluster_id}: {topic}" if topic else f"cluster {cluster_id}"
        tag_chips.append(_tag_chip(label, "meta"))
    tags_html = "".join(tag_chips)

    # Full ingredient list with original quantity text. Dedupe on canonical
    # food name so "1 cup rice" and "more rice" (same food) don't double up.
    ing_lines: list[str] = []
    seen: set[str] = set()
    for item in recipe.ingredients_raw:
        if not isinstance(item, dict):
            continue
        food = str(item.get("food") or "").strip()
        if not food or food in seen:
            continue
        seen.add(food)
        text = str(item.get("text") or food).strip()
        ing_lines.append(
            f'<li style="margin:2px 0;color:#cbd5e1;">{escape(text)}</li>'
        )
    ingredients_html = (
        '<ul style="margin:6px 0 0 0;padding-left:22px;font-size:13px;line-height:1.55;">'
        + "".join(ing_lines)
        + "</ul>"
    )

    # Meta strip: calories · servings · time. Colours are set explicitly on
    # both the <b> and plain-text segments so Gradio's theme CSS can't fade
    # them out in dark mode.
    meta_parts: list[str] = []
    _BOLD = 'style="color:#f8fafc;font-weight:700;"'
    if recipe.calories:
        meta_parts.append(f"<b {_BOLD}>{int(recipe.calories)}</b> kcal")
    if recipe.servings and recipe.servings > 0:
        sv = int(recipe.servings)
        meta_parts.append(f"<b {_BOLD}>{sv}</b> serving{'s' if sv != 1 else ''}")
    if recipe.total_time and recipe.total_time > 0:
        meta_parts.append(f"<b {_BOLD}>{int(recipe.total_time)}</b> min")
    meta_html = " &nbsp;·&nbsp; ".join(meta_parts)

    link_html = ""
    if url_val:
        link_html = (
            f'<a href="{escape(url_val)}" target="_blank" '
            'style="display:inline-block;margin-top:14px;padding:8px 14px;'
            'background:linear-gradient(90deg, #4f46e5 0%, #d946ef 100%);color:#ffffff;border-radius:8px;'
            'text-decoration:none;font-size:13px;font-weight:700;box-shadow:0 4px 12px rgba(0,0,0,0.3);'
            'transition:transform 0.2s;" onmouseover="this.style.transform=\\\'scale(1.02)\\\'" '
            'onmouseout="this.style.transform=\\\'scale(1)\\\'">View full recipe with steps →</a>'
        )
    else:
        link_html = (
            '<div style="margin-top:14px;font-size:12px;color:#64748b;">'
            "<em>No source URL available for this recipe.</em></div>"
        )

    return (
        '<div style="display:flex;gap:20px;background:rgba(30,41,59,0.4);border:1px solid rgba(255,255,255,0.08);'
        'border-radius:16px;overflow:hidden;box-shadow:0 10px 40px rgba(0,0,0,0.25);'
        'color:#f8fafc;width:100%;backdrop-filter:blur(10px);">'
        # Left: image — fixed width, stretches full card height
        '<div style="flex:0 0 320px;min-height:320px;background:rgba(15,23,42,0.8);">'
        f'{img_html}'
        '</div>'
        # Right: textual content, fills remaining space
        '<div style="flex:1;padding:24px;min-width:0;color:#f8fafc;">'
        f'<div style="font-size:24px;font-weight:700;line-height:1.3;color:#f8fafc;">'
        f'{name}</div>'
        f'<div style="margin-top:12px;">{tags_html}</div>'
        + (f'<div style="margin-top:14px;font-size:14px;color:#cbd5e1;">{meta_html}</div>'
           if meta_html else '')
        + '<div style="margin-top:18px;font-size:12px;color:#94a3b8;font-weight:700;'
        'text-transform:uppercase;letter-spacing:0.5px;">Ingredients (with quantities)</div>'
        f'{ingredients_html}'
        f'{link_html}'
        '</div>'
        '</div>'
    )
