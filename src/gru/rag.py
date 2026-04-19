"""Heuristics for drafting a grounded recipe from retrieved examples."""

from collections import Counter
import re

from .tokens import normalize_ingredient_name


_STOPWORDS = {
    "and",
    "or",
    "with",
    "the",
    "a",
    "an",
    "of",
    "in",
    "to",
    "for",
}

_EXTRA_FOOD_WORDS = {
    "broth",
    "cheese",
    "cream",
    "dough",
    "noodle",
    "noodles",
    "pasta",
    "rice",
    "sauce",
    "soup",
    "yogurt",
}

_HOT_METHODS = {"bake", "grill", "simmer", "saute", "boil"}

_DELICATE_INGREDIENTS = {
    "lettuce",
    "tomato",
    "spinach",
    "basil",
    "parsley",
    "cilantro",
    "avocado",
}

_META_STEP_PATTERNS = (
    re.compile(r"^\s*serves?\b", re.I),
    re.compile(r"^\s*yield\b", re.I),
    re.compile(r"\bgood served with\b", re.I),
    re.compile(r"\bserved with\b", re.I),
    re.compile(r"\bgood idea to buy\b", re.I),
    re.compile(r"\bbuy\b", re.I),
)


def _should_attach_to_previous(step):
    """Detect fragments that should be merged into the previous direction step."""
    step = (step or "").strip().lower()
    if not step:
        return False
    if len(step) <= 12:
        return True
    if step.startswith(("and ", "or ", "then ", "until ", "over ", "into ", "to ")):
        return True
    return False


def _append_step(steps, step):
    """Append a cleaned step, merging short fragments when needed."""
    cleaned = re.sub(r"\s+", " ", (step or "").strip())
    if not cleaned:
        return
    if steps and steps[-1].rstrip().lower().endswith(("over", "low", "medium", "high")):
        steps[-1] = steps[-1].rstrip() + " " + cleaned
        return
    if steps and _should_attach_to_previous(cleaned):
        prev = steps[-1].rstrip()
        joiner = "" if prev.endswith(("-", "/", ",")) else " "
        steps[-1] = prev + joiner + cleaned
        return
    steps.append(cleaned)


def _split_user_items(text):
    """Normalize the user's ingredient text into deduplicated ingredient names."""
    parts = re.split(r"[,/\n]| and ", (text or "").lower())
    out = []
    seen = set()
    for part in parts:
        name = normalize_ingredient_name(part)
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _title_case(text):
    return " ".join(word.capitalize() for word in str(text).split())


def _format_series(items, limit=6):
    items = [str(item).strip() for item in (items or []) if str(item).strip()]
    items = items[:limit]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _choose_title(predicted_title, matches, main_items):
    """Prefer the model title, then retrieval titles, while anchoring on a main item."""
    if predicted_title and str(predicted_title).strip():
        base = str(predicted_title).strip()
    elif matches:
        base = matches[0].get("title") or "Pantry Recipe"
    else:
        base = "Pantry Recipe"

    if main_items:
        anchor = _title_case(main_items[0])
        lower = base.lower()
        if main_items[0] not in lower:
            base = f"{anchor} {base}"
    return _title_case(base)


def _rank_candidate_lines(matches, allowed_names):
    """Rank retrieved ingredient lines by frequency and retrieval order."""
    scores = Counter()
    line_by_name = {}
    for match_rank, match in enumerate(matches):
        weight = max(1, len(matches) - match_rank)
        for line in match.get("ingredient_lines") or []:
            name = normalize_ingredient_name(line)
            if not name or name not in allowed_names:
                continue
            scores[name] += weight
            line_by_name.setdefault(name, line)
    ranked = [name for name, _ in scores.most_common()]
    return [line_by_name[name] for name in ranked]


def _step_mentions_name(step, name):
    if not step or not name:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(part) for part in name.split()) + r"\b"
    return re.search(pattern, step.lower()) is not None


def _step_mentions_any(step, names):
    return any(_step_mentions_name(step, name) for name in (names or []) if name)


def _step_main_item_mentions(step, names):
    return sum(1 for name in (names or []) if name and _step_mentions_name(step, name))


def _ingredient_aliases(name):
    parts = [part for part in (name or "").split() if part]
    aliases = set()
    if not parts:
        return aliases
    aliases.add(" ".join(parts))
    if len(parts) > 1:
        aliases.add(parts[-1])
    return {alias for alias in aliases if len(alias) >= 3}


def _match_ingredient_names(match):
    names = set()
    for line in match.get("ingredient_lines") or []:
        name = normalize_ingredient_name(line)
        if name:
            names.add(name)
    return names


def _step_uses_only_allowed_ingredients(step, allowed_names, match_names):
    allowed_aliases = set()
    for name in allowed_names:
        allowed_aliases.update(_ingredient_aliases(name))

    disallowed_aliases = set()
    for name in match_names - allowed_names:
        disallowed_aliases.update(_ingredient_aliases(name))

    disallowed_aliases -= allowed_aliases
    return not any(_step_mentions_name(step, alias) for alias in disallowed_aliases)


def _step_mentions_extra_food(step, allowed_names):
    tokens = set(re.findall(r"[a-z]+", (step or "").lower()))
    allowed_tokens = set()
    for name in allowed_names:
        allowed_tokens.update(re.findall(r"[a-z]+", name.lower()))
        allowed_tokens.update(re.findall(r"[a-z]+", " ".join(_ingredient_aliases(name)).lower()))
    return any(token in _EXTRA_FOOD_WORDS and token not in allowed_tokens for token in tokens)


def _is_meta_step(step):
    step = (step or "").strip()
    if not step:
        return True
    return any(pattern.search(step) for pattern in _META_STEP_PATTERNS)


def _classify_step(step):
    text = (step or "").lower()
    if re.search(r"\bserve|serving platter|plate\b", text):
        return "serve"
    if re.search(r"\bprep|prepare|cut|measure|chop|slice|remove bones|pound\b", text):
        return "prep"
    if re.search(r"\bcook|bake|boil|simmer|steam|saute|grill|heat|transfer|pour|combine|fold|mix|stir|add\b", text):
        return "cook"
    return "other"


def _infer_cooking_method(predicted_title, matches):
    """Infer a broad cooking method from the predicted or retrieved titles."""
    title_text = str(predicted_title or "").lower()
    combined_text = " ".join(
        [title_text] + [str(match.get("title") or "").lower() for match in (matches or [])[:3]]
    )

    for text in (title_text, combined_text):
        if re.search(r"\b(bake|baked|roast|roasted|casserole)\b", text):
            return "bake"
        if re.search(r"\b(grill|grilled|bbq|barbecue)\b", text):
            return "grill"
        if re.search(r"\b(soup|stew|chili|curry|braise|braised)\b", text):
            return "simmer"
        if re.search(r"\b(pasta|noodle|ramen|macaroni)\b", text):
            return "boil"
        if re.search(r"\b(salad|slaw)\b", text):
            return "toss"
        if re.search(r"\b(stir fry|stir-fry|saute|sauteed|skillet|fried)\b", text):
            return "saute"
    return "cook"


def _default_prep_step(main_items):
    if main_items:
        return (
            "Prep the main ingredients: "
            + _format_series(main_items)
            + ". Cut or measure them before cooking."
        )
    return "Gather and measure the ingredients before you start cooking."


def _default_setup_step(method, main_items, pantry_items):
    target = _format_series(main_items[:4]) or "the main ingredients"
    seasoners = [
        item
        for item in (pantry_items or [])
        if item in {"salt", "pepper", "black pepper", "olive oil", "vegetable oil", "oil", "butter", "garlic powder"}
    ]
    seasoning_text = f" with {_format_series(seasoners, limit=4)}" if seasoners else ""

    if method == "bake":
        return f"Preheat the oven to 400 F. Season {target}{seasoning_text} and arrange them in a lightly oiled baking dish or sheet pan."
    if method == "grill":
        return f"Heat the grill or a grill pan to medium-high. Season {target}{seasoning_text} so everything is ready to cook."
    if method == "simmer":
        return f"Set a pot over medium heat and combine {target}{seasoning_text} with enough liquid or base ingredients to cook gently."
    if method == "boil":
        return f"Bring a pot of salted water to a boil, then season and organize {target}{seasoning_text} so they can be added in stages."
    if method == "toss":
        return f"Combine {target}{seasoning_text} in a large bowl and get any dressing or pantry additions ready."
    return f"Season {target}{seasoning_text} and combine them so they are ready to cook evenly."


def _default_cook_step(method, main_items):
    anchor = _format_series(main_items[:2]) or "the main ingredients"
    if method == "bake":
        if len(main_items) <= 1:
            return f"Bake until {anchor} is cooked through and any vegetables are tender, turning or stirring once if needed for even browning."
        return "Bake until the main ingredients are cooked through and any vegetables are tender, turning or stirring once if needed for even browning."
    if method == "grill":
        return f"Grill {anchor} until nicely marked and cooked through, flipping once and adjusting the heat if anything browns too quickly."
    if method == "simmer":
        return f"Simmer gently until {anchor} is tender and the broth or sauce has developed good flavor."
    if method == "boil":
        return f"Cook until {anchor} is just tender, then drain or reduce the liquid so the final dish is not watery."
    if method == "toss":
        return "Toss until everything is evenly coated, then chill briefly if you want the flavors to come together before serving."
    return "Cook over medium heat until the main ingredients are tender and the flavors have come together."


def _default_finish_step(method, main_items):
    delicate = [item for item in (main_items or []) if item in _DELICATE_INGREDIENTS]
    delicate_text = _format_series(delicate[:3])
    if method in _HOT_METHODS and delicate_text:
        return f"Add delicate ingredients such as {delicate_text} near the end or serve them on the side so they stay fresh."
    if method == "toss":
        return "Taste, adjust the seasoning, and serve chilled or at cool room temperature."
    return "Taste, adjust the seasoning if needed, and serve while everything is warm."


def _default_bridge_step(method):
    if method == "bake":
        return "Check the pan partway through cooking and spoon any juices over the top so the dish stays flavorful."
    if method == "simmer":
        return "Stir occasionally and adjust the heat so the mixture stays at a steady simmer rather than boiling hard."
    if method == "boil":
        return "Taste as you go and reserve a little cooking liquid if the dish needs moisture at the end."
    return "Check the seasoning and texture as you cook, adding a small splash of water or a little more seasoning if needed."


def _step_conflicts_with_method(step, method):
    text = (step or "").lower()
    if method in _HOT_METHODS and re.search(r"\b(chill|refrigerate|cold)\b", text):
        return True
    if method != "toss" and re.search(r"\bdressing\b", text):
        return True
    return False


def _generate_ingredient_lines(main_items, pantry_items, matches):
    """Build an ingredient list grounded in retrieval matches plus user inputs."""
    allowed = list(dict.fromkeys(main_items + pantry_items))
    chosen = _rank_candidate_lines(matches, set(allowed))

    existing = {normalize_ingredient_name(line) for line in chosen}
    for name in allowed:
        if name not in existing:
            if name in pantry_items:
                chosen.append(f"Pantry: {name}")
            else:
                chosen.append(name)
    return chosen[:16]


def _generate_directions(main_items, pantry_items, matches, predicted_title=None):
    """Build a fuller direction list grounded in retrieval plus a structured fallback flow."""
    allowed_names = set(main_items + pantry_items)
    method = _infer_cooking_method(predicted_title, matches)
    steps = []
    seen = set()

    def add_step(step):
        cleaned = re.sub(r"\s+", " ", (step or "").strip())
        if not cleaned:
            return False
        key = cleaned.lower()
        if key in seen:
            return False
        seen.add(key)
        _append_step(steps, cleaned)
        return True

    add_step(_default_prep_step(main_items))
    add_step(_default_setup_step(method, main_items, pantry_items))

    saw_serve = False
    added_cook_steps = 0
    for match in matches[:2]:
        match_names = _match_ingredient_names(match)
        for step in match.get("directions") or []:
            cleaned = step.strip()
            if _is_meta_step(cleaned):
                continue
            if not _step_uses_only_allowed_ingredients(cleaned, allowed_names, match_names):
                continue
            if _step_mentions_extra_food(cleaned, allowed_names):
                continue
            if _step_conflicts_with_method(cleaned, method):
                continue
            phase = _classify_step(cleaned)
            if phase == "prep" and main_items:
                mentions_anchor = _step_mentions_name(cleaned, main_items[0])
                mentions_multiple = _step_main_item_mentions(cleaned, main_items) >= 2
                if method in _HOT_METHODS and not (mentions_anchor or mentions_multiple):
                    continue
                if not mentions_anchor and _step_main_item_mentions(cleaned, main_items) == 0:
                    continue
            if phase == "other" and not _step_mentions_any(cleaned, main_items):
                continue
            if saw_serve and phase in {"prep", "cook"}:
                continue
            if phase == "prep" and len(steps) > 2:
                continue
            if phase == "cook" and added_cook_steps >= 2:
                continue
            if add_step(cleaned):
                if phase == "cook":
                    added_cook_steps += 1
                if phase == "serve":
                    saw_serve = True
            if len(steps) >= 6:
                break
        if len(steps) >= 6:
            break

    if added_cook_steps == 0:
        add_step(_default_cook_step(method, main_items))
    if len(steps) < 4:
        add_step(_default_bridge_step(method))
    if not saw_serve:
        add_step(_default_finish_step(method, main_items))
    elif len(steps) < 5:
        add_step(_default_bridge_step(method))
    return steps[:8]


def generate_recipe_draft(user_text, pantry_items, matches, predicted_title=None):
    """Create a lightweight recipe draft for the web UI."""
    matches = [m for m in (matches or []) if isinstance(m, dict) and not m.get("error")]
    main_items = _split_user_items(user_text)
    pantry_items = [normalize_ingredient_name(x) for x in (pantry_items or [])]
    pantry_items = [x for x in pantry_items if x]

    title = _choose_title(predicted_title, matches, main_items)
    ingredient_lines = _generate_ingredient_lines(main_items, pantry_items, matches)
    directions = _generate_directions(main_items, pantry_items, matches, predicted_title=title)

    notes = []
    if matches:
        notes.append("Grounded in retrieved recipes from the training corpus.")
    else:
        notes.append("Built from the user inputs without supporting retrieval examples.")

    return {
        "title": title,
        "ingredient_lines": ingredient_lines,
        "directions": directions,
        "notes": notes,
    }
