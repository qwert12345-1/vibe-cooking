"""Typo, alias, and translation test cases."""
import pytest

from src.normalizer import Normalizer, flatten_resolved


VOCAB = [
    "chicken", "beef", "pork", "tomato", "onion", "garlic", "ginger",
    "soy sauce", "rice", "noodle", "carrot", "potato", "bell pepper",
    "mushroom", "egg", "shrimp", "fish", "cilantro", "green onion",
    "chili pepper", "eggplant", "zucchini",
]


@pytest.fixture(scope="module")
def norm():
    # Disable SBERT so tests run fast and have no network dependency.
    return Normalizer(VOCAB, use_semantic=False)


def test_exact_match(norm):
    t = norm.normalize("chicken")
    assert t.canonical == ["chicken"]
    assert t.stage == "exact"
    assert t.confidence == 1.0


def test_lemmatize_plural(norm):
    t = norm.normalize("tomatoes")
    assert t.canonical == ["tomato"]
    # Could be resolved either by alias (which we hand-curated) or lemmatization.
    assert t.stage in ("lemmatize", "alias", "exact")


def test_alias_chinese_translation(norm):
    t = norm.normalize("洋葱")
    assert t.canonical == ["onion"]
    assert t.stage == "alias"


def test_alias_british_to_american(norm):
    t = norm.normalize("aubergine")
    assert t.canonical == ["eggplant"]
    assert t.stage == "alias"


def test_fuzzy_typo(norm):
    # "chiken" → "chicken" — requires rapidfuzz
    pytest.importorskip("rapidfuzz")
    t = norm.normalize("chiken")
    assert t.canonical == ["chicken"]
    assert t.stage == "fuzzy"
    assert t.confidence >= 0.8


def test_superordinate_expansion(norm):
    t = norm.normalize("poultry")
    assert "chicken" in t.canonical
    assert t.stage in ("superordinate", "alias")


def test_unresolved_returns_empty(norm):
    t = norm.normalize("qqqqzzz123")
    assert t.canonical == []
    assert t.stage == "unresolved"


def test_flatten_dedupes():
    tokens = [
        type("T", (), dict(canonical=["onion"], original="", stage="", confidence=1.0, candidates=[]))(),
        type("T", (), dict(canonical=["onion", "garlic"], original="", stage="", confidence=1.0, candidates=[]))(),
    ]
    # Can't use the real NormalizedToken here because of dataclass field checks;
    # construct simple stand-ins with compatible attrs.
    out = flatten_resolved(tokens)
    assert out == ["onion", "garlic"]
