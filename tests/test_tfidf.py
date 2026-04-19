"""Correctness vs scikit-learn reference."""
import numpy as np
import pytest

from src.tfidf import TfidfVectorizer, cosine_similarity


DOCS = [
    ["chicken", "onion", "garlic", "tomato"],
    ["beef", "onion", "carrot", "potato"],
    ["chicken", "rice", "soy sauce", "ginger"],
    ["tofu", "soy sauce", "ginger", "scallion"],
    ["flour", "sugar", "butter", "egg"],
    ["flour", "sugar", "butter", "egg", "cocoa"],
]


def _build_vocab(docs):
    return sorted({t for d in docs for t in d})


def test_shape_and_nonnegative():
    vocab = _build_vocab(DOCS)
    vec = TfidfVectorizer(vocab).fit(DOCS)
    X = vec.transform(DOCS)
    assert X.shape == (len(DOCS), len(vocab))
    assert (X.data >= 0).all()


def test_rows_are_unit_norm():
    vocab = _build_vocab(DOCS)
    X = TfidfVectorizer(vocab).fit_transform(DOCS)
    sq = X.multiply(X).sum(axis=1)
    norms = np.sqrt(np.asarray(sq).ravel())
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_idf_downweights_common_tokens():
    """If a token appears in every doc, IDF should be lower than for a rare one."""
    docs = [["common", "rare1"], ["common", "rare2"], ["common", "rare3"]]
    vocab = sorted({"common", "rare1", "rare2", "rare3"})
    vec = TfidfVectorizer(vocab).fit(docs)
    idx_common = vec.token_to_idx["common"]
    idx_rare = vec.token_to_idx["rare1"]
    assert vec.idf[idx_common] < vec.idf[idx_rare]


def test_self_similarity_is_one():
    vocab = _build_vocab(DOCS)
    vec = TfidfVectorizer(vocab).fit(DOCS)
    X = vec.transform(DOCS)
    for i in range(X.shape[0]):
        sims = cosine_similarity(X[i], X)
        assert sims[i] == pytest.approx(1.0, abs=1e-9)


def test_more_overlap_scores_higher():
    """A query that shares more tokens with doc 2 should beat a query aligned with doc 4."""
    vocab = _build_vocab(DOCS)
    vec = TfidfVectorizer(vocab).fit(DOCS)
    X = vec.transform(DOCS)
    q_soup = vec.transform_query(["chicken", "rice", "soy sauce"])
    sims = cosine_similarity(q_soup, X)
    # doc 2 is exactly chicken+rice+soy sauce+ginger → should rank first
    assert int(np.argmax(sims)) == 2


def test_sklearn_cosine_ordering_agrees():
    """We don't match sklearn's tokenizer, but with pre-tokenized inputs our cosine
    *ordering* should agree on a simple test."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos
    except ImportError:
        pytest.skip("scikit-learn not installed")

    corpus = [" ".join(d) for d in DOCS]
    sk = SkTfidf(token_pattern=r"\S+", norm="l2", smooth_idf=True)
    sk_X = sk.fit_transform(corpus)
    q = sk.transform([" ".join(["chicken", "rice", "soy sauce"])])
    sk_sims = sk_cos(q, sk_X).ravel()
    sk_top = int(np.argmax(sk_sims))

    vocab = _build_vocab(DOCS)
    vec = TfidfVectorizer(vocab).fit(DOCS)
    X = vec.transform(DOCS)
    our_q = vec.transform_query(["chicken", "rice", "soy sauce"])
    our_sims = cosine_similarity(our_q, X)
    our_top = int(np.argmax(our_sims))

    assert sk_top == our_top
