"""Main recommendation engine: filter → recall → rank.

Usage:

    engine = RecipeRecommender.build_or_load()
    result = engine.recommend(
        ingredients=["chicken", "tomato", "洋葱"],
        top_k=10,
        filters=Filters(cuisine=["italian"], max_calories=500),
    )
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy import sparse

from . import constants
from .constants import CLUSTER_TOPIC_CANDIDATES, PANTRY_STAPLES
from .embeddings import DEFAULT_SBERT_MODEL, get_sbert_model
from .data_loader import (
    CACHE_DIR,
    Recipe,
    build_vocabulary,
    load_recipes,
    load_vocab,
    save_vocab,
)
from .kmeans import KMeansResult, choose_k_silhouette, fit_kmeans, silhouette_analysis
from .normalizer import NormalizedToken, Normalizer, flatten_resolved
from .scoring import (
    coverage,
    fused_score,
    jaccard,
    match_ratio,
    missing_ingredients,
)
from .tfidf import TfidfVectorizer, cosine_similarity


# ----------------------------------------------------------------------
# Filter + result types
# ----------------------------------------------------------------------
@dataclass
class Filters:
    cuisine: list[str] = field(default_factory=list)
    diet: list[str] = field(default_factory=list)
    meal_type: list[str] = field(default_factory=list)
    excluded_ingredients: list[str] = field(default_factory=list)
    max_calories: float | None = None
    max_time_minutes: float | None = None


@dataclass
class RankedRecipe:
    recipe: Recipe
    score: float
    cosine: float
    jaccard: float
    coverage: float
    match_ratio: float
    missing: list[str]               # non-pantry ingredients the recipe needs but user lacks
    missing_pantry: list[str]        # pantry items the recipe needs but user didn't list
    have: list[str]                  # ingredients of the recipe the user already has
    unused_from_user: list[str]      # user's ingredients this recipe doesn't use
    cluster: int


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------
class RecipeRecommender:
    def __init__(
        self,
        recipes: list[Recipe],
        vocabulary: list[str],
        vectorizer: TfidfVectorizer,
        recipe_matrix: sparse.csr_matrix,
        cluster_labels: np.ndarray,
        projection_2d: np.ndarray,
        cluster_embeddings: np.ndarray | None = None,
        svd_components: np.ndarray | None = None,
        viz_projector=None,
        recipe_sbert: np.ndarray | None = None,
        cluster_topics: dict[int, str] | None = None,
        cluster_embed_source: str = "svd",  # "svd" | "sbert"
        normalizer: Normalizer | None = None,
    ):
        self.recipes = recipes
        self.vocabulary = vocabulary
        self.vectorizer = vectorizer
        self.recipe_matrix = recipe_matrix
        self.cluster_labels = cluster_labels
        self.projection_2d = projection_2d

        # `cluster_embeddings` is the high-dim space K-Means was actually fit on.
        # `projection_2d` is its 2-D projection used only for visualization.
        # If not supplied (old caches), fall back to the 2-D projection.
        self.cluster_embeddings = (
            cluster_embeddings if cluster_embeddings is not None else projection_2d
        )

        # `svd_components` is the V basis (vocab_size, n_components) from the SVD
        # used to build `cluster_embeddings` — lets us project new query vectors
        # into the same 2-D space where the recipes live.
        self.svd_components = svd_components

        # LDA (or fallback) transformer from high-dim cluster space → 2-D.
        # Supervised on cluster labels so clusters visually separate.
        self.viz_projector = viz_projector

        # Dense SBERT recipe embeddings (N, 384), L2-normalized. Enables hybrid
        # retrieval (TF-IDF + SBERT cosine fusion) and semantic cluster labels.
        # Optional — old caches built before this change have None here.
        self.recipe_sbert = recipe_sbert

        # Human-readable topic label per cluster, picked by matching cluster
        # mean embedding to `CLUSTER_TOPIC_CANDIDATES`.
        self.cluster_topics = cluster_topics or {}

        # Which feature space K-Means was actually fit on. Determines how we
        # encode a user query into the same space for `project_user_position`.
        self.cluster_embed_source = cluster_embed_source
        self.normalizer = normalizer or Normalizer(vocabulary)
        self._ingredient_sets: list[set[str]] = [set(r.ingredients) for r in recipes]
        
        # Per-recipe mass map {food: grams} for non-pantry ingredients only.
        # Used by `fused_score` to make coverage mass-weighted — e.g. a
        # recipe's 500 g of chicken counts much more than its 5 g of basil,
        # so a chicken-centric dish ranks above a caprese for a user who
        # has both chicken and basil. Built once here, read on every query.
        self._ingredient_weights: list[dict[str, float]] = [
            _build_ingredient_weights(r) for r in recipes
        ]

        # Precompute which vocabulary columns correspond to pantry staples — used
        # both when building the clustering space and when labeling clusters.
        self._pantry_col_idx = np.array(
            [vectorizer.token_to_idx[t] for t in PANTRY_STAPLES
             if t in vectorizer.token_to_idx],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------
    @classmethod
    def build_or_load(
        cls,
        *,
        force_rebuild: bool = False,
        recipe_limit: int | None = None,
        k_range: tuple[int, int] = (8, 50),
        use_semantic_normalizer: bool = True,
    ) -> "RecipeRecommender":
        cache = Path(CACHE_DIR) / "engine.pkl"
        if cache.exists() and not force_rebuild:
            print("[build] Loading engine from cache/engine.pkl...", flush=True)
            with open(cache, "rb") as f:
                payload = pickle.load(f)
            print("[build] Cache loaded. Initializing normalizer...", flush=True)
            return cls(
                recipes=payload["recipes"],
                vocabulary=payload["vocabulary"],
                vectorizer=payload["vectorizer"],
                recipe_matrix=payload["recipe_matrix"],
                cluster_labels=payload["cluster_labels"],
                projection_2d=payload["projection_2d"],
                cluster_embeddings=payload.get("cluster_embeddings"),
                svd_components=payload.get("svd_components"),
                viz_projector=payload.get("viz_projector"),
                recipe_sbert=payload.get("recipe_sbert"),
                cluster_topics=payload.get("cluster_topics"),
                cluster_embed_source=payload.get("cluster_embed_source", "svd"),
                normalizer=Normalizer(payload["vocabulary"], use_semantic=use_semantic_normalizer),
            )

        print("[build] Loading recipes from HuggingFace (this may take a few minutes)...")
        recipes = load_recipes(use_cache=True, limit=recipe_limit)
        print(f"[build] Loaded {len(recipes)} recipes")

        print("[build] Building ingredient vocabulary...")
        vocab = load_vocab()
        if vocab is None or force_rebuild:
            vocab = build_vocabulary(recipes, min_df=3)
            save_vocab(vocab)
        print(f"[build] Vocabulary size: {len(vocab)}")

        print("[build] Fitting TF-IDF...")
        vectorizer = TfidfVectorizer(vocab)
        matrix = vectorizer.fit_transform([r.ingredients for r in recipes])
        print(f"[build] TF-IDF matrix: {matrix.shape}, nnz={matrix.nnz}")

        # Always build SVD space — used either as the clustering space OR as
        # the query-projection path for old caches. Cheap (~seconds).
        print("[build] Building SVD-of-TF-IDF space (pantry-masked)...")
        svd_space, svd_components = _build_cluster_embeddings(
            matrix, vectorizer, n_components=50
        )
        print(f"[build] SVD space: {svd_space.shape}")

        # ------- SBERT recipe embeddings (before clustering + viz) ----------
        # SBERT encodes "what kind of dish this is" (semantic meaning), while
        # TF-IDF encodes "what's in it" (ingredient composition). For clustering
        # into dish-type groups (soups vs stir-fries vs desserts), SBERT is
        # qualitatively better — two recipes using "egg + butter + milk" could
        # be a cheesecake and a carbonara, which TF-IDF treats as similar and
        # SBERT correctly separates.
        import os as _os
        recipe_sbert = None
        cluster_topics: dict[int, str] = {}
        partial_cache_path = cache.parent / "sbert_partial.pkl"
        sbert_disabled = _os.environ.get("RR_NO_SBERT_RECIPES", "0") == "1"

        if sbert_disabled:
            print("[build] RR_NO_SBERT_RECIPES=1 — skipping SBERT recipe encoding.")
        else:
            if partial_cache_path.exists():
                try:
                    with open(partial_cache_path, "rb") as f:
                        partial = pickle.load(f)
                    candidate = partial.get("recipe_sbert")
                    if candidate is not None and candidate.shape[0] == len(recipes):
                        recipe_sbert = candidate
                        print(f"[build] Resumed SBERT from partial cache "
                              f"({recipe_sbert.shape}). Skipping re-encode.")
                except Exception as e:
                    print(f"[build] Partial SBERT cache unreadable ({e}); rebuilding.")

            if recipe_sbert is None:
                print("[build] Encoding recipes with SBERT "
                      "(~5-10 min on CPU, cached afterwards)...")
                recipe_sbert = _encode_recipes_sbert(recipes)
                if recipe_sbert is None:
                    print("[build] SBERT unavailable; falling back to SVD clustering.")
                else:
                    print(f"[build] Recipe SBERT matrix: {recipe_sbert.shape}")
                    partial_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(partial_cache_path, "wb") as f:
                        pickle.dump({"recipe_sbert": recipe_sbert}, f)
                    print(f"[build] Saved partial SBERT cache → {partial_cache_path}")

        # ------- Pick clustering space -------------------------------------
        # Default: SBERT (semantic, much better cluster coherence).
        # Fallback: SVD of TF-IDF (works even without SBERT).
        requested = _os.environ.get("RR_CLUSTER_EMBED", "sbert").lower()
        if requested == "sbert" and recipe_sbert is not None:
            cluster_space = recipe_sbert.astype(np.float64, copy=False)
            cluster_embed_source = "sbert"
            print(f"[build] Clustering space: SBERT embeddings {cluster_space.shape}")
        else:
            cluster_space = svd_space
            cluster_embed_source = "svd"
            print(f"[build] Clustering space: SVD of TF-IDF {cluster_space.shape}"
                  f"{' (SBERT unavailable)' if requested == 'sbert' else ''}")

        # ------- Silhouette + K-Means on chosen space ----------------------
        print("[build] Choosing k via silhouette analysis on the clustering space...")
        ks = list(range(k_range[0], k_range[1] + 1, 2))
        silhouette_sample = min(cluster_space.shape[0], 2500)
        silhouette_scores = silhouette_analysis(
            cluster_space, ks, seed=42, n_init=1,
            sample_size=silhouette_sample, metric="cosine",
        )
        k = choose_k_silhouette(silhouette_scores)
        print(f"[build] Selected k={k}. Silhouette scores: {silhouette_scores}")

        print(f"[build] Fitting spherical K-Means with k={k} in {cluster_space.shape[1]}-D...")
        km = fit_kmeans(cluster_space, k=k, n_init=5, seed=42, metric="cosine")

        # ------- Semantic cluster labels (always from SBERT, if we have it) -
        if recipe_sbert is not None:
            print("[build] Assigning semantic topic labels to clusters...")
            cluster_topics = _label_clusters_semantically(recipe_sbert, km.labels)
            for cid in sorted(cluster_topics):
                print(f"  cluster {cid}: {cluster_topics[cid]}")
            # Re-save partial cache with topics attached.
            partial_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(partial_cache_path, "wb") as f:
                pickle.dump(
                    {"recipe_sbert": recipe_sbert, "cluster_topics": cluster_topics},
                    f,
                )

        # ------- 2-D viz projection on the same space K-Means used ---------
        viz_method = _os.environ.get("RR_VIZ", "pca")
        print(f"[build] Building 2-D viz projection (method={viz_method!r})...")
        projection, viz_projector = _build_viz_projection(
            cluster_space, km.labels, method=viz_method
        )

        engine = cls(
            recipes=recipes,
            vocabulary=vocab,
            vectorizer=vectorizer,
            recipe_matrix=matrix,
            cluster_labels=km.labels,
            projection_2d=projection,
            cluster_embeddings=cluster_space,
            svd_components=svd_components,
            viz_projector=viz_projector,
            recipe_sbert=recipe_sbert,
            cluster_topics=cluster_topics,
            cluster_embed_source=cluster_embed_source,
            normalizer=Normalizer(vocab, use_semantic=use_semantic_normalizer),
        )

        print("[build] Saving engine cache...")
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(
                {
                    "recipes": recipes,
                    "vocabulary": vocab,
                    "vectorizer": vectorizer,
                    "recipe_matrix": matrix,
                    "cluster_labels": km.labels,
                    "projection_2d": projection,
                    "cluster_embeddings": cluster_space,
                    "svd_components": svd_components,
                    "viz_projector": viz_projector,
                    "recipe_sbert": recipe_sbert,
                    "cluster_topics": cluster_topics,
                    "cluster_embed_source": cluster_embed_source,
                },
                f,
            )
        print("[build] Done.")
        return engine

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def _filter_indices(self, filters: Filters) -> np.ndarray:
        keep = np.ones(len(self.recipes), dtype=bool)
        excluded_set = {e.lower().strip() for e in filters.excluded_ingredients if e.strip()}
        for i, r in enumerate(self.recipes):
            if filters.cuisine:
                wanted = {c.lower() for c in filters.cuisine}
                if not (set(r.cuisine) & wanted):
                    keep[i] = False
                    continue
            if filters.diet:
                wanted = {d.lower() for d in filters.diet}
                have = {d.lower() for d in (r.diet_labels + r.health_labels)}
                if not (have & wanted):
                    keep[i] = False
                    continue
            if filters.meal_type:
                wanted = {m.lower() for m in filters.meal_type}
                if not (set(r.meal_type) & wanted):
                    keep[i] = False
                    continue
            if excluded_set:
                ingset = self._ingredient_sets[i]
                if excluded_set & ingset:
                    keep[i] = False
                    continue
            if filters.max_calories is not None and r.calories > filters.max_calories:
                keep[i] = False
                continue
            if filters.max_time_minutes is not None and r.total_time and r.total_time > filters.max_time_minutes:
                keep[i] = False
                continue
        return np.where(keep)[0]

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------
    def normalize(self, raw_ingredients: list[str]) -> list[NormalizedToken]:
        return self.normalizer.normalize_batch(raw_ingredients)

    def recommend(
        self,
        ingredients: list[str] | list[NormalizedToken],
        *,
        top_k: int = 10,
        filters: Filters | None = None,
        candidate_pool: int = 500,
    ) -> list[RankedRecipe]:
        filters = filters or Filters()

        # Resolve ingredients if the caller passed raw strings.
        if ingredients and isinstance(ingredients[0], str):
            normalized = self.normalize(ingredients)  # type: ignore[arg-type]
        else:
            normalized = ingredients  # type: ignore[assignment]
        resolved = flatten_resolved(normalized)
        if not resolved:
            return []

        user_set = set(resolved)

        # Stage 1 — hard filter
        candidate_idx = self._filter_indices(filters)
        if candidate_idx.size == 0:
            return []

        # Stage 2 — recall. Hybrid: TF-IDF cosine + optional SBERT dense cosine,
        # fused by weighted sum of normalized scores. TF-IDF captures exact
        # ingredient overlap (rare-ingredient signal from IDF); SBERT captures
        # semantic similarity ("poultry" ≈ "chicken", "creamy pasta" ≈ "alfredo")
        # — they complement each other.
        query = self.vectorizer.transform_query(resolved)
        sub_matrix = self.recipe_matrix[candidate_idx]
        tfidf_sims = cosine_similarity(query, sub_matrix)

        sbert_sims = None
        if self.recipe_sbert is not None:
            try:
                q_sbert = self._encode_query_sbert(resolved)
                if q_sbert is not None:
                    sbert_sims = self.recipe_sbert[candidate_idx] @ q_sbert
            except Exception as e:
                print(f"[recommend] SBERT dense retrieval failed ({e}); "
                      "falling back to TF-IDF only.")
                sbert_sims = None

        if sbert_sims is not None:
            # Both scores are cosine on L2-normalized vectors → already in [-1, 1].
            # Min-max scale each to [0, 1] for stable fusion across queries where
            # the absolute scales differ by orders of magnitude.
            sims = 0.55 * _minmax(tfidf_sims) + 0.45 * _minmax(sbert_sims)
        else:
            sims = tfidf_sims

        # Narrow to the top `candidate_pool` for the slower set-based re-rank.
        if sims.size > candidate_pool:
            top_local = np.argpartition(-sims, candidate_pool - 1)[:candidate_pool]
        else:
            top_local = np.arange(sims.size)

        # Stage 3 — rerank with fused score. The scoring uses *TF-IDF cosine*
        # as the continuous relevance signal (it's on a well-understood scale),
        # but the candidate pool above already came from TF-IDF+SBERT hybrid
        # recall, so the fused_score is acting on the better shortlist.
        # Precompute per-query-token IDF weights so weighted_match_ratio can
        # reward recipes that use the user's RARE ingredients (saffron, miso)
        # over recipes that only use common ones (salt).
        user_idf_weights: dict[str, float] = {}
        for tok in user_set:
            idx = self.vectorizer.token_to_idx.get(tok)
            if idx is not None and self.vectorizer.idf is not None:
                user_idf_weights[tok] = float(self.vectorizer.idf[idx])
        ranked: list[RankedRecipe] = []
        for local_i in top_local:
            global_i = int(candidate_idx[int(local_i)])
            cos = float(tfidf_sims[int(local_i)])
            recipe_set = self._ingredient_sets[global_i]
            recipe_obj = self.recipes[global_i]
            # Intentionally NOT passing recipe_weights: mass-weighted coverage
            # lets a heavy-mass ingredient (chicken = 1500g) hide missing
            # small-mass ones (rosemary = 2g), causing "Roast Chicken" to
            # fake 93% coverage while actually missing 4 of 6 non-pantry
            # ingredients. Binary coverage treats every missing ingredient
            # as a real gap; the "main ingredient" intuition is carried by
            # name_match_bonus (recipe-name mentions) and weighted_match_ratio
            # (IDF of user's query tokens) instead.
            score = fused_score(
                cos, user_set, recipe_set,
                user_weights=user_idf_weights or None,
                recipe_name=recipe_obj.name,
            )

            # Per-recipe set breakdowns used by the UI cards.
            have = sorted(user_set & recipe_set)
            missing_non_pantry = missing_ingredients(user_set, recipe_set)
            missing_pantry_list = sorted(
                [ing for ing in recipe_set
                 if ing not in user_set and ing in PANTRY_STAPLES]
            )
            unused = sorted(user_set - recipe_set)

            ranked.append(
                RankedRecipe(
                    recipe=self.recipes[global_i],
                    score=score,
                    cosine=cos,
                    jaccard=jaccard(user_set, recipe_set),
                    coverage=coverage(user_set, recipe_set),
                    match_ratio=match_ratio(user_set, recipe_set),
                    missing=missing_non_pantry,
                    missing_pantry=missing_pantry_list,
                    have=have,
                    unused_from_user=unused,
                    cluster=int(self.cluster_labels[global_i]),
                )
            )

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # SBERT query encoding (for hybrid retrieval)
    # ------------------------------------------------------------------
    def _encode_query_sbert(self, resolved_tokens: list[str]) -> np.ndarray | None:
        """Encode a normalized ingredient list with SBERT, L2-normalized.

        Reuses the Normalizer's SBERT model so the app keeps one copy in memory.
        Returns None if SBERT isn't available.
        """
        model = getattr(self.normalizer, "_semantic", None)
        if model is None:
            model = get_sbert_model()
        if model is None:
            return None
        text = ", ".join(resolved_tokens)
        vec = model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float64)

    # ------------------------------------------------------------------
    # Cluster topic label (semantic name)
    # ------------------------------------------------------------------
    def cluster_topic(self, cluster_id: int) -> str | None:
        """Return the semantic topic label for a cluster, or None if unknown."""
        return self.cluster_topics.get(int(cluster_id))

    # ------------------------------------------------------------------
    # Query projection (for placing the user's input on the 2-D map)
    # ------------------------------------------------------------------
    def project_user_position(
        self, ingredients: list[str] | list[NormalizedToken]
    ) -> tuple[float, float] | None:
        """Return the 2-D coordinate where the user's input vector lands.

        Uses the SVD V basis stored from `cluster_embeddings` construction.
        Pantry columns are zeroed — mirroring how the corpus itself was
        embedded — so the user's point sits in the same space as the recipes.

        Returns None if the query resolves to zero tokens or if we don't have
        the SVD basis (old caches built before this change).
        """
        if self.svd_components is None:
            return None
        if ingredients and isinstance(ingredients[0], str):
            tokens = self.normalize(ingredients)  # type: ignore[arg-type]
        else:
            tokens = ingredients  # type: ignore[assignment]
        resolved = flatten_resolved(tokens)
        if not resolved:
            return None

        # Encode the query into the same feature space K-Means was fit on.
        # This matters for dropping the "you" marker at a meaningful spot on
        # the 2-D map — if clustering was SBERT-based, we must SBERT-encode
        # the query; if SVD-based, we use TF-IDF → pantry-mask → SVD basis.
        q_cluster: np.ndarray | None = None

        if self.cluster_embed_source == "sbert":
            q_cluster = self._encode_query_sbert(resolved)
            # _encode_query_sbert returns a 384-D (or whatever SBERT dim) vector;
            # the viz_projector (KNN) was fit in the same space, so this works.

        if q_cluster is None:
            # SVD path (either the engine clusters on SVD, or SBERT encode failed).
            if self.svd_components is None:
                return None
            q = self.vectorizer.transform_query(resolved)
            q_dense = np.asarray(q.todense()).ravel()
            if self._pantry_col_idx.size > 0:
                q_dense[self._pantry_col_idx] = 0.0
            q_cluster = q_dense @ self.svd_components

        if self.viz_projector is not None:
            try:
                q_viz = self.viz_projector.transform(q_cluster.reshape(1, -1))[0]
                return float(q_viz[0]), float(q_viz[1])
            except Exception:
                pass
        return float(q_cluster[0]), float(q_cluster[1])

    # ------------------------------------------------------------------
    # Cluster browsing
    # ------------------------------------------------------------------
    def cluster_neighbors(self, recipe_index: int, limit: int = 12) -> list[Recipe]:
        cid = int(self.cluster_labels[recipe_index])
        mask = self.cluster_labels == cid
        indices = np.where(mask)[0]
        # Sort by distance in 2-D projection space.
        anchor = self.projection_2d[recipe_index]
        dists = np.linalg.norm(self.projection_2d[indices] - anchor, axis=1)
        order = indices[np.argsort(dists)]
        return [self.recipes[int(i)] for i in order[:limit]]

    def cluster_label_guess(self, cluster_id: int, top_terms: int = 3) -> str:
        """Human-readable label for a cluster.

        Preferred form: "<semantic topic> · <top distinctive non-pantry ingredients>"
        (e.g. "stir-fries · chicken, soy sauce, ginger"). The topic comes from
        `cluster_topics` (built by matching cluster mean SBERT embedding to a
        curated candidate list). If SBERT wasn't available at build time, we
        fall back to the ingredients-only label.

        Pantry staples are always excluded from the ingredient half — otherwise
        clusters differ mostly by "how much salt/oil" and the labels read like
        a condiment shelf.
        """
        mask = self.cluster_labels == cluster_id
        if not mask.any():
            return f"cluster-{cluster_id}"
        sub = self.recipe_matrix[mask]
        mean_in = np.asarray(sub.mean(axis=0)).ravel()
        mean_out = np.asarray(self.recipe_matrix.mean(axis=0)).ravel()
        distinctive = mean_in - mean_out
        if self._pantry_col_idx.size > 0:
            distinctive[self._pantry_col_idx] = -np.inf
        top = np.argsort(-distinctive)[:top_terms]
        ingredient_part = ", ".join(self.vocabulary[int(i)] for i in top)
        topic = self.cluster_topics.get(int(cluster_id))
        if topic:
            return f"{topic} · {ingredient_part}"
        return ingredient_part


# ----------------------------------------------------------------------
# Clustering / visualization spaces
# ----------------------------------------------------------------------
def _build_cluster_embeddings(
    matrix: sparse.csr_matrix,
    vectorizer: TfidfVectorizer,
    n_components: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the space K-Means is actually fit on.

    Two intentional design choices:

    1. **High-dim, not 2-D.** We keep 50 SVD components. Clustering in 2-D throws
       away almost all of the signal TF-IDF encoded — natural recipe groupings
       (baked goods, stir-fries, soups) don't all separate along the first two
       principal components. The 2-D scatter is built later by slicing the first
       two components *of this same space*, so the visualization stays coherent
       with what K-Means saw.

    2. **Pantry staples are zeroed out before SVD.** IDF downweights them but
       doesn't kill them; and because they co-occur in the majority of recipes,
       even small per-cluster variations in their weights dominate the leading
       principal components. Masking them gives the "main ingredient" signal
       room to express itself in the top SVD dimensions.
    """
    from scipy.sparse.linalg import svds

    # Mask pantry columns to zero. CSR → LIL for column editing, then back to CSR.
    pantry_cols = [
        vectorizer.token_to_idx[t]
        for t in PANTRY_STAPLES
        if t in vectorizer.token_to_idx
    ]
    if pantry_cols:
        masked = matrix.tolil(copy=True)
        for c in pantry_cols:
            masked[:, c] = 0
        masked = masked.tocsr()
        masked.eliminate_zeros()
    else:
        masked = matrix

    k = min(n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
    try:
        u, s, vt = svds(masked.astype(np.float64), k=k, random_state=seed)
        # svds returns singular values ascending; reorder descending.
        order = np.argsort(-s)
        coords = (u * s)[:, order]
        # V basis (vocab_size, n_components). We reorder columns the same way so
        # that `(q_pantry_masked @ components) ≈ u_q * s` matches `coords`.
        components = vt[order, :].T
    except Exception:
        rng = np.random.default_rng(seed)
        coords = rng.standard_normal((matrix.shape[0], k))
        components = rng.standard_normal((matrix.shape[1], k))
    return (
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(components, dtype=np.float64),
    )


def _build_ingredient_weights(recipe: Recipe) -> dict[str, float]:
    """Return {canonical food name → grams} for the recipe's non-pantry items.

    Uses the `weight` field on each entry of `ingredients_raw` (which the HF
    dataset stores as grams per ingredient). Duplicate foods are summed.
    Pantry staples are dropped so the mass-weighted coverage isn't dominated
    by water/oil/salt. If the dataset has no mass info, an empty dict is
    returned and the caller falls back to unweighted coverage.
    """
    out: dict[str, float] = {}
    for item in recipe.ingredients_raw:
        if not isinstance(item, dict):
            continue
        food = str(item.get("food") or "").strip().lower()
        if not food or food in PANTRY_STAPLES:
            continue
        try:
            w = float(item.get("weight") or 0.0)
        except (TypeError, ValueError):
            w = 0.0
        if w <= 0:
            continue
        out[food] = out.get(food, 0.0) + w
    return out


def _minmax(x: np.ndarray) -> np.ndarray:
    """Stable min-max scaling to [0, 1]. Zero range → all zeros (not NaN)."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x.astype(np.float64) - lo) / (hi - lo)


def _encode_recipes_sbert(recipes: list[Recipe], batch_size: int = 64) -> np.ndarray | None:
    """Encode every recipe as a dense SBERT vector for hybrid retrieval.

    Representation: ONLY the comma-joined ingredient list — no recipe name.
    Critical for query–recipe symmetry: the user's query at runtime is just
    a list of ingredients, so encoding recipes with the SAME format ("a,b,c")
    puts them in the same region of SBERT space as the queries. Previously
    we prefixed with the recipe name ("Marzetti: ground beef, onion, ...")
    which systematically pushed recipes away from queries and made the user
    marker land far from its own recommendations on the 2-D map.

    L2-normalized so cosine similarity is a dot product.
    """
    model = get_sbert_model()
    if model is None:
        return None
    texts: list[str] = [
        ", ".join(r.ingredients[:25]) if r.ingredients else "empty"
        for r in recipes
    ]
    try:
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except Exception as e:
        print(f"[build] SBERT recipe encoding failed: {e}")
        return None
    return np.ascontiguousarray(vecs, dtype=np.float32)


def _label_clusters_semantically(
    recipe_sbert: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[int, str]:
    """For each cluster, pick the candidate topic whose embedding is closest
    to the cluster's mean recipe embedding. Cosine similarity on
    L2-normalized vectors → dot product.
    """
    model = get_sbert_model()
    if model is None:
        return {}
    candidate_embs = model.encode(
        CLUSTER_TOPIC_CANDIDATES,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    candidate_embs = np.asarray(candidate_embs, dtype=np.float32)

    topics: dict[int, str] = {}
    unique_cids = sorted(set(int(c) for c in cluster_labels.tolist()))
    for cid in unique_cids:
        mask = cluster_labels == cid
        if not mask.any():
            continue
        mean_emb = recipe_sbert[mask].mean(axis=0)
        n = np.linalg.norm(mean_emb)
        if n <= 1e-12:
            continue
        mean_emb = mean_emb / n
        sims = candidate_embs @ mean_emb
        best = int(np.argmax(sims))
        topics[cid] = CLUSTER_TOPIC_CANDIDATES[best]
    return topics


class _KNNProjector:
    """Out-of-sample projector for non-parametric embeddings (e.g. t-SNE).

    t-SNE has no `.transform()` method because it's optimized per-point — there's
    no closed-form mapping from arbitrary high-dim points into the 2-D layout.
    We approximate it with k-nearest-neighbors interpolation: find the k nearest
    recipes of the query in cluster-space (the space t-SNE was fit on), and
    place the query at the inverse-distance-weighted mean of their 2-D
    positions. This is the standard trick for putting new points on a t-SNE
    map without refitting.
    """

    def __init__(self, cluster_space: np.ndarray, projection_2d: np.ndarray,
                 k: int = 10, metric: str = "cosine"):
        from sklearn.neighbors import NearestNeighbors
        self._nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(cluster_space)
        self._projection = projection_2d
        self._k = k

    def transform(self, X: np.ndarray) -> np.ndarray:
        dists, idx = self._nn.kneighbors(X)
        weights = 1.0 / (dists + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        picks = self._projection[idx]                # (m, k, 2)
        return (picks * weights[..., None]).sum(axis=1)


def _build_viz_projection(
    cluster_space: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
):
    """2-D projection for visualization.

    Returns (projection_2d, transformer). The transformer has a `.transform`
    method so we can place query points on the same map.

    Methods, picked with RR_VIZ env var (default "pca"):
      - "pca"  : linear projection that preserves pairwise distances as
                 faithfully as 2 dimensions allow. This is the default
                 because it gives the strongest "retrieved recipes visually
                 close to the user marker" property — t-SNE's non-linear
                 warping breaks that even when the underlying similarity
                 is high.
      - "tsne" : non-linear, preserves local neighborhood. Clusters look
                 more visually separated, but global distances are lost.
                 Use if you want cluster-hunting over accuracy.
      - "lda"  : supervised linear, maximises between-cluster over within-
                 cluster variance. Clusters separate but queries can still
                 land far from their own recommendations.
    """
    method = method.lower()
    if method == "pca":
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            projection = pca.fit_transform(cluster_space)
            print(f"[build] PCA explained variance: "
                  f"{pca.explained_variance_ratio_.sum() * 100:.1f}%")
            return np.ascontiguousarray(projection, dtype=np.float64), pca
        except Exception as e:
            print(f"[build] PCA failed ({e}); falling back to t-SNE.")
            method = "tsne"

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            n = cluster_space.shape[0]
            perp = int(min(50, max(5, n // 250)))
            print(f"[build] Running t-SNE (n={n}, perplexity={perp})...")
            tsne = TSNE(
                n_components=2,
                init="pca",
                perplexity=perp,
                learning_rate="auto",
                random_state=42,
                method="barnes_hut",
                n_jobs=-1,
            )
            projection = tsne.fit_transform(cluster_space)
            projector = _KNNProjector(cluster_space, projection, k=10, metric="cosine")
            return np.ascontiguousarray(projection, dtype=np.float64), projector
        except Exception as e:
            print(f"[build] t-SNE failed ({e}); falling back to LDA.")
            method = "lda"

    if method == "lda":
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            lda = LinearDiscriminantAnalysis(n_components=2)
            projection = lda.fit_transform(cluster_space, labels)
            return np.ascontiguousarray(projection, dtype=np.float64), lda
        except Exception as e:
            print(f"[build] LDA viz projection failed ({e}); falling back to first-2-SVD.")

    return cluster_space[:, :2].copy(), None
