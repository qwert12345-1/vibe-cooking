# Algorithms Used in Vibe Cooking

This document describes every algorithm and technique used in this project, written to be clear and approachable even if you are not deeply familiar with machine learning.

---

## Table of Contents

1. [TF-IDF Vectorization](#1-tf-idf-vectorization)
2. [Cosine Similarity](#2-cosine-similarity)
3. [K-Means++ Clustering](#3-k-means-clustering)
4. [Spherical K-Means (Cosine K-Means)](#4-spherical-k-means-cosine-k-means)
5. [Elbow Analysis for Choosing K](#5-elbow-analysis-for-choosing-k)
6. [Singular Value Decomposition (SVD)](#6-singular-value-decomposition-svd)
7. [Principal Component Analysis (PCA)](#7-principal-component-analysis-pca)
8. [t-SNE (t-Distributed Stochastic Neighbor Embedding)](#8-t-sne)
9. [Linear Discriminant Analysis (LDA)](#9-linear-discriminant-analysis-lda)
10. [KNN Out-of-Sample Projector](#10-knn-out-of-sample-projector)
11. [SBERT (Sentence-BERT) Embeddings](#11-sbert-sentence-bert-embeddings)
12. [Multi-Stage Input Normalization Pipeline](#12-multi-stage-input-normalization-pipeline)
13. [Fuzzy String Matching (Levenshtein / RapidFuzz)](#13-fuzzy-string-matching-levenshtein--rapidfuzz)
14. [WordNet Lemmatization](#14-wordnet-lemmatization)
15. [Hybrid Retrieval (TF-IDF + SBERT)](#15-hybrid-retrieval-tf-idf--sbert)
16. [Fused Scoring](#16-fused-scoring)
17. [Jaccard Similarity](#17-jaccard-similarity)
18. [F1 Score of Coverage and Match-Ratio](#18-f1-score-of-coverage-and-match-ratio)
19. [IDF-Weighted Match Ratio](#19-idf-weighted-match-ratio)
20. [Missing-Ingredient Exponential Penalty](#20-missing-ingredient-exponential-penalty)
21. [Name-Match Bonus](#21-name-match-bonus)
22. [Seq2Seq (GRU Encoder–Decoder)](#22-seq2seq-gru-encoderdecoder)
23. [Greedy Decoding](#23-greedy-decoding)
24. [Retrieval-Augmented Generation (RAG)](#24-retrieval-augmented-generation-rag)
25. [Causal Language Model (Qwen2.5 LLM)](#25-causal-language-model-qwen25-llm)

---

## 1. TF-IDF Vectorization

**File:** `src/tfidf.py`

**What it does:**
TF-IDF stands for *Term Frequency – Inverse Document Frequency*. It turns a list of ingredients into a mathematical vector (a list of numbers) that captures which ingredients are present and how distinctive they are across the whole recipe collection.

**How it works step by step:**

- **Term Frequency (TF):** For this project TF is *binary* — an ingredient is either in a recipe (1) or not (0). Having "onion" listed twice in a recipe doesn't make it more of an onion dish.
- **Inverse Document Frequency (IDF):** Rare ingredients (like "saffron") get a high IDF score; very common ones (like "salt") get a low score. The formula used is the *smoothed* form:
  ```
  IDF(ingredient) = log((total_recipes + 1) / (recipes_containing_ingredient + 1)) + 1
  ```
- **TF-IDF weight:** Multiply TF × IDF to produce the weight for each ingredient in each recipe.
- **L2 normalisation:** Each recipe's vector is divided by its own length so all vectors sit on the surface of a unit sphere. This makes cosine similarity the natural next step.

**Why it's used here:**
Ingredients like "salt" appear in almost every recipe and carry almost no useful signal. "Saffron" appears in very few recipes, so if a user mentions it, any recipe containing saffron is a very strong match. TF-IDF captures exactly that intuition.

**Key implementation detail:** Built from scratch without scikit-learn, using sparse SciPy matrices for memory efficiency on a large recipe corpus.

---

## 2. Cosine Similarity

**Files:** `src/tfidf.py`, `src/recommender.py`

**What it does:**
Measures how similar two vectors are by computing the angle between them, ignoring their magnitudes. Two perfectly identical ingredient lists → similarity = 1.0. Two completely different lists → similarity = 0.0.

**How it works:**
Because the TF-IDF vectors are already L2-normalised (length = 1), cosine similarity reduces to a simple dot product:
```
cosine_similarity(query, recipe) = query · recipe
```
For millions of recipe comparisons at once, this is expressed as a sparse matrix multiply: `recipes_matrix @ query_vector`, which is extremely fast.

**Why it's used here:**
It is the primary relevance signal — how closely does a recipe's ingredient profile match what the user typed?

---

## 3. K-Means++ Clustering

**File:** `src/kmeans.py`

**What it does:**
Groups all recipes into *k* clusters of similar dishes. Recipes in the same cluster tend to share main ingredients and cooking styles (e.g. one cluster might be "stir-fries", another "baked goods").

**How the algorithm works:**

1. **Initialisation (K-Means++):** Pick the first cluster centre randomly. Then pick each subsequent centre from the remaining data points with probability proportional to its squared distance from the nearest already-chosen centre. This "spreads out" the initial centres so the algorithm starts with a good guess, rather than clumping centres together by bad luck.

2. **Assignment:** Assign every recipe to its nearest centre using the formula:
   ```
   distance² = ||recipe||² + ||centre||² − 2 × (recipe · centre)
   ```
   The heavy lifting is a single matrix multiply, making it fast even for tens of thousands of recipes.

3. **Update:** Move each centre to the average (mean) of all the recipes assigned to it. If a cluster ends up empty, reseed it with a random recipe.

4. **Repeat:** Alternate steps 2 and 3 until the centres stop moving (convergence). "Stopped moving" is defined as total shift < tolerance threshold.

5. **Multiple restarts:** The whole process is run `n_init` times with different random seeds; the run with the lowest total inertia (sum of squared distances to centres) wins.

**Key metric — Inertia:**
```
inertia = Σ (distance from each recipe to its nearest centre)²
```
Lower inertia = tighter, more coherent clusters.

---

## 4. Spherical K-Means (Cosine K-Means)

**File:** `src/kmeans.py` (the `metric="cosine"` path)

**What it does:**
A variant of K-Means designed to work with directional data — cases where the *direction* of a vector matters but not its length.

**How it differs from regular K-Means:**
- Before clustering, all recipe vectors are L2-normalised onto the unit sphere.
- After each update step, the new cluster centres are also L2-normalised (projected back onto the sphere).
- On the unit sphere, Euclidean distance and cosine dissimilarity are equivalent:
  ```
  ||x − y||² = 2 − 2·cos(x, y)
  ```
  So minimising Euclidean distance on normalised vectors is the same as maximising cosine similarity.

**Why it's used here:**
TF-IDF and SBERT vectors are directional. Two recipes with the same ingredient proportions but different total ingredient counts should be considered similar — spherical K-Means handles this correctly.

---

## 5. Elbow Analysis for Choosing K

**File:** `src/kmeans.py`

**What it does:**
Automatically decides the best number of clusters (*k*) to use, rather than guessing.

**How it works:**
1. Run K-Means for each candidate *k* (e.g. 8, 10, 12, … 50) on a random subsample of recipes for speed.
2. Plot *k* on the x-axis and inertia on the y-axis — this usually forms a curve that bends ("elbows") somewhere.
3. Draw a straight line from the first point (smallest *k*, highest inertia) to the last point (largest *k*, lowest inertia).
4. Find the point on the curve that is farthest from this line — that's the elbow, the *k* where adding more clusters gives diminishing returns.

**Implementation note:** Both axes are normalised before computing distances to avoid the scale of inertia values dominating.

---

## 6. Singular Value Decomposition (SVD)

**File:** `src/recommender.py` (`_build_cluster_embeddings`)

**What it does:**
Compresses the high-dimensional TF-IDF matrix (one column per unique ingredient, potentially thousands) down to 50 dimensions, keeping the most important variance.

**How it works:**
SVD factorises a matrix *M* into three matrices:
```
M ≈ U × S × Vᵀ
```
- *U* contains recipe coordinates in the compressed space.
- *S* contains singular values (importance of each dimension).
- *Vᵀ* contains the ingredient directions.

The top 50 components (those with the largest singular values) are kept. New query vectors can be projected into the same 50-D space using the stored *V* basis.

**Design choices in this project:**
- Pantry staples (salt, oil, water, etc.) are zeroed out before SVD. They co-occur in nearly every recipe and would dominate the leading dimensions, washing out the real signal.
- 50 dimensions is used for clustering, not 2. Clustering in 2-D discards almost all the useful information; the 2-D view is only for visualisation.

---

## 7. Principal Component Analysis (PCA)

**File:** `src/recommender.py` (`_build_viz_projection`)

**What it does:**
Projects the 50-D clustering space down to 2 dimensions for display on the scatter map, while preserving as much of the original structure as possible.

**How it works:**
PCA finds the two directions in the data that capture the most variance and projects every recipe onto those two axes. It is a linear transformation — straight lines in high-D remain straight lines in 2-D.

**Why it's preferred for this app:**
PCA is the default visualisation method because it preserves *global* distances. When a user searches for "chicken, saffron", their marker on the 2-D map should land physically close to the top-ranked recipes. PCA's linear nature maintains this property; t-SNE's non-linear warping does not.

---

## 8. t-SNE

**File:** `src/recommender.py` (`_build_viz_projection`, selectable via `RR_VIZ=tsne`)

**What it does:**
A non-linear dimensionality reduction technique that is especially good at making clusters visually "pop" on a 2-D map by preserving local neighbourhood structure.

**How it works:**
t-SNE converts high-dimensional distances to probabilities and places points in 2-D so that nearby points in high-D are also nearby in 2-D. It iteratively minimises the difference between the high-D and 2-D probability distributions using gradient descent.

**Trade-off vs PCA:**
Clusters look more visually separated, but global distances are distorted. A user's query marker may land far from their own recommendations even when the underlying similarity is high.

**Out-of-sample problem:**
t-SNE has no way to project new points (queries). This is handled by the KNN Projector described below.

---

## 9. Linear Discriminant Analysis (LDA)

**File:** `src/recommender.py` (`_build_viz_projection`, selectable via `RR_VIZ=lda`)

**What it does:**
A *supervised* dimensionality reduction technique that uses the cluster labels (from K-Means) to find the 2 directions that best separate the clusters from each other.

**How it works:**
LDA maximises the ratio of between-cluster variance to within-cluster variance. It finds the linear combinations of features that push different clusters as far apart as possible while keeping points within the same cluster close together.

**Trade-off:**
Clusters visually separate well, but query points can still land far from their own top-ranked recipes.

---

## 10. KNN Out-of-Sample Projector

**File:** `src/recommender.py` (`_KNNProjector`)

**What it does:**
Approximates where a new query point should land on a t-SNE map, which normally has no way to project unseen points.

**How it works:**
1. Find the *k* nearest recipes to the query in the original high-dimensional clustering space (using cosine distance).
2. Place the query at the *inverse-distance-weighted average* of those recipes' 2-D positions:
   - Closer neighbours get more weight.
   - Weight = 1 / (distance + ε) to avoid division by zero.

This is the standard trick for placing new points on a t-SNE map without refitting the entire embedding.

---

## 11. SBERT (Sentence-BERT) Embeddings

**Files:** `src/embeddings.py`, `src/recommender.py`, `src/normalizer.py`

**What it does:**
Converts any text (an ingredient name, a list of ingredients, a recipe title) into a 384-dimensional dense vector that captures *meaning*. Words or phrases with similar meaning end up near each other in this space.

**How it works:**
SBERT is a fine-tuned version of BERT (a large neural network pre-trained on hundreds of billions of words). It reads a sentence and outputs a single fixed-size vector representing its semantics. The model used here (`all-MiniLM-L6-v2`) produces 384-dimensional vectors.

**Three uses in this project:**

1. **Clustering:** Encoding every recipe as a dense vector so K-Means groups by *dish type* rather than just ingredient overlap. Two recipes using "egg, butter, milk" could be a cheesecake and a carbonara — TF-IDF sees them as similar, SBERT correctly separates them.

2. **Hybrid retrieval:** At query time, the user's ingredient list is encoded and compared against all recipe SBERT vectors. This catches semantic matches that TF-IDF misses (e.g. "poultry" ≈ "chicken").

3. **Semantic normalisation fallback:** If a user types an ingredient the vocabulary doesn't recognise at all (after trying exact match, lemmatisation, aliases, and fuzzy matching), SBERT is used as a last resort to find the closest known ingredient by meaning.

**Practical details:**
- All vectors are L2-normalised so cosine similarity = dot product.
- The model is loaded once and shared across all three uses to avoid duplicating ~80 MB in RAM.

---

## 12. Multi-Stage Input Normalization Pipeline

**File:** `src/normalizer.py`

**What it does:**
Converts whatever a user types ("Chickens", "洋葱", "tomatos", "poulet") into the canonical ingredient names used by the recipe corpus. Works for English, Chinese, and misspelled inputs.

**The 7 stages (tried in order; first match wins):**

| Stage | Technique | Example |
|---|---|---|
| 1. Clean | Lowercase, strip punctuation, collapse whitespace | `"  CHICKEN!! "` → `"chicken"` |
| 2. Exact match | Direct lookup in the vocabulary set | `"chicken"` → ✓ found |
| 3. Lemmatise + exact | Reduce word to base form, then look up | `"chickens"` → `"chicken"` → ✓ found |
| 4. Alias dictionary | Hand-curated synonyms and translations | `"poulet"` → `"chicken"` |
| 5. Superordinate expansion | Abstract category → set of members | `"poultry"` → `{chicken, turkey, duck, ...}` |
| 6. Fuzzy matching | Levenshtein edit-distance typo correction | `"chikken"` → `"chicken"` (score 92/100) |
| 7. Semantic SBERT | Neural meaning-based nearest-neighbour | `"rooster meat"` → `"chicken"` (cosine 0.82) |

Each result carries a **confidence score** and the **stage** that resolved it. Low-confidence fuzzy and semantic matches are flagged so the UI can ask the user to confirm.

---

## 13. Fuzzy String Matching (Levenshtein / RapidFuzz)

**File:** `src/normalizer.py`

**What it does:**
Finds the closest word in the vocabulary to a mistyped input by counting the minimum number of single-character edits (insertions, deletions, substitutions) needed to transform one string into another.

**How it works:**
Uses the `RapidFuzz` library's `WRatio` scorer, which is a weighted combination of several Levenshtein-based metrics:
- Simple ratio: direct character-by-character comparison.
- Partial ratio: best alignment of the shorter string within the longer one.
- Token sort / token set ratios: handles word order differences.

The top-5 candidate matches are returned. A match is accepted only if the best score is ≥ 80 out of 100.

**Example:** `"samon"` → `"salmon"` (score: 86)

---

## 14. WordNet Lemmatization

**File:** `src/normalizer.py`

**What it does:**
Reduces inflected word forms to their dictionary base form (the *lemma*). Handles plurals, verb conjugations, and other morphological variations.

**How it works:**
Uses NLTK's `WordNetLemmatizer`, which looks up words in the WordNet lexical database and returns the canonical noun form:
- `"tomatoes"` → `"tomato"`
- `"chickens"` → `"chicken"`
- `"onions"` → `"onion"`

**Fallback:** If NLTK/WordNet is not available, a naive rule strips a trailing "s" from words longer than 3 characters. This handles ~80% of English plurals without any library.

---

## 15. Hybrid Retrieval (TF-IDF + SBERT)

**File:** `src/recommender.py` (`recommend` method)

**What it does:**
Combines two complementary similarity signals into one score for retrieving the candidate recipe pool:
- **TF-IDF cosine** captures exact ingredient overlap, boosted by rare ingredients (IDF).
- **SBERT cosine** captures semantic meaning — "poultry" matches "chicken", "creamy pasta" matches "alfredo".

**How it works:**
Both scores are independently min-max scaled to [0, 1] so they live on comparable scales. Then they are linearly combined:
```
hybrid_score = 0.55 × TF-IDF_score + 0.45 × SBERT_score
```
The top `candidate_pool` (default 500) recipes by this hybrid score are passed to the slower re-ranking stage.

**Why both are needed:**
TF-IDF misses semantic relatives ("poultry" and "chicken"). SBERT is slower and catches broader associations that don't always mean a recipe is a practical match. Together they produce a better shortlist than either alone.

---

## 16. Fused Scoring

**File:** `src/scoring.py` (`fused_score`)

**What it does:**
The final ranking score that orders the candidate recipes from best to worst match. It combines three signals and applies a penalty for missing ingredients.

**The formula:**
```
score = w_cos × cosine  +  w_f1 × F1(match_ratio, coverage)  +  w_name × name_bonus
score ×= exp(−α × |non-pantry missing ingredients|)
```
Default weights: `w_cos = 0.50`, `w_f1 = 0.30`, `w_name = 0.20`, `α = 0.03`.

**What each signal captures:**

| Signal | What it measures | Why it's needed |
|---|---|---|
| TF-IDF cosine | Continuous ingredient similarity (IDF-weighted) | Rare ingredients like "saffron" count more than "salt" |
| F1(match_ratio, coverage) | Symmetric set overlap quality | Prevents trivially simple recipes from ranking too high |
| Name-match bonus | Whether user ingredients appear in the recipe's title | "Beer Can Chicken" should rank high when user has chicken |
| Missing-ingredient penalty | How many non-pantry ingredients the user lacks | Deprioritises recipes requiring many ingredients the user doesn't have |

---

## 17. Jaccard Similarity

**File:** `src/scoring.py` (`jaccard`)

**What it does:**
Measures the overlap between two ingredient sets as a fraction of their union.

**Formula:**
```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```
Returns 0 when there is no overlap, 1 when the sets are identical.

**Example:** User has `{chicken, tomato, garlic}`, recipe needs `{chicken, tomato, basil, garlic, olive oil}`. Intersection = 3, Union = 5 → Jaccard = 0.60.

**Note:** Jaccard is computed and returned for display in the UI cards, but the primary ranking uses the fused score above.

---

## 18. F1 Score of Coverage and Match-Ratio

**File:** `src/scoring.py`

Two complementary ingredient overlap metrics, combined into an F1 score:

**Coverage** (recall-like): What fraction of the recipe's non-pantry ingredients does the user already have?
```
coverage = |user ∩ recipe_core| / |recipe_core|
```

**Match-Ratio** (precision-like): What fraction of the user's typed ingredients does this recipe actually use?
```
match_ratio = |user ∩ recipe| / |user|
```

**F1 combination:**
```
F1 = 2 × match_ratio × coverage / (match_ratio + coverage)
```
F1 penalises extremes: a recipe that covers 100% of the user's inputs but only needs 1 ingredient scores very low on match_ratio, while a 100-ingredient recipe that uses only 1 of the user's inputs scores very low on coverage.

---

## 19. IDF-Weighted Match Ratio

**File:** `src/scoring.py` (`weighted_match_ratio`)

**What it does:**
A smarter version of Match-Ratio that weights each user ingredient by its IDF score — how rare and distinctive it is.

**Formula:**
```
weighted_match_ratio = Σ IDF(i) for i in (user ∩ recipe)  /  Σ IDF(i) for i in user
```

**Example:** User types "chicken, saffron, salt". A recipe with chicken + saffron but no salt captures nearly 100% of the *information* in the query (salt carries almost zero IDF weight). A recipe with chicken + salt but no saffron captures only the chicken share — saffron is the expensive, distinctive ask.

---

## 20. Missing-Ingredient Exponential Penalty

**File:** `src/scoring.py` (`missing_penalty`)

**What it does:**
Softly penalises recipes that require many non-pantry ingredients the user doesn't have, without completely eliminating them from results.

**Formula:**
```
penalty = exp(−α × |missing non-pantry ingredients|)
```
With `α = 0.03`:
- 0 missing → penalty = 1.00 (no penalty)
- 5 missing → penalty ≈ 0.86
- 10 missing → penalty ≈ 0.74
- 20 missing → penalty ≈ 0.55

Pantry staples (salt, oil, water, etc.) are excluded from the count — almost every recipe needs them and the user is expected to have them.

---

## 21. Name-Match Bonus

**File:** `src/scoring.py` (`name_match_bonus`)

**What it does:**
Gives a score boost to recipes whose *title* explicitly mentions one of the user's ingredients. This captures the "main ingredient" signal — if a user has chicken, "Beer Can Chicken" should rank higher than a recipe that merely contains chicken as a minor ingredient.

**How it works:**
1. Find which of the recipe's non-pantry ingredients appear in the recipe's title.
2. Compute the fraction of those "title-mentioned" ingredients that the user has.
3. Scale the bonus by `min(1, recipe_non_pantry_size / 3)` — single-ingredient recipes get only ⅓ of the bonus to avoid them gaming the score unfairly.

**Motivation (from IR):** This is the classical "title-match boost" from Information Retrieval: a query term appearing in a document's title is worth much more than one buried in the body.

---

## 22. Seq2Seq (GRU Encoder–Decoder)

**Files:** `src/seq2seq/model.py`, `src/seq2seq/predict.py`

**What it does:**
A neural network trained to read a list of ingredients and predict a recipe title. For example: "chicken, tomato, basil, pasta" → "Chicken Tomato Basil Pasta".

**How it works:**

**Encoder (GRU):**
- Takes the ingredient list as a sequence of tokens.
- Uses a *Gated Recurrent Unit* (GRU) — a type of recurrent neural network cell that has a memory mechanism. It processes tokens one by one and compresses the entire sequence into a fixed-size "hidden state" vector that summarises what it read.
- Padding is handled with `pack_padded_sequence` so the network ignores blank tokens.

**Decoder (GRU):**
- Takes the encoder's hidden state as its starting memory.
- Generates the output title one word at a time: at each step it takes the last generated word, runs it through a GRU cell, and outputs a probability distribution over the vocabulary. The most likely next word is chosen (greedy decoding — see below).

**Training:**
- Teacher forcing: during training, the true previous word (not the model's prediction) is fed as input at each decoding step, which stabilises learning.

---

## 23. Greedy Decoding

**File:** `src/seq2seq/predict.py` (`greedy`)

**What it does:**
The inference strategy used with the Seq2Seq model to generate a title token by token.

**How it works:**
1. Encode the ingredient list → hidden state.
2. Feed a Start-of-Sequence (`<sos>`) token to the decoder.
3. At each step, the decoder outputs logits (raw scores) for every word in the vocabulary.
4. Pick the word with the highest score (`argmax`) — this is the "greedy" choice.
5. Feed that word as input for the next step.
6. Stop when the decoder outputs an End-of-Sequence (`<eos>`) token or a maximum length is reached.

**Trade-off:** Greedy decoding is fast but may miss the globally optimal sequence. More sophisticated alternatives like Beam Search keep multiple candidate sequences alive, but greedy is sufficient for the title-generation use case here.

---

## 24. Retrieval-Augmented Generation (RAG)

**Files:** `src/seq2seq_backend.py`, `src/seq2seq/rag.py`, `src/seq2seq/allrecipes.py`

**What it does:**
Grounds the generated recipe in real examples from the corpus to prevent hallucination (inventing implausible steps or unexpected ingredients).

**The full pipeline:**

```
User ingredients
       ↓
[Seq2Seq] → Predicted title ("Chicken Tomato Pasta")
       ↓
[Retrieval] → Find top-5 real AllRecipes recipes that match the ingredients + title
       ↓
[RAG composer] → Build a new recipe draft using real steps as a template
```

**Retrieval scoring (token overlap):**
```
score = |query_tokens ∩ recipe_tokens| / |query_tokens|  +  0.10 (if any query token appears in the recipe title)
```
Simple but effective for an in-memory corpus of 50k recipes.

**Draft composition heuristics:**
1. Build an ingredient list by ranking lines from retrieved recipes by frequency and retrieval order.
2. Infer a cooking method from the predicted title (bake, grill, simmer, etc.).
3. Select and filter direction steps from the top-2 retrieved recipes, keeping only steps that:
   - Use ingredients the user actually has.
   - Don't mention extra food words not in the user's list.
   - Don't conflict with the inferred cooking method.
4. Fill any structural gaps (no prep step, no cook step, no finish step) with parameterised default templates.

---

## 25. Causal Language Model (Qwen2.5 LLM)

**File:** `src/creative.py`

**What it does:**
Uses a small but capable instruction-tuned large language model to generate a fully original, creative recipe from the user's ingredients. Unlike the Seq2Seq + RAG approach, this model generates free-form natural language without needing to retrieve templates.

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB, runs on CPU). Can be swapped for larger variants (1.5B, 3B) via the `RR_LLM_REPO` environment variable.

**How it works:**
1. Detect language: if the user typed Chinese characters, use the Chinese system prompt; otherwise use English.
2. Build a conversation with a strict system prompt that specifies the exact output format (markdown headers: `# Dish Name`, `## Ingredients`, `## Steps`, `## Tip`) and rules (only use user-supplied ingredients plus common pantry items).
3. Format the conversation using the model's chat template (`apply_chat_template`).
4. Run autoregressive text generation with `do_sample=True`, `temperature=0.75`, `top_p=0.9` — these parameters introduce controlled randomness so different runs produce different recipes.
5. Strip the prompt tokens from the output and return only the newly generated text.

**Sampling parameters explained:**
- `temperature = 0.75`: Slightly sharpens the probability distribution over the next token, making the output less random than pure sampling but more varied than greedy. Lower = more predictable, higher = more creative.
- `top_p = 0.9`: "Nucleus sampling" — at each step, consider only the smallest set of tokens whose cumulative probability exceeds 0.9. Prevents the model from choosing very unlikely tokens while still allowing diversity.
