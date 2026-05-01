# 🧑‍🍳 Vibe Cooking!

## Project Overview

Ever stare into your fridge full of random ingredients and wonder what on earth to make? **Vibe Cooking** is here to rescue your dinner (or ruin it)! It's a fully locally-run recipe recommendation and generation engine that turns your pantry leftovers into culinary masterpieces or lets you virtually "bomb the kitchen" by inventing unhinged "dark cuisine" from chaotic ingredient combinations. By seamlessly blending our custom-built algorithms (TF-IDF, K-Means clustering, and an in-house trained Seq2Seq model) with powerful open-source generative AI (LLMs and Stable Diffusion), Vibe Cooking bridges the gap between finding real, trustworthy recipes and conjuring entirely new dishes on the fly.

This interactive web app takes a few ingredients and gives you:

1. **A brand-new creative recipe** — either from a local LLM (Qwen2.5-0.5B via
   Hugging Face Transformers) or from a hand-trained Seq2Seq + retrieval-augmented draft
   composer. Each generator also produces a preview image via a local Stable
   Diffusion (SDXS-512) pipeline.
2. **Closest real recipes from the dataset** as a safety net — ranked by a
   fused IDF-weighted match score with a recipe-name-mention bonus.
3. **An interactive 2-D map of the ~39 k-recipe universe** — hand-implemented
   Spherical K-Means on SBERT embeddings, visualised with PCA. Click any
   point to open that recipe.

---

## Quickstart

The application runs entirely locally. It will automatically download the required Hugging Face models and dataset on the first launch.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/vibe-cooking.git
cd vibe-cooking

# 2. Create the conda env (recommended — resolves torch
#    binary dependencies cleanly, especially on Windows)
conda env create -f environment.yml
conda activate vibe_cooking

# 3. Launch
python -m app.demo_web
```

Then open the URL Gradio prints (typically <http://localhost:7860>).

**First launch takes ~30–60 s** while the LLM, image pipeline and K-Means
engine warm up from disk. Subsequent launches are ~10 s.

---

## Project layout

```
vibe-cooking/
├── app/                        # Gradio web interface
│   ├── __init__.py
│   ├── demo_web.py            # Gradio entry point (runs everything)
│   ├── components.py          # HTML card / chip renderers
│   └── visualizer.py          # Plotly cluster scatter + circle overlay
├── src/                        # Core engine modules
│   ├── __init__.py
│   ├── data_loader.py         # HF dataset loader + Recipe dataclass
│   ├── normalizer.py          # 7-stage query cleanup: clean → lemma →
│   │                          #   alias → superord. → fuzzy → SBERT
│   ├── tfidf.py               # HAND-IMPLEMENTED TF-IDF + cosine
│   ├── kmeans.py              # HAND-IMPLEMENTED spherical K-Means++
│   ├── scoring.py             # IDF-weighted F1 + name-mention bonus
│   ├── recommender.py         # filter → hybrid recall → rerank engine
│   ├── embeddings.py          # Shared SBERT model loader
│   ├── creative.py            # LLM backend (Qwen2.5-0.5B inference)
│   ├── visualize_dish.py      # T2I backend (diffusers + SDXS-512)
│   ├── seq2seq_backend.py     # Seq2Seq backend integration layer
│   ├── seq2seq/               # Seq2Seq model + inference
│   │   ├── __init__.py
│   │   ├── model.py           # Encoder / Decoder / Seq2Seq architecture
│   │   ├── vocab.py           # Vocabulary management
│   │   ├── tokens.py          # Token utilities
│   │   ├── predict.py         # Greedy title decoding
│   │   ├── rag.py             # Retrieval-augmented recipe draft composition
│   │   └── allrecipes.py      # Hugging Face dataset integration
│   └── constants.py           # Pantry staples, aliases, cuisine/diet lists
├── tests/                      # pytest test suite (24 passing, 1 skipped optional)
│   ├── __init__.py
│   ├── test_normalizer.py
│   ├── test_kmeans.py
│   ├── test_tfidf.py
│   ├── test_recommender.py
│   └── test_seq2seq_allrecipes.py
├── docs/                       # Documentation (proposal.md + design.md)
├── notebooks/                  # Jupyter notebooks for exploration
├── scripts/
│   └── bundle_caches.py       # Run once before zipping for distribution
├── seq2seq_checkpoints/
│   └── best.pt                # Final Seq2Seq inference checkpoint
├── cache/                      # Built engine artifacts (created at runtime, .gitignored)
│   ├── engine.pkl             # Prebuilt TF-IDF + K-Means + t-SNE artifacts
│   ├── recipes.pkl            # Parsed Recipe objects
│   └── sbert_partial.pkl      # SBERT embeddings of the 39k recipes
├── models/                     # HuggingFace model cache (downloaded at runtime, .gitignored)
├── nltk_data/                  # Wordnet corpus (downloaded at runtime, .gitignored)
├── data/                       # HF recipes dataset cache (downloaded at runtime, .gitignored)
├── .gitattributes
├── .gitignore
├── requirements.txt
├── environment.yml
├── wordnet.zip                 # Pre-extracted wordnet data for quick startup
└── README.md
```

---

## Installation alternatives

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate vibe_cooking
```

### Option B — pip + venv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Torch CPU wheel (Windows/Linux) — safer than the default PyPI wheel:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Everything else
pip install -r requirements.txt

# Note: You no longer need external tools like `llama-cpp-python`.
# The pipeline natively uses PyTorch and Hugging Face Transformers for robust
# model inferencing entirely from Python!
```

---

## Running the app

```bash
python -m app.demo_web
```

Environment variables (all optional):

| variable | default | effect |
| --- | --- | --- |
| `RR_PORT` | `7860` | Gradio server port |
| `RR_SERVER_NAME` | `127.0.0.1` | Bind address. Use `0.0.0.0` only if you intentionally want LAN access. |
| `RR_NO_LLM_PRELOAD` | `0` | `1` = skip LLM preload at startup (lazy-load on first click) |
| `RR_NO_Seq2Seq_PRELOAD` | `0` | `1` = same, for Seq2Seq |
| `RR_NO_IMAGE_PRELOAD` | `0` | `1` = same, for SDXS image pipeline |
| `RR_NO_IMAGE` | `0` | `1` = disable the T2I backend entirely |
| `RR_NO_Seq2Seq` | `0` | `1` = hide the Seq2Seq generator option |
| `RR_NO_SBERT` | `0` | `1` = skip the normalizer's SBERT fallback (faster startup, slightly worse typo recovery) |
| `RR_NO_SBERT_RECIPES` | `0` | `1` = skip the whole-corpus SBERT encoding step (degrades clustering quality) |
| `RR_CLUSTER_EMBED` | `sbert` | `svd` = cluster on SVD-of-TF-IDF instead of SBERT |
| `RR_VIZ` | `pca` | `tsne` / `lda` = alternative 2-D projections |
| `RR_T2I_REPO` | `IDKiro/sdxs-512-dreamshaper` | e.g. `stable-diffusion-v1-5/stable-diffusion-v1-5` for better quality |

---

## How The Seq2Seq Path Works

The Seq2Seq generator is a classic encoder-decoder seq2seq model that predicts a
recipe title from an ingredient prompt such as `i have chicken, rice, garlic`.
At inference time, the ingredients are normalized into the same prompt format
seen during training, encoded once, and then greedily decoded token-by-token
until the title ends. In the UI, this title is genuinely model-generated,
though we apply a small post-processing step to keep it anchored to the main
ingredient when needed.

Training provenance: this model was trained in a separate project workspace on
the Hugging Face [`corbt/all-recipes`](https://huggingface.co/datasets/corbt/all-recipes)
corpus(~2M recipes), and the final `best.pt` checkpoint was later copied into this repo for
inference. This repository contains the bundled checkpoint plus the inference
and retrieval-grounding path, but not the original end-to-end training
workspace. 

The cooking steps are not generated by the Seq2Seq model itself. Instead, the
Seq2Seq path now grounds against the Hugging Face [`corbt/all-recipes`](https://huggingface.co/datasets/corbt/all-recipes)
corpus: it retrieves similar real recipes, parses their ingredient lines and
directions, filters borrowed steps so they stay compatible with the user's
ingredients and the inferred cooking method, and fills any gaps with structured
fallback steps. So the Seq2Seq title comes from seq2seq decoding, while the final
ingredients and directions come from retrieval-grounded composition.


---

## Local Caches and Checkpoints

The repository includes the essential custom models:
- `seq2seq_checkpoints/best.pt` for the Seq2Seq title model at inference time.

The following directories will be automatically created on your first launch to cache downloaded data and models, and they are ignored by git:
- `cache/` for prebuilt recommender artifacts and parsed recipe caches
- `models/` for local Hugging Face model caches used by SBERT, the LLM, and image generation
- `nltk_data/` and `data/` for local WordNet and dataset caches

These project-local paths are loaded at runtime via the `HF_HOME`, `HF_HUB_CACHE`, `SENTENCE_TRANSFORMERS_HOME`, and `NLTK_DATA` environment variables set by `app/demo_web.py`.

---

## Troubleshooting

**`AttributeError: 'Plot' object has no attribute 'select'`**
You're on Gradio ≥ 6.0, which removed that event. Pin to 5.x:
`pip install "gradio>=5.0,<6.0"`.

**`set_module_tensor_to_device() got an unexpected keyword argument 'non_blocking'`**
`accelerate` is too old. `pip install --upgrade accelerate` (>= 0.33).

**`OMP: Error #15: Initializing libomp.dll, but found libomp.dll already initialized`**
Two copies of OpenMP loaded. Already handled in `demo_web.py` via
`KMP_DUPLICATE_LIB_OK=TRUE`; if you disabled that, re-enable it.



**Model download is slow or fails**
Since the application automatically downloads Hugging Face models on the first launch, it might take a few minutes depending on your network speed. Please be patient during the first launch.

---

## Credits

- Dataset: [`datahiveai/recipes-with-nutrition`](https://huggingface.co/datasets/datahiveai/recipes-with-nutrition) (~39 k recipes)
- LLM: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- Image model: [`IDKiro/sdxs-512-dreamshaper`](https://huggingface.co/IDKiro/sdxs-512-dreamshaper)
- SBERT: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Seq2Seq: model trained separately on [`corbt/all-recipes`](https://huggingface.co/datasets/corbt/all-recipes); final inference checkpoint bundled at `seq2seq_checkpoints/best.pt`
