# Design Document: Ingredient-Based Recipe Recommender

## 1. Repository Structure

```
recipe_recommender/
├── README.md                      # Setup, run instructions, demo screenshots
├── requirements.txt               # pip dependencies
├── environment.yml                # conda environment (reproducible alt)
├── .gitignore                     # ignores cache/, data/, __pycache__
│
├── data/                          # Raw dataset (downloaded at first run, gitignored)
├── cache/                         # Pickled offline artifacts (gitignored)
│
├── src/                           # Pure algorithmic logic, no UI dependencies
│   ├── __init__.py
│   ├── constants.py               # Pantry staples, alias dictionary, cuisine/flavor keywords
│   ├── creative.py                # Local Qwen2.5-0.5B generative LLM pipeline
│   ├── data_loader.py             # HF dataset download, field parsing, vocab builder
│   ├── embeddings.py              # Shared SBERT loader for semantic match/fallback
│   ├── kmeans.py                  # [HAND-IMPLEMENTED] K-Means++ with elbow analysis
│   ├── normalizer.py              # Multi-stage input normalization pipeline
│   ├── recommender.py             # Main ranking engine: filter → recall → rank
│   ├── scoring.py                 # Fused retrieval scoring metrics
│   ├── seq2seq_backend.py         # Application interface for Seq2Seq models
│   ├── tfidf.py                   # [HAND-IMPLEMENTED] TF-IDF + cosine similarity
│   ├── visualize_dish.py          # Stable Diffusion (SDXS) recipe image generation
│   └── seq2seq/                   # Custom PyTorch Transformer Architecture
│       ├── model.py               # Encoder/Decoder Transformer architecture
│       ├── predict.py             # Inference pipeline
│       ├── rag.py                 # RAG augmentation pipeline
│       └── allrecipes.py          # Data preprocessing
│
├── app/                           # Gradio web frontend
│   └── demo_web.py                # Monolithic Application entry point + Plotly visualizations
│
└── docs/
    ├── proposal.md                # Two-paragraph written proposal
    └── design.md                  # This design document
```

**Separation of concerns.** `src/` contains pure algorithmic logic with no UI
dependencies and can be unit-tested or reused from a notebook. `app/` contains
all Gradio/Plotly code and imports from `src/`. The two hand-implemented
algorithms live in dedicated files (`tfidf.py`, `kmeans.py`) so their
correctness and design choices can be reviewed and defended independently.

## 2. Division of Labor (5-person team)

| Member | Owned Modules | Key Deliverables |
|---|---|---|
| **A — Core ML & Retrieval** | `src/tfidf.py`, `src/recommender.py`, `src/scoring.py` | Hand-implemented TF-IDF; recommendation engine architecture; fusion scoring metric (cosine + missing-ingredient penalty). |
| **B — Clustering & Semantic Space** | `src/kmeans.py`, `src/embeddings.py` | Hand-implemented Spherical K-Means++ with elbow method; unifying SBERT embeddings for the feature space. |
| **C — Generative AI Integrations** | `src/creative.py`, `src/visualize_dish.py` | Integration with Local Qwen2.5 LLMs and System Prompting; visual generation via Stable Diffusion (SDXS) models. |
| **D — Seq2Seq Transformer Architecture** | `src/seq2seq/` | Custom PyTorch Transformer architecture; training pipelines; data preprocessing (`allrecipes.py`); decoding strategies for AI generation. |
| **E — Data Pipeline & Visual Dashboard** | `src/data_loader.py`, `src/normalizer.py`, `app/demo_web.py` | Multi-stage typo/alias normalization pipeline; Plotly interactive 2-D clustering graph; interactive Gradio Dashboard interface. |

## 3. Repository Contents

The finalized project repository is live at `https://github.com/qwert12345-1/vibe-cooking` and contains:

- `README.md` with instructions for environment setup, offline dataset caching, and UI launching.
- `requirements.txt` containing all specialized ML dependencies (`torch`, `datasets`, `transformers`, `diffusers`, `rapidfuzz`, `sentence-transformers`, `scipy`, `numpy`, `gradio`, `plotly`).
- `environment.yml` for a strictly reproducible conda environment setup.
- Fully implemented, completely offline-capable AI/ML modules in `src/` (SBERT, Spherical K-Means, PyTorch Transformers, GenAI integration). Each pipeline component evaluates input data natively without requiring external API keys.
