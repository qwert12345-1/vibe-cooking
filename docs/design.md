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
│   ├── data_loader.py             # HF dataset download, JSON field parsing,
│   │                              #   ingredient-vocabulary construction
│   ├── normalizer.py              # Multi-stage input normalization pipeline:
│   │                              #   clean → lemmatize → exact → alias →
│   │                              #   fuzzy (Levenshtein) → semantic (SBERT)
│   ├── tfidf.py                   # [HAND-IMPLEMENTED] TF-IDF vectorizer
│   │                              #   + cosine similarity (numpy/scipy only)
│   ├── kmeans.py                  # [HAND-IMPLEMENTED] K-Means++ with elbow
│   │                              #   analysis for choosing k
│   ├── scoring.py                 # Jaccard, coverage, match-ratio, and
│   │                              #   missing-ingredient penalty metrics
│   ├── recommender.py             # Main engine: filter → recall → rank
│   └── constants.py               # Pantry staples, alias dictionary, cuisine
│                                  #   and flavor keyword lists
│
├── tests/
│   ├── test_tfidf.py              # Correctness vs scikit-learn reference
│   ├── test_kmeans.py             # Correctness vs scikit-learn reference
│   ├── test_normalizer.py         # Typo, alias, and translation test cases
│   └── test_recommender.py        # End-to-end smoke tests
│
├── app/                           # Gradio web frontend
│   ├── demo_web.py                # Application entry point
│   ├── components.py              # Input chips, filter panel, result cards
│   └── visualizer.py              # Plotly 2-D K-Means scatter with click-
│                                  #   to-explore interaction
│
├── notebooks/
│   └── exploration.ipynb          # Dataset exploration, elbow-method plot,
│                                  #   qualitative result inspection
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

## 2. Division of Labor (3-person team)

| Member | Owned Modules | Key Deliverables |
|---|---|---|
| **A — ML Core** | `src/tfidf.py`, `src/scoring.py`, `src/recommender.py`, `tests/test_tfidf.py` | Hand-implemented TF-IDF and cosine-similarity retrieval; fusion scoring formula combining cosine, coverage, and missing-ingredient penalty; candidate filtering and ranking pipeline |
| **B — ML Visualization** | `src/kmeans.py`, `app/visualizer.py`, `tests/test_kmeans.py`, `notebooks/exploration.ipynb` | Hand-implemented K-Means++ with elbow-method selection of k; Plotly interactive 2-D scatter plot with click-to-explore and cluster labeling; qualitative evaluation notebook |
| **C — Data and UI** | `src/data_loader.py`, `src/normalizer.py`, `app/demo_web.py`, `app/components.py`, `tests/test_normalizer.py` | HF dataset loader and vocabulary builder; multi-stage input-normalization pipeline with fuzzy and semantic matching; Gradio web UI with filter panel, confirmation chips for ambiguous input, and result cards |

Every member owns at least one concrete algorithmic module and one
user-visible deliverable. Interfaces between `src/` modules are documented
with type-annotated function signatures so members can integrate independently.

## 3. GitHub Stub

The project repository is live at `https://github.com/<team-org>/recipe_recommender` and contains, at checkpoint submission:

- `README.md` with setup instructions (`pip install -r requirements.txt`; `python -m app.demo_web`), project description, and placeholder for demo screenshots.
- `requirements.txt` pinning `datasets`, `numpy`, `scipy`, `pandas`, `nltk`, `rapidfuzz`, `sentence-transformers`, `gradio`, and `plotly`.
- `environment.yml` for a reproducible conda environment.
- Empty placeholder files with module docstrings and function stubs for every file listed in §1, each importable without errors (`python -c "import src.tfidf"` succeeds).
