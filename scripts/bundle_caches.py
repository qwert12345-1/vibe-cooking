"""Pre-package all downloaded assets so the project tree is fully self-
contained before zipping up for a TA / grader.

Run this ONCE from the project root after you've already launched the app
at least once (i.e. after all the model downloads have completed into your
per-user HuggingFace / NLTK caches):

    python scripts/bundle_caches.py

It copies:
  - ~/.cache/huggingface/hub/*        →  ./models/hub/
  - ~/.cache/huggingface/<other>/*    →  ./models/<other>/
  - nltk_data corpora (wordnet)       →  ./nltk_data/corpora/

After running, `demo_web.py` will find every model locally via the HF_HOME
/ SENTENCE_TRANSFORMERS_HOME / NLTK_DATA env vars it sets at startup, so the
TA never has to download anything.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _copy_tree(src: Path, dst: Path, label: str) -> None:
    if not src.exists():
        print(f"[bundle] skip {label}: source not found at {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[bundle] {label}: destination exists, merging into {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        print(f"[bundle] {label}: {src}  →  {dst}")
        shutil.copytree(src, dst)


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    home = Path.home()

    hf_cache = home / ".cache" / "huggingface"
    if hf_cache.exists():
        _copy_tree(hf_cache, project_root / "models", "HuggingFace cache")
    else:
        print(
            f"[bundle] HuggingFace cache not found at {hf_cache}. "
            "Run `python -m app.demo_web` at least once so the models download "
            "into it, then retry."
        )

    # NLTK: wordnet may live in several possible locations depending on how
    # the user installed it. Search the canonical spots and copy each one
    # that we find.
    nltk_candidates = [
        home / "nltk_data",
        Path("C:/nltk_data"),
        Path("/usr/share/nltk_data"),
        Path("/usr/local/share/nltk_data"),
    ]
    dst_nltk = project_root / "nltk_data"
    copied_nltk = False
    for c in nltk_candidates:
        if c.exists():
            _copy_tree(c, dst_nltk, f"NLTK data ({c})")
            copied_nltk = True
    if not copied_nltk:
        print(
            "[bundle] NLTK data not found. If you rely on the lemmatizer, "
            "download wordnet first: python -m nltk.downloader -d ./nltk_data wordnet"
        )

    # Sanity check: the app-built artifacts that should also be in the zip.
    for mandatory in (
        project_root / "cache" / "engine.pkl",
        project_root / "seq2seq_checkpoints" / "best.pt",
    ):
        if mandatory.exists():
            print(f"[bundle] OK — {mandatory.relative_to(project_root)}")
        else:
            print(
                f"[bundle] WARNING — {mandatory.relative_to(project_root)} "
                "is missing. Run `python -m app.demo_web` once to build it, "
                "then re-run this bundler."
            )

    print("\n[bundle] Done. You can now zip the project tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
