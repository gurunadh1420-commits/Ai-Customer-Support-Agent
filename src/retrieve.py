"""
Historical support-pair retrieval via TF-IDF + cosine similarity.

Index is built from processed support pairs (customer_text), excluding Golden Set
customer tweet IDs. Does not use weak_intent labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    GOLDEN_SET_CANDIDATES_PATH,
    PROCESSED_SUPPORT_PAIRS_PATH,
    RETRIEVAL_INDEX_PATH,
)
from src.weak_labels import load_golden_customer_tweet_ids

REQUIRED_PAIR_COLUMNS = [
    "customer_tweet_id",
    "agent_tweet_id",
    "customer_text",
    "agent_text",
]


def load_pairs_for_retrieval(
    pairs_path: Path | None = None,
    golden_path: Path | None = None,
    exclude_golden: bool = True,
) -> pd.DataFrame:
    """Load support pairs and optionally drop Golden Set customer tweets."""
    path = Path(pairs_path) if pairs_path is not None else PROCESSED_SUPPORT_PAIRS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Support pairs not found: {path}. "
            "Run: python scripts/prepare_support_pairs.py"
        )

    df = pd.read_csv(
        path,
        dtype={"customer_tweet_id": str, "agent_tweet_id": str},
    )
    missing = [c for c in REQUIRED_PAIR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Support pairs missing columns: {missing}")

    df = df[REQUIRED_PAIR_COLUMNS].copy()
    df["customer_text"] = df["customer_text"].fillna("").astype(str)
    df["agent_text"] = df["agent_text"].fillna("").astype(str)
    df = df[df["customer_text"].str.strip().astype(bool)].reset_index(drop=True)

    if exclude_golden:
        exclude = load_golden_customer_tweet_ids(
            Path(golden_path) if golden_path is not None else GOLDEN_SET_CANDIDATES_PATH
        )
        if exclude:
            df = df[~df["customer_tweet_id"].astype(str).isin(exclude)].reset_index(drop=True)

    if df.empty:
        raise ValueError("No support pairs left to index after filtering.")
    return df


def build_retrieval_index(
    pairs_path: Path | None = None,
    golden_path: Path | None = None,
    index_path: Path | None = None,
    exclude_golden: bool = True,
) -> dict[str, Any]:
    """
    Fit TF-IDF on customer_text and save vectorizer + document matrix + metadata.

    Deterministic: fixed TfidfVectorizer settings; no randomness.
    """
    df = load_pairs_for_retrieval(
        pairs_path=pairs_path,
        golden_path=golden_path,
        exclude_golden=exclude_golden,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        lowercase=True,
        strip_accents="unicode",
    )
    doc_matrix = vectorizer.fit_transform(df["customer_text"].tolist())

    payload = {
        "vectorizer": vectorizer,
        "doc_matrix": doc_matrix,
        "customer_tweet_id": df["customer_tweet_id"].astype(str).tolist(),
        "agent_tweet_id": df["agent_tweet_id"].astype(str).tolist(),
        "customer_text": df["customer_text"].tolist(),
        "agent_text": df["agent_text"].tolist(),
        "meta": {
            "n_documents": int(len(df)),
            "n_features": int(len(vectorizer.get_feature_names_out())),
            "method": "tfidf_cosine",
            "ngram_range": (1, 2),
            "min_df": 2,
            "excluded_golden": bool(exclude_golden),
        },
    }

    out = Path(index_path) if index_path is not None else RETRIEVAL_INDEX_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)

    return {
        **payload["meta"],
        "index_path": str(out),
    }


def load_retrieval_index(index_path: Path | None = None) -> dict[str, Any]:
    """Load a previously saved retrieval index payload."""
    path = Path(index_path) if index_path is not None else RETRIEVAL_INDEX_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Retrieval index not found: {path}. "
            "Run: python scripts/build_retrieval_index.py"
        )
    payload = joblib.load(path)
    required = {
        "vectorizer",
        "doc_matrix",
        "customer_tweet_id",
        "agent_tweet_id",
        "customer_text",
        "agent_text",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Retrieval index missing keys: {missing}")
    if not sparse.issparse(payload["doc_matrix"]):
        raise TypeError("doc_matrix must be a scipy sparse matrix")
    return payload


def retrieve_similar(
    query: str,
    k: int = 5,
    index: dict[str, Any] | None = None,
    index_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k historical support pairs by cosine similarity on customer_text.

    Tie-break is deterministic: higher score first, then smaller customer_tweet_id.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    payload = index if index is not None else load_retrieval_index(index_path)
    text = str(query or "").strip()
    if not text:
        raise ValueError("query must be a non-empty customer message")

    vectorizer: TfidfVectorizer = payload["vectorizer"]
    doc_matrix = payload["doc_matrix"]
    query_vec = vectorizer.transform([text])
    scores = cosine_similarity(query_vec, doc_matrix).ravel()

    n = len(scores)
    take = min(k, n)
    # Stable ranking: (-score, customer_tweet_id)
    ids = np.asarray(payload["customer_tweet_id"], dtype=object)
    order = np.lexsort((ids, -scores))
    top_idx = order[:take]

    results: list[dict[str, Any]] = []
    for i in top_idx:
        results.append(
            {
                "customer_tweet_id": str(payload["customer_tweet_id"][i]),
                "agent_tweet_id": str(payload["agent_tweet_id"][i]),
                "customer_text": str(payload["customer_text"][i]),
                "agent_text": str(payload["agent_text"][i]),
                "similarity_score": float(scores[i]),
            }
        )
    return results
