"""
Load the Customer Support on Twitter CSV and rebuild AppleSupport conversations.

Dataset columns (twcs.csv):
  tweet_id, author_id, inbound, created_at, text,
  response_tweet_id, in_response_to_tweet_id

How conversations are rebuilt (simple English):
  1. Every tweet may point to a parent via in_response_to_tweet_id.
  2. Walk upward until there is no parent — that tweet is the conversation root.
  3. conversation_id = root tweet_id.
  4. Keep only conversations where AppleSupport appears at least once.

Memory note:
  The raw CSV is large (~492MB). We read it in chunks, force ID columns to
  strings (Twitter IDs are not safe as int64), and only keep tweets linked to
  the chosen brand before reconstructing threads.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.config import BRAND, RAW_CSV_PATH

REQUIRED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# Never let pandas infer these as integers — values are messy / too large / mixed.
ID_COLUMN_DTYPES = {
    "tweet_id": "string",
    "author_id": "string",
    "response_tweet_id": "string",
    "in_response_to_tweet_id": "string",
}

DEFAULT_CHUNKSIZE = 100_000

# Float-style artifacts like "123456.0" from CSV parsers (not scientific notation).
_FLOAT_ARTIFACT_RE = re.compile(r"^(\d+)\.0$")


def normalize_id(value: Any) -> str | None:
    """
    Normalize an identifier to a plain string, or None if missing/unusable.

    Important: do NOT use int()/float() for Twitter snowflake IDs — float loses
    precision above 2^53 and int parsing fails on mixed/malformed values.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "<na>", "none", "null", "nat"}:
        return None

    # Clean only the common "123.0" artifact; keep every other token as-is.
    match = _FLOAT_ARTIFACT_RE.fullmatch(text)
    if match:
        return match.group(1)
    return text


def parse_response_ids(value: Any) -> list[str]:
    """Split response_tweet_id which may contain comma-separated ids."""
    text = normalize_id(value)
    if text is None:
        return []
    parts: list[str] = []
    for part in text.split(","):
        cleaned = normalize_id(part)
        if cleaned is not None:
            parts.append(cleaned)
    return parts


def _normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Clean one CSV chunk: string IDs, text, timestamps, inbound flag."""
    missing = [c for c in REQUIRED_COLUMNS if c not in chunk.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    out = chunk.copy()
    out["tweet_id"] = out["tweet_id"].map(normalize_id)
    out["author_id"] = out["author_id"].map(normalize_id)
    out["in_response_to_tweet_id"] = out["in_response_to_tweet_id"].map(normalize_id)
    # May contain comma-separated ids; store as a single string token.
    out["response_tweet_id"] = out["response_tweet_id"].map(normalize_id)
    out["text"] = out["text"].fillna("").astype(str)
    # Dataset format: "Tue Oct 31 16:10:58 +0000 2017"
    out["created_at"] = pd.to_datetime(
        out["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
        utc=True,
    )

    if out["inbound"].dtype != bool:
        out["inbound"] = out["inbound"].astype(str).str.lower().isin({"true", "1", "yes"})

    out = out.dropna(subset=["tweet_id"])
    # author_id should always exist for brand filtering; if missing, drop.
    out = out.dropna(subset=["author_id"])
    return out


def iter_raw_chunks(
    csv_path: Path,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> Iterator[pd.DataFrame]:
    """Yield normalized chunks; all identifier columns forced to string."""
    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        dtype=ID_COLUMN_DTYPES,
        keep_default_na=True,
        # Strings like "" become NA; avoids silent int coercion.
        na_filter=True,
    )
    for chunk in reader:
        yield _normalize_chunk(chunk)


def _collect_brand_seed_ids(
    csv_path: Path,
    brand: str,
    chunksize: int,
) -> set[str]:
    """Pass 1: gather tweet ids for brand rows and their immediate parent/response links."""
    seed: set[str] = set()
    for chunk in iter_raw_chunks(csv_path, chunksize=chunksize):
        brand_rows = chunk.loc[chunk["author_id"] == brand]
        if brand_rows.empty:
            continue
        seed.update(brand_rows["tweet_id"].tolist())
        seed.update(
            tid for tid in brand_rows["in_response_to_tweet_id"].tolist() if tid is not None
        )
        for raw_resp in brand_rows["response_tweet_id"].tolist():
            seed.update(parse_response_ids(raw_resp))
    seed.discard(None)  # type: ignore[arg-type]
    return seed


def _load_tweets_for_ids(
    csv_path: Path,
    brand: str,
    seed_ids: set[str],
    chunksize: int,
) -> pd.DataFrame:
    """
    Pass 2: keep tweets that are brand-authored or linked to seed ids.

    Link rules (enough for typical short Twitter support threads):
      - tweet_id in seed_ids
      - author_id == brand
      - in_response_to_tweet_id in seed_ids (follow-ups to brand / seed tweets)
    """
    parts: list[pd.DataFrame] = []
    for chunk in iter_raw_chunks(csv_path, chunksize=chunksize):
        mask = (
            chunk["author_id"].eq(brand)
            | chunk["tweet_id"].isin(seed_ids)
            | chunk["in_response_to_tweet_id"].isin(seed_ids)
        )
        part = chunk.loc[mask]
        if not part.empty:
            parts.append(part)

    if not parts:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["tweet_id"], keep="first")
    return df


def load_raw_tweets(
    csv_path: Path | None = None,
    brand: str | None = BRAND,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """
    Load tweets with safe string IDs using chunked CSV reads.

    If brand is provided (default AppleSupport), only tweets linked to that brand
    are returned so the full 492MB file is not held in memory.
    If brand is None, all chunks are concatenated (use only for small test files).
    """
    path = Path(csv_path) if csv_path is not None else RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download Customer Support on Twitter and place twcs.csv there."
        )

    # Peek header so we fail fast on a bad file.
    header = pd.read_csv(path, nrows=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in header.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if brand is None:
        parts = list(iter_raw_chunks(path, chunksize=chunksize))
        if not parts:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        return pd.concat(parts, ignore_index=True).drop_duplicates(
            subset=["tweet_id"], keep="first"
        )

    seed_ids = _collect_brand_seed_ids(path, brand=brand, chunksize=chunksize)
    return _load_tweets_for_ids(path, brand=brand, seed_ids=seed_ids, chunksize=chunksize)


def build_parent_map(df: pd.DataFrame) -> dict[str, str | None]:
    """Map each tweet_id to its parent tweet_id (None if it has no parent)."""
    parents: dict[str, str | None] = {}
    for tweet_id, parent in zip(df["tweet_id"], df["in_response_to_tweet_id"], strict=True):
        parents[str(tweet_id)] = parent if parent is not None else None
    return parents


def find_root_id(
    tweet_id: str,
    parents: dict[str, str | None],
    cache: dict[str, str],
) -> str:
    """Walk up the reply chain until there is no known parent."""
    tweet_id = str(tweet_id)
    if tweet_id in cache:
        return cache[tweet_id]

    path: list[str] = []
    current = tweet_id

    while True:
        if current in cache:
            root = cache[current]
            break

        path.append(current)
        parent = parents.get(current)

        if parent is None:
            root = current
            break

        if parent not in parents:
            root = current
            break

        if parent in path:
            root = path[0]
            break

        current = parent

    for node in path:
        cache[node] = root
    return root


def assign_conversation_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add conversation_id (= root tweet_id) to every tweet row."""
    parents = build_parent_map(df)
    cache: dict[str, str] = {}
    out = df.copy()
    out["conversation_id"] = [
        find_root_id(str(tid), parents, cache) for tid in out["tweet_id"]
    ]
    return out


def filter_brand_conversations(df: pd.DataFrame, brand: str = BRAND) -> pd.DataFrame:
    """Keep tweets that belong to any conversation involving the brand account."""
    if "conversation_id" not in df.columns:
        df = assign_conversation_ids(df)

    brand_conversation_ids = set(df.loc[df["author_id"] == brand, "conversation_id"])
    return df[df["conversation_id"].isin(brand_conversation_ids)].copy()


def format_tweet_line(row: pd.Series) -> str:
    role = "customer" if bool(row["inbound"]) else "agent"
    return f"[{role} | {row['author_id']}] {row['text']}"


def build_conversation_table(brand_tweets: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse tweet-level rows into one row per conversation.

    Useful later for retrieval (customer_text → agent_text) and for exploration.
    """
    columns = [
        "conversation_id",
        "created_at",
        "num_tweets",
        "num_customer_tweets",
        "num_agent_tweets",
        "customer_text",
        "agent_text",
        "first_customer_tweet_id",
        "first_agent_tweet_id",
        "thread_text",
    ]
    if brand_tweets.empty:
        return pd.DataFrame(columns=columns)

    ordered = brand_tweets.sort_values(["conversation_id", "created_at", "tweet_id"])
    rows: list[dict[str, Any]] = []

    for conversation_id, group in ordered.groupby("conversation_id", sort=False):
        customer = group[group["inbound"]]
        agent = group[~group["inbound"]]

        rows.append(
            {
                "conversation_id": conversation_id,
                "created_at": group["created_at"].min(),
                "num_tweets": len(group),
                "num_customer_tweets": len(customer),
                "num_agent_tweets": len(agent),
                "customer_text": " ||| ".join(customer["text"].tolist()),
                "agent_text": " ||| ".join(agent["text"].tolist()),
                "first_customer_tweet_id": (
                    customer["tweet_id"].iloc[0] if len(customer) else pd.NA
                ),
                "first_agent_tweet_id": (
                    agent["tweet_id"].iloc[0] if len(agent) else pd.NA
                ),
                "thread_text": "\n".join(format_tweet_line(r) for _, r in group.iterrows()),
            }
        )

    return pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)


def prepare_brand_conversations(
    csv_path: Path | None = None,
    brand: str = BRAND,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full Step 1 pipeline.

    Returns:
      brand_tweets: tweet-level rows for conversations involving the brand
      conversations: one row per reconstructed conversation
    """
    raw = load_raw_tweets(csv_path, brand=brand, chunksize=chunksize)
    with_roots = assign_conversation_ids(raw)
    brand_tweets = filter_brand_conversations(with_roots, brand=brand)
    conversations = build_conversation_table(brand_tweets)
    return brand_tweets, conversations


def conversation_summary(conversations: pd.DataFrame) -> dict[str, Any]:
    """Small printable stats for sanity checks after prepare_data."""
    if conversations.empty:
        return {
            "num_conversations": 0,
            "avg_tweets_per_conversation": 0.0,
            "conversations_with_agent_reply": 0,
            "conversations_with_customer_message": 0,
        }

    return {
        "num_conversations": int(len(conversations)),
        "avg_tweets_per_conversation": float(conversations["num_tweets"].mean()),
        "conversations_with_agent_reply": int((conversations["num_agent_tweets"] > 0).sum()),
        "conversations_with_customer_message": int(
            (conversations["num_customer_tweets"] > 0).sum()
        ),
    }
