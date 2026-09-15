"""
Temporary exploratory analysis of AppleSupport support pairs.

Theme counts are keyword heuristics only — NOT ground-truth intent labels.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import PROCESSED_SUPPORT_PAIRS_PATH

# Exploratory keyword themes only (not final intents / not eval labels).
THEME_PATTERNS: dict[str, re.Pattern[str]] = {
    "battery_charging": re.compile(
        r"\b(batter(?:y|ies)|charg(?:e|ing|er)|drain(?:ing|s|ed)?|"
        r"power\s*bank|won'?t\s*charge|not\s*charging|dies?\s*fast)\b",
        re.I,
    ),
    "ios_software_update": re.compile(
        r"\b(ios\s*\d*|ipados|macos|watchos|software|update(?:d|s|ing)?|"
        r"upgrad(?:e|ed|ing)|high\s*sierra|sierra|install(?:ing|ed|ation)?|"
        r"bug|glitch|crash(?:es|ing|ed)?|freeze|frozen|boot\s*loop)\b",
        re.I,
    ),
    "device_hardware": re.compile(
        r"\b(iphone|ipad|macbook|imac|airpods|watch|apple\s*watch|ipod|"
        r"screen|display|touch\s*bar|speaker|microphone|camera|button|"
        r"home\s*button|lightning|usb[- ]?c|hardware|crack(?:ed)?|broken|"
        r"overheat(?:ing)?|heat(?:ing)?)\b",
        re.I,
    ),
    "connectivity": re.compile(
        r"\b(wifi|wi[- ]?fi|bluetooth|cellular|lte|5g|4g|signal|network|"
        r"connect(?:ion|ivity|ing|ed)?|internet|hotspot|carrier|sim\s*card|"
        r"no\s*service|airdrop|icloud\s*drive)\b",
        re.I,
    ),
    "apple_id_account": re.compile(
        r"\b(apple\s*id|icloud|password|two[- ]factor|2fa|verification\s*code|"
        r"locked\s*(out|account)|account\s*locked|sign[- ]?in|login|log\s*in|"
        r"authentication|security\s*questions|forgot\s*(my\s*)?password)\b",
        re.I,
    ),
    "apps_app_store": re.compile(
        r"\b(app\s*store|apps?|download(?:ing|ed)?|install(?:ing|ed)?\s*app|"
        r"podcast|music|itunes|safari|messages|mail\s*app|photos\s*app|"
        r"testflight|delete\s*apps?)\b",
        re.I,
    ),
    "payments_refunds": re.compile(
        r"\b(refund|payment|charged|billing|invoice|receipt|purchase|"
        r"credit\s*card|debit|money|paid|pay\s*for)\b",
        re.I,
    ),
    "subscriptions": re.compile(
        r"\b(subscription|subscribe|renew(?:al|ed|ing)?|apple\s*music|"
        r"apple\s*tv\+|icloud\s*\+|apple\s*care|applecare|family\s*sharing|"
        r"monthly|cancel\s*(my\s*)?subscription)\b",
        re.I,
    ),
}

RESPONSE_PATTERNS: dict[str, re.Pattern[str]] = {
    "dm_redirect": re.compile(
        r"\b(dm\b|direct\s*message|send\s*us\s*a\s*dm|join\s*us\s*in\s*a\s*dm|"
        r"more\s*room)\b|https://t\.co/\w+",
        re.I,
    ),
    "asks_device_or_ios": re.compile(
        r"\b(which\s*(iphone|ipad|device|mac|version)|what\s*(iphone|ipad|device|version)|"
        r"ios\s*version|software\s*version|model)\b",
        re.I,
    ),
    "troubleshooting_question": re.compile(
        r"\b(can\s*you|could\s*you|have\s*you|did\s*you|does\s*(it|this)|"
        r"when\s*did|how\s*long|any\s*steps|tried|happening)\b.*\?",
        re.I,
    ),
    "link_or_article": re.compile(r"https?://|support\.apple\.com|article", re.I),
    "direct_instructions": re.compile(
        r"\b(go\s*to|open\s*settings|tap|select|follow\s*these\s*steps|"
        r"try\s*these|restart|reset|update\s*to|check\s*from)\b",
        re.I,
    ),
}

TOKEN_RE = re.compile(r"[a-z0-9']+")
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "as", "is", "are", "was", "were", "be", "been", "am", "i", "me", "my",
    "we", "you", "your", "it", "its", "this", "that", "with", "from", "by",
    "not", "no", "so", "do", "does", "did", "have", "has", "had", "will",
    "would", "can", "could", "just", "about", "when", "what", "which", "who",
    "how", "why", "all", "any", "some", "than", "then", "there", "here",
    "out", "up", "down", "into", "over", "after", "before", "also", "too",
    "very", "really", "get", "got", "im", "ive", "dont", "cant", "wont",
    "applesupport", "https", "http", "t", "co", "rt", "amp", "please", "hi",
    "hey", "hello", "thanks", "thank", "help", "need", "still", "now", "one",
    "like", "even", "back", "via",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(str(text).lower()) if t not in STOPWORDS and len(t) > 2]


def bigrams(tokens: list[str]) -> list[str]:
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def matched_themes(text: str) -> list[str]:
    return [name for name, pat in THEME_PATTERNS.items() if pat.search(text)]


def main() -> None:
    path = PROCESSED_SUPPORT_PAIRS_PATH
    print(f"Loading {path}")
    df = pd.read_csv(path, dtype={"customer_tweet_id": str, "agent_tweet_id": str})
    print(f"rows={len(df):,}")

    # --- word / phrase frequencies ---
    word_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    for text in df["customer_text"].astype(str):
        toks = tokenize(text)
        word_counts.update(toks)
        bigram_counts.update(bigrams(toks))

    print("\n=== TOP CUSTOMER UNIGRAMS ===")
    for w, c in word_counts.most_common(40):
        print(f"{c:7d}  {w}")

    print("\n=== TOP CUSTOMER BIGRAMS ===")
    for w, c in bigram_counts.most_common(40):
        print(f"{c:7d}  {w}")

    # --- theme tagging ---
    theme_lists = df["customer_text"].astype(str).map(matched_themes)
    df = df.copy()
    df["themes"] = theme_lists
    df["theme_count"] = df["themes"].map(len)
    df["theme_primary"] = df["themes"].map(lambda xs: xs[0] if xs else "other")

    # multi-label counts (a message can count in multiple themes)
    multi_counts = {name: 0 for name in THEME_PATTERNS}
    multi_counts["other"] = 0
    for themes in df["themes"]:
        if not themes:
            multi_counts["other"] += 1
        else:
            for t in themes:
                multi_counts[t] += 1

    print("\n=== EXPLORATORY THEME COUNTS (multi-label; NOT ground truth) ===")
    for name, c in sorted(multi_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        pct = 100.0 * c / len(df)
        print(f"{c:7d}  ({pct:5.1f}%)  {name}")

    multi_problem = df[df["theme_count"] >= 2]
    print(f"\nMessages matching >=2 theme patterns: {len(multi_problem):,} "
          f"({100*len(multi_problem)/len(df):.1f}%)")
    print(f"Messages matching 0 themes (other): {(df['theme_count']==0).sum():,}")

    print("\n=== MULTI-PROBLEM EXAMPLES ===")
    for i, row in multi_problem.head(8).iterrows():
        print(f"- themes={row['themes']}")
        print(f"  {row['customer_text'][:240]}")

    # --- examples per theme ---
    print("\n=== REPRESENTATIVE EXAMPLES PER THEME ===")
    for theme in list(THEME_PATTERNS) + ["other"]:
        if theme == "other":
            subset = df[df["theme_count"] == 0]
        else:
            subset = df[df["themes"].map(lambda xs, t=theme: t in xs)]
        print(f"\n--- {theme} (n={len(subset):,}) ---")
        # prefer shorter clearer examples
        sample = subset.copy()
        sample["len"] = sample["customer_text"].astype(str).str.len()
        sample = sample[(sample["len"] >= 40) & (sample["len"] <= 220)].head(8)
        if sample.empty:
            sample = subset.head(8)
        for _, row in sample.iterrows():
            print(f"  C: {row['customer_text']}")
            print(f"  A: {row['agent_text'][:180]}")
            print()

    # --- agent response patterns ---
    print("\n=== AGENT RESPONSE PATTERN COUNTS (exploratory; overlapping) ===")
    agent = df["agent_text"].astype(str)
    for name, pat in RESPONSE_PATTERNS.items():
        c = int(agent.map(lambda t, p=pat: bool(p.search(t))).sum())
        print(f"{c:7d}  ({100*c/len(df):5.1f}%)  {name}")

    # DM-like more precise
    dm_only = agent.str.contains(r"\bDM\b|direct message", case=False, na=False)
    has_link = agent.str.contains(r"https?://", case=False, na=False)
    print(f"{int(dm_only.sum()):7d}  ({100*dm_only.mean()*100/100:5.1f}%)  dm_keyword_only")
    print(f"{int(has_link.sum()):7d}  ({100*has_link.mean()*100/100:5.1f}%)  contains_url")

    print("\n=== AGENT PATTERN EXAMPLES ===")
    for name, pat in RESPONSE_PATTERNS.items():
        hits = df[agent.map(lambda t, p=pat: bool(p.search(t)))].head(3)
        print(f"\n[{name}]")
        for _, row in hits.iterrows():
            print(f"  A: {row['agent_text'][:220]}")


if __name__ == "__main__":
    main()
