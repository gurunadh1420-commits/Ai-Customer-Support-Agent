"""
Conservative keyword/rule weak labels for Baseline 1 training data.

IMPORTANT:
  - These labels are NOT ground truth.
  - Ambiguous / conflicting texts are left unlabeled (empty weak_intent).
  - Golden Set examples must not be used as training rows.
  - This module does not train a classifier.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    GOLDEN_SET_CANDIDATES_PATH,
    INTENT_TAXONOMY,
    PROCESSED_SUPPORT_PAIRS_PATH,
)

# Patterns for other_or_unclear extensions (Rule C and Rule D)
_OOU_RULE_A = re.compile(
    r"^\s*(@\w+\s*)*"
    r"(thanks?(?:\s+you)?|thank\s+you|thx|ty|"
    r"please\s+help!?|help!?|"
    r"fix\s+(this|it)!?|"
    r"this\s+sucks!?|"
    r"ok\.?|okay\.?)\s*$",
    re.I,
)

_OOU_RULE_C = re.compile(
    r"^\s*(@\w+\s*)*(please\s+)?fix\s+"
    r"(this|it|that|your\s+[a-z0-9_-]+|the\s+[a-z0-9_-]+|everything)"
    r"(\s*@\w+)*\s*[.!?!]*\s*$",
    re.I,
)

_OOU_RULE_D = re.compile(
    r"\b("
    r"nothing\s*(helps?|works?|fixed|is\s*working)|"
    r"tried\s*everything|"
    r"still\s*(not\s*working|doesn'?t\s*work|the\s*same)|"
    r"same\s*(issue|problem)\s*(as\s*before|still)|"
    r"this\s*(still\s*)?isn'?t\s*(fixed|working|resolved)"
    r")\b",
    re.I,
)

_ANY_SPECIFIC_GUARD = re.compile(
    r"\b("
    r"batter(?:y|ies)|charg(?:e|ing|er)|lightning|cable|"
    r"update[sd]?|upgrad(?:e|ed|ing)|ios|ipados|macos|watchos|"
    r"screen|display|speaker|microphone|mic|camera|button|touch\s*bar|hardware|headphone|jack|"
    r"wi[- ]?fi|bluetooth|airdrop|internet|service|cellular|sim|hotspot|"
    r"apple\s*id|icloud|account|password|2fa|verification|passcode|sign\s*in|log\s*in|login|"
    r"app\s*store|"
    r"safari|messages|imessage|mail|photos|facetime|podcasts?|notes|calendar|maps|itunes|"
    r"crash|crashes|crashing|freez(?:e|es|ing)|frozen|"
    r"refund|charged|payment|billing|purchase[sd]?|bought|"
    r"subscription|family\s*sharing|apple\s*music|"
    r"storage|space|"
    r"how\s+(do|can|to)\s+i|where\s+(do|can)\s+i"
    r")\b",
    re.I,
)


class _OOUCustomMatcher:
    """Matches Rule A unconditionally, or Rule C/D if no specific keyword matches."""

    def search(self, text: str) -> re.Match[str] | None:
        ma = _OOU_RULE_A.search(text)
        if ma:
            return ma
        if _ANY_SPECIFIC_GUARD.search(text):
            return None
        return _OOU_RULE_C.search(text) or _OOU_RULE_D.search(text)


# Stronger, more conservative patterns than golden-set stratification hints.
_STRONG: dict[str, Any] = {
    "battery_drain": re.compile(
        r"\b("
        r"battery\s*life|battery\s*drain|battery\s*drains|battery\s*draining|"
        r"draining\s*(my\s*)?battery|killing\s*(my\s*)?battery|"
        r"battery\s*(dies|dying|dead|performance)|"
        r"batter(?:y|ies)\s*(issue|problem)|"
        r"(drain|drains|draining)\s*(so\s*)?(fast|quickly)|"
        r"%\s*(drop|drops|dropping)|percent\s*(drop|drops)"
        r")\b",
        re.I,
    ),
    "charging_issue": re.compile(
        r"\b("
        r"won'?t\s*charge|will\s*not\s*charge|not\s*charging|doesn'?t\s*charge|"
        r"slow\s*charg(?:e|ing)|charg(?:er|ing)\s*(broke|broken|burst|split|faulty|issue|problem)|"
        r"lightning\s*(cable|port)|wireless\s*charg|"
        r"accessory\s*(not\s*supported|may\s*not\s*be\s*supported)|"
        r"keeps?\s*(unplugg|disconnect)|plug(?:ged)?\s*in.{0,40}(not|won'?t|doesn'?t)\s*charg"
        r")\b",
        re.I,
    ),
    "software_update_issue": re.compile(
        r"\b("
        r"after\s*(the\s*)?(ios|ipados|macos|watchos)?\s*(update|upgrading|upgrade)|"
        r"since\s*(the\s*)?(ios|ipados|macos|watchos)?\s*(update|upgraded|updating|upgrade)|"
        r"latest\s*(ios\s*)?update|new\s*(ios\s*)?update|software\s*update|"
        r"updated\s*(to\s*)?(ios|ipados|macos)\s*\d+|"
        r"ios\s*11(\.\d+)?\s*(bug|issue|problem|broke|broken|sucks|trash|ruined)|"
        r"(bug|issue|problem|broke|broken|sucks|trash|ruined).{0,30}ios\s*11|"
        r"can'?t\s*(install|download)\s*(the\s*)?(update|ios)|"
        r"verifying\s*update|update\s*(failed|error|stuck)"
        r")\b",
        re.I,
    ),
    "hardware_issue": re.compile(
        r"\b("
        r"touch\s*bar\s*(not|isn'?t|won'?t|broken|dead)|"
        r"(screen|display)\s*(crack(?:ed)?|black|blank|flicker|frozen|unresponsive)|"
        r"(speaker|microphone|camera)\s*(not|isn'?t|won'?t|broken|faulty|dead|issue|problem)|"
        r"(home|power|side)\s*button\s*(not|isn'?t|won'?t|broken|stuck|issue)|"
        r"won'?t\s*turn\s*on|boot\s*loop|overheat(?:ing)?|"
        r"swollen\s*battery|hardware\s*(fault|failure|issue|problem)"
        r")\b",
        re.I,
    ),
    "network_connectivity": re.compile(
        r"\b("
        r"wi[- ]?fi\s*(not|won'?t|can'?t|keeps?|drop|drops|dropping|issue|problem)|"
        r"(can'?t|cannot|unable\s*to)\s*connect\s*(to\s*)?(wi[- ]?fi|wifi|bluetooth)|"
        r"bluetooth\s*(not|won'?t|issue|problem|garbled|drop)|"
        r"airdrop\s*(not|won'?t|issue)|"
        r"no\s*(internet|service)|cellular\s*data\s*(not|won'?t|issue)|"
        r"sim\s*card|no\s*sim|invalid\s*sim|sim\s*(failure|not\s*support)|"
        r"hotspot\s*(not|won'?t|issue)"
        r")\b",
        re.I,
    ),
    "account_access": re.compile(
        r"\b("
        r"(apple\s*id|icloud|account)\s*(is\s*)?(locked|hacked|stolen|disabled)|"
        r"locked\s*out\s*(of\s*)?(my\s*)?(apple\s*id|icloud|account|itunes)|"
        r"(can'?t|cannot|unable\s*to)\s*(sign|log)\s*in|"
        r"two[- ]factor|2fa|verification\s*code|confirmation\s*code|"
        r"(forgot|reset|change)\s*(my\s*)?(password|apple\s*id)|"
        r"password\s*(not\s*working|incorrect|wrong|not\s*recognised|not\s*recognized)|"
        r"trusted\s*(phone|device)\s*(number)?"
        r")\b",
        re.I,
    ),
    "app_store_issue": re.compile(
        r"\b("
        r"app\s*store\s*(not|won'?t|can'?t|isn'?t|down|crash|freez|loading|spinning|play(?:ing)?\s*up)|"
        r"(can'?t|cannot|unable\s*to)\s*(download|update)\s*(any\s*)?(apps?|from\s*(the\s*)?app\s*store)|"
        r"(download|update)\s*(apps?|from\s*(the\s*)?app\s*store).{0,20}(can'?t|cannot|unable|fail)|"
        r"app\s*store.{0,40}(password\s*again|blank\s*page|keeps?\s*asking)|"
        r"updates?\s*available.{0,40}app\s*store|app\s*store.{0,40}updates?\s*available"
        r")\b",
        re.I,
    ),
    "app_issue": re.compile(
        r"\b("
        r"(safari|messages|imessage|mail|photos|facetime|podcasts?|notes|calendar|maps|"
        r"music\s*app|watch\s*app)\s*"
        r"(not|isn'?t|won'?t|can'?t|crash|crashes|crashing|freez|frozen|broken|issue|problem)|"
        r"(crash|crashes|crashing|freez(?:e|es|ing)|won'?t\s*open|not\s*working)\b.{0,20}\b"
        r"(safari|messages|imessage|mail|photos|facetime|podcasts?|itunes)|"
        r"(safari|messages|imessage|mail|photos|facetime|podcasts?|itunes)\b.{0,20}\b"
        r"(crash|crashes|crashing|freez(?:e|es|ing)|won'?t\s*open|not\s*working)"
        r")\b",
        re.I,
    ),
    "payment_or_refund": re.compile(
        r"\b("
        r"refund|want\s*(my\s*)?money\s*back|"
        r"charged\s*(me\s*)?(twice|again|for)|unauthorized\s*charge|"
        r"payment\s*(method|declined|rejected|failed|issue|info)|"
        r"accidentally\s*(purchased|bought)|didn'?t\s*(mean|want)\s*to\s*(purchase|buy)|"
        r"billing\s*(issue|problem|error)"
        r")\b",
        re.I,
    ),
    "subscription_management": re.compile(
        r"\b("
        r"manage\s*subscription|cancel\s*(my\s*)?(subscription|apple\s*music|icloud)|"
        r"(can'?t|cannot|unable\s*to)\s*(find|see|manage|change|cancel)\s*"
        r"(my\s*)?(subscription|manage\s*button)|"
        r"subscription\s*button|family\s*sharing|"
        r"apple\s*music\s*(subscription|family\s*plan|plan)|"
        r"renew\s*(my\s*)?(subscription|plan)|telling\s*me\s*to\s*renew"
        r")\b",
        re.I,
    ),
    "storage_issue": re.compile(
        r"\b("
        r"storage\s*(almost\s*)?full|not\s*enough\s*storage|out\s*of\s*storage|"
        r"(iphone|icloud)\s*storage\s*(full|almost|issue|problem)|"
        r"free\s*up\s*space|storage\s*notification|"
        r"says?\s*(not\s*enough\s*storage|storage\s*full)"
        r")\b",
        re.I,
    ),
    "howto_or_feature": re.compile(
        r"^\s*(@\w+\s*)*(hi[,!]?\s+|hello[,!]?\s+|hey[,!]?\s+)*"
        r"(how\s+(do|can|to)\s+i|where\s+(do|can)\s+i|is\s+it\s+possible\s+to|"
        r"how\s+to\s+)\b",
        re.I,
    ),
    "other_or_unclear": _OOUCustomMatcher(),
}

# When both match, prefer the more specific primary need (single-label policy).
_PRIORITY_WHEN_TIE: list[tuple[str, str, str]] = [
    # (a, b, winner) — only used when exactly these two match
    ("battery_drain", "software_update_issue", "battery_drain"),
    ("charging_issue", "software_update_issue", "charging_issue"),
    ("battery_drain", "charging_issue", ""),  # conflict → unlabeled unless resolved below
    ("payment_or_refund", "subscription_management", ""),
    ("app_store_issue", "app_issue", "app_store_issue"),
    ("hardware_issue", "software_update_issue", ""),
    ("network_connectivity", "software_update_issue", "network_connectivity"),
    ("account_access", "software_update_issue", "account_access"),
    ("storage_issue", "software_update_issue", "storage_issue"),
    ("app_issue", "software_update_issue", ""),
    ("howto_or_feature", "battery_drain", "battery_drain"),
    ("howto_or_feature", "charging_issue", "charging_issue"),
    ("howto_or_feature", "account_access", "account_access"),
    ("howto_or_feature", "app_store_issue", "app_store_issue"),
    ("howto_or_feature", "payment_or_refund", "payment_or_refund"),
    ("howto_or_feature", "subscription_management", "subscription_management"),
    ("howto_or_feature", "storage_issue", "storage_issue"),
    ("howto_or_feature", "network_connectivity", "network_connectivity"),
    ("howto_or_feature", "hardware_issue", "hardware_issue"),
    ("howto_or_feature", "app_issue", "app_issue"),
    ("howto_or_feature", "software_update_issue", "software_update_issue"),
]


def find_candidate_intents(text: str) -> list[str]:
    """Return all strong-pattern matches (may be empty or multiple)."""
    text = str(text or "")
    return [name for name in INTENT_TAXONOMY if _STRONG[name].search(text)]


def _resolve_battery_vs_charging(text: str) -> str | None:
    t = text.lower()
    drainish = bool(
        re.search(r"battery\s*life|drain|draining|%|percent|dies\s*fast|dying", t)
    )
    chargeish = bool(
        re.search(
            r"won'?t\s*charge|not\s*charging|charger|cable|lightning|wireless\s*charg|"
            r"accessory\s*(not|may\s*not)",
            t,
        )
    )
    if drainish and not chargeish:
        return "battery_drain"
    if chargeish and not drainish:
        return "charging_issue"
    return None


def _resolve_payment_vs_subscription(text: str) -> str | None:
    t = text.lower()
    payish = bool(
        re.search(r"refund|charged|money\s*back|payment\s*method|accidentally\s*(purchased|bought)", t)
    )
    subish = bool(
        re.search(r"manage\s*subscription|cancel\s*(my\s*)?subscription|subscription\s*button|renew\s*(my\s*)?plan", t)
    )
    if payish and not subish:
        return "payment_or_refund"
    if subish and not payish:
        return "subscription_management"
    return None


def _resolve_update_vs_app(text: str) -> str | None:
    """Several things broken after update → software_update; one named app → app_issue."""
    t = text.lower()
    multi = bool(
        re.search(
            r"(all\s*apps|every\s*app|apps?\s*(keep\s*)?crash|phone\s*(is\s*)?(slow|sluggish|freez)|"
            r"ruined\s*my\s*(phone|experience)|everything\s*(broke|broken))",
            t,
        )
    )
    if multi:
        return "software_update_issue"
    named = bool(
        re.search(
            r"\b(safari|messages|imessage|mail|photos|facetime|podcasts?|maps|notes|calendar)\b",
            t,
        )
    )
    if named and not multi:
        return "app_issue"
    return None


def _resolve_update_vs_hardware(text: str) -> str | None:
    t = text.lower()
    physical = bool(
        re.search(
            r"crack(?:ed)?|swollen|won'?t\s*turn\s*on|boot\s*loop|"
            r"(home|power|side)\s*button|touch\s*bar",
            t,
        )
    )
    if physical:
        return "hardware_issue"
    # "after update speaker not working" — treat as update regression unless clearly physical damage
    if re.search(r"after\s*(the\s*)?update|since\s*(the\s*)?update|ios\s*\d+", t):
        return "software_update_issue"
    return None


def choose_weak_intent(text: str) -> str | None:
    """
    Assign a single weak intent, or None if evidence is weak/ambiguous.

    Returns None (unlabeled) rather than guessing.
    """
    text = str(text or "").strip()
    if not text:
        return None

    matches = find_candidate_intents(text)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    matched = set(matches)

    # Special pairwise resolvers first.
    if matched == {"battery_drain", "charging_issue"} or matched == {
        "battery_drain",
        "charging_issue",
        "software_update_issue",
    }:
        # Drop update for battery/charging primary policy, then resolve pair.
        resolved = _resolve_battery_vs_charging(text)
        if resolved:
            return resolved
        if matched == {"battery_drain", "charging_issue", "software_update_issue"}:
            # Still conflicting battery vs charging.
            return None

    if "battery_drain" in matched and "charging_issue" in matched:
        resolved = _resolve_battery_vs_charging(text)
        if not resolved:
            return None
        # Continue with remaining matches after replacing the pair with winner.
        matched = (matched - {"battery_drain", "charging_issue"}) | {resolved}
        if len(matched) == 1:
            return next(iter(matched))

    if matched == {"payment_or_refund", "subscription_management"}:
        return _resolve_payment_vs_subscription(text)

    if matched == {"app_issue", "software_update_issue"}:
        return _resolve_update_vs_app(text)

    if matched == {"hardware_issue", "software_update_issue"}:
        return _resolve_update_vs_hardware(text)

    # Generic pairwise priority table for exactly two matches.
    if len(matched) == 2:
        a, b = sorted(matched)
        for x, y, winner in _PRIORITY_WHEN_TIE:
            if matched == {x, y}:
                return winner or None

    # howto + one other already covered by table; if howto + many → drop howto
    if "howto_or_feature" in matched and len(matched) > 1:
        matched = matched - {"howto_or_feature"}
        if len(matched) == 1:
            return next(iter(matched))
        # re-enter simple path
        if matched == {"payment_or_refund", "subscription_management"}:
            return _resolve_payment_vs_subscription(text)
        if matched == {"app_issue", "software_update_issue"}:
            return _resolve_update_vs_app(text)
        if matched == {"hardware_issue", "software_update_issue"}:
            return _resolve_update_vs_hardware(text)
        if matched == {"battery_drain", "charging_issue"}:
            return _resolve_battery_vs_charging(text)
        if len(matched) == 2:
            for x, y, winner in _PRIORITY_WHEN_TIE:
                if matched == {x, y}:
                    return winner or None

    # battery/charging + update (+ maybe others): prefer symptom over update
    if "software_update_issue" in matched:
        if "battery_drain" in matched and "charging_issue" not in matched:
            return "battery_drain"
        if "charging_issue" in matched and "battery_drain" not in matched:
            return "charging_issue"

    # Too many or unresolved conflicts → unlabeled
    return None


def load_golden_customer_tweet_ids(golden_path: Path | None = None) -> set[str]:
    """Tweet IDs already reserved for the Golden Set (must not train on them)."""
    path = Path(golden_path) if golden_path is not None else GOLDEN_SET_CANDIDATES_PATH
    if not path.exists():
        return set()
    g = pd.read_csv(path, dtype={"customer_tweet_id": str})
    return set(g["customer_tweet_id"].dropna().astype(str))


def build_weak_training_frame(
    pairs: pd.DataFrame,
    exclude_customer_tweet_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply weak labels to support pairs; return labelled rows only + summary stats.

    Stats cover the full pairs table after Golden Set exclusion.
    """
    exclude = exclude_customer_tweet_ids or set()
    df = pairs.copy()
    df["customer_tweet_id"] = df["customer_tweet_id"].astype(str)
    df["agent_tweet_id"] = df["agent_tweet_id"].astype(str)

    total_pairs = int(len(df))
    df = df[~df["customer_tweet_id"].isin(exclude)].copy()
    after_exclude = int(len(df))

    labels: list[str | None] = [
        choose_weak_intent(t) for t in df["customer_text"].astype(str)
    ]
    df["weak_intent"] = labels

    labelled = df[df["weak_intent"].notna()].copy()
    unlabelled_n = int(df["weak_intent"].isna().sum())
    labelled_n = int(len(labelled))

    counts = (
        labelled["weak_intent"].value_counts().reindex(INTENT_TAXONOMY, fill_value=0).astype(int).to_dict()
    )

    out = labelled[
        ["customer_tweet_id", "agent_tweet_id", "customer_text", "weak_intent"]
    ].reset_index(drop=True)

    summary = {
        "total_historical_pairs": total_pairs,
        "excluded_golden_customer_tweets": int(len(exclude)),
        "pairs_after_golden_exclusion": after_exclude,
        "labelled_examples": labelled_n,
        "unlabelled_examples": unlabelled_n,
        "count_per_weak_intent": counts,
    }
    return out, summary


def prepare_weak_training(
    pairs_path: Path | None = None,
    golden_path: Path | None = None,
    out_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load pairs, exclude Golden Set IDs, write weak training CSV."""
    from src.config import WEAK_TRAINING_PATH

    pairs_path = Path(pairs_path) if pairs_path else PROCESSED_SUPPORT_PAIRS_PATH
    out_path = Path(out_path) if out_path else WEAK_TRAINING_PATH

    pairs = pd.read_csv(
        pairs_path,
        dtype={"customer_tweet_id": str, "agent_tweet_id": str},
    )
    exclude = load_golden_customer_tweet_ids(golden_path)
    labelled, summary = build_weak_training_frame(pairs, exclude_customer_tweet_ids=exclude)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(out_path, index=False)
    summary["out_path"] = str(out_path)
    return labelled, summary
