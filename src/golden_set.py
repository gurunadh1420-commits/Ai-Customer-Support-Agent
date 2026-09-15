"""
Stratified sampling of Golden Set *candidates* for manual annotation.

IMPORTANT:
  - proposed_intent is a MACHINE-GENERATED suggestion only.
  - It is NOT ground truth.
  - annotator_label must be filled by a human.
  - This module does not train a classifier.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.config import INTENT_TAXONOMY, PROCESSED_SUPPORT_PAIRS_PATH

# Heuristic patterns used only to stratify / suggest candidates — not final labels.
_PATTERNS: dict[str, re.Pattern[str]] = {
    "battery_drain": re.compile(
        r"\b(battery\s*life|batter(?:y|ies)\s*(drain|drains|draining|dies|dying|dead|low|issue|problem)|"
        r"drain(?:s|ing)?\s*(my\s*)?battery|killing\s*(my\s*)?battery|battery\s*performance|"
        r"runs?\s*down|dies?\s*fast)\b",
        re.I,
    ),
    "charging_issue": re.compile(
        r"\b(charg(?:e|ing|er)|won'?t\s*charge|not\s*charging|slow\s*charg|"
        r"lightning\s*cable|wireless\s*charg|plug(?:ged)?\s*in)\b",
        re.I,
    ),
    "software_update_issue": re.compile(
        r"\b((ios|ipados|macos|watchos)\s*\d*|ios11|high\s*sierra|"
        r"after\s*(the\s*)?(update|upgrading|upgrade)|since\s*(the\s*)?(update|upgraded|updating)|"
        r"new\s*update|latest\s*update|software\s*update|updated\s*(my\s*)?(phone|iphone|ipad|mac))\b",
        re.I,
    ),
    "hardware_issue": re.compile(
        r"\b(touch\s*bar|screen\s*(crack|black|blank|flicker|freeze|frozen|unresponsive)|"
        r"display\s*(issue|problem|black)|speaker|microphone|"
        r"home\s*button|power\s*button|overheat|hardware|"
        r"won'?t\s*turn\s*on|boot\s*loop|frozen\s*screen)\b",
        re.I,
    ),
    "network_connectivity": re.compile(
        r"\b(wi[- ]?fi|wifi|bluetooth|cellular|lte|5g|4g|hotspot|airdrop|"
        r"no\s*internet|can'?t\s*connect|connection\s*(issue|problem|drop)|"
        r"network\s*(issue|problem)|sim\s*card|no\s*sim|invalid\s*sim|no\s*service|"
        r"carrier\s*(settings|update)|sim\s*failure)\b",
        re.I,
    ),
    "account_access": re.compile(
        r"\b((apple\s*id|icloud|account)\s*(locked|hacked|stolen|disabled)|"
        r"locked\s*out|can'?t\s*(sign|log)\s*in|two[- ]factor|2fa|"
        r"verification\s*code|confirmation\s*code|"
        r"reset\s*(my\s*)?(password|apple\s*id)|forgot\s*(my\s*)?password|"
        r"change\s*(my\s*)?password|password\s*(not\s*working|incorrect|wrong))\b",
        re.I,
    ),
    "app_store_issue": re.compile(
        r"\b(app\s*store|download\s*(app|apps|from)|can'?t\s*download|"
        r"unable\s*to\s*download|app\s*store\s*(not|won|crash|connect|loading))\b",
        re.I,
    ),
    "app_issue": re.compile(
        r"\b(safari|messages|imessage|mail|photos|facetime|itunes|"
        r"podcast|notes|calendar|maps|"
        r"(app|apps)\s*(crash|crashes|crashing|freez|not\s*working|won'?t\s*open))\b",
        re.I,
    ),
    "payment_or_refund": re.compile(
        r"\b(refund|charged\s*(me|twice|again)|unauthorized\s*charge|"
        r"payment\s*(method|declined|rejected|failed|issue)|billing|"
        r"want\s*(my\s*)?money\s*back|receipt|invoice)\b",
        re.I,
    ),
    "subscription_management": re.compile(
        r"\b(subscription|subscribe|renew(?:al)?|cancel\s*(my\s*)?(subscription|apple\s*music|icloud)|"
        r"manage\s*subscription|family\s*sharing|apple\s*music\s*(family|plan)|"
        r"icloud\+|apple\s*tv\+|applecare|apple\s*care)\b",
        re.I,
    ),
    "storage_issue": re.compile(
        r"\b(storage\s*(full|almost\s*full)|not\s*enough\s*storage|"
        r"iphone\s*storage|icloud\s*storage|out\s*of\s*storage|free\s*up\s*space)\b",
        re.I,
    ),
    "howto_or_feature": re.compile(
        r"\b(how\s*(do|can|to)\s+i|where\s*(do|can)\s+i|is\s*it\s*possible|"
        r"how\s*to\s*(delete|turn|enable|disable|setup|set\s*up|find|connect))\b",
        re.I,
    ),
    "other_or_unclear": re.compile(
        r"^\s*@?\w*\s*(help|please\s*help|fix\s*(this|it)?|this\s*sucks|"
        r"what('?s|\s+is)\s*wrong|not\s*working|doesn'?t\s*work|issue|problem|"
        r"thanks|thank\s*you|ok\s*that\s*settled|you'?re\s*welcome)\b",
        re.I,
    ),
}

# Boundary pools: (name, intent_a, intent_b) — rows matching both patterns.
_BOUNDARIES: list[tuple[str, str, str]] = [
    ("battery_vs_charging", "battery_drain", "charging_issue"),
    ("update_vs_app", "software_update_issue", "app_issue"),
    ("update_vs_hardware", "software_update_issue", "hardware_issue"),
    ("app_store_vs_app", "app_store_issue", "app_issue"),
    ("payment_vs_subscription", "payment_or_refund", "subscription_management"),
]

_SIM_PATTERN = re.compile(
    r"\b(sim\s*card|no\s*sim|invalid\s*sim|sim\s*failure|sim\s*not\s*support)\b",
    re.I,
)
_PASSWORD_PATTERN = re.compile(
    r"\b(reset\s*(my\s*)?(password|apple\s*id)|forgot\s*(my\s*)?password|"
    r"change\s*(my\s*)?password)\b",
    re.I,
)
_LOCK_PATTERN = re.compile(
    r"\b((apple\s*id|icloud|account)\s*(locked|disabled)|locked\s*out|"
    r"verification\s*code|two[- ]factor|2fa)\b",
    re.I,
)


def matched_intents(text: str) -> list[str]:
    """Return taxonomy intents whose heuristic patterns match the text."""
    return [name for name in INTENT_TAXONOMY if _PATTERNS[name].search(str(text))]


def format_proposed_intent(primary: str, also_matched: list[str] | None = None) -> str:
    """Machine-generated candidate string — never ground truth."""
    base = f"MACHINE_CANDIDATE: {primary}"
    extras = [m for m in (also_matched or []) if m != primary]
    if extras:
        base += f" (also_matched: {','.join(extras)})"
    return base


def _sample_indices(
    pool: np.ndarray,
    k: int,
    rng: np.random.Generator,
    used: set[int],
) -> list[int]:
    available = [i for i in pool.tolist() if i not in used]
    if not available:
        return []
    take = min(k, len(available))
    chosen = rng.choice(available, size=take, replace=False).tolist()
    used.update(chosen)
    return chosen


def build_golden_set_candidates(
    pairs: pd.DataFrame | None = None,
    target_size: int = 200,
    per_intent: int = 14,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Stratified sample of support pairs for manual golden-set annotation.

    Returns:
      candidates dataframe ready for CSV export
      summary stats (not evaluation metrics)
    """
    if pairs is None:
        pairs = pd.read_csv(
            PROCESSED_SUPPORT_PAIRS_PATH,
            dtype={"customer_tweet_id": str, "agent_tweet_id": str},
        )

    required = {
        "customer_tweet_id",
        "agent_tweet_id",
        "customer_text",
        "agent_text",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"support pairs missing columns: {missing}")

    df = pairs.reset_index(drop=True).copy()
    texts = df["customer_text"].astype(str)
    match_matrix = {intent: texts.map(lambda t, p=_PATTERNS[intent]: bool(p.search(t))) for intent in INTENT_TAXONOMY}
    match_df = pd.DataFrame(match_matrix)
    n_matched = match_df.sum(axis=1)
    multi_mask = n_matched >= 2
    vague_mask = match_df["other_or_unclear"] & (n_matched <= 1)

    rng = np.random.default_rng(seed)
    used: set[int] = set()
    selected: list[dict[str, Any]] = []

    def add_row(idx: int, primary: str, sampling_tag: str) -> None:
        row = df.iloc[idx]
        also = matched_intents(str(row["customer_text"]))
        selected.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "agent_tweet_id": row["agent_tweet_id"],
                "customer_text": row["customer_text"],
                "agent_text": row["agent_text"],
                "proposed_intent": format_proposed_intent(primary, also),
                "annotator_label": "",
                "annotator_notes": "",
                "_sampling_tag": sampling_tag,
                "_n_matched": int(n_matched.iloc[idx]),
                "_is_multi": bool(multi_mask.iloc[idx]),
                "_is_vague_candidate": bool(
                    primary == "other_or_unclear" or vague_mask.iloc[idx]
                ),
            }
        )

    # 1) Per-intent stratified core (favor exclusive matches when possible).
    for intent in INTENT_TAXONOMY:
        exclusive = np.flatnonzero(match_df[intent] & (n_matched == 1))
        inclusive = np.flatnonzero(match_df[intent])
        got = _sample_indices(exclusive, per_intent, rng, used)
        if len(got) < per_intent:
            more = _sample_indices(inclusive, per_intent - len(got), rng, used)
            got.extend(more)
        for idx in got:
            add_row(idx, intent, sampling_tag=f"stratified:{intent}")

    # 2) Boundary / difficult examples.
    boundary_quota = 3
    for name, a, b in _BOUNDARIES:
        pool = np.flatnonzero(match_df[a] & match_df[b])
        for idx in _sample_indices(pool, boundary_quota, rng, used):
            # Primary suggestion: first pattern name as a weak hint only.
            primary = a
            add_row(idx, primary, sampling_tag=f"boundary:{name}")

    # connectivity vs SIM
    sim_pool = np.flatnonzero(texts.map(lambda t: bool(_SIM_PATTERN.search(t))))
    for idx in _sample_indices(sim_pool, 4, rng, used):
        add_row(idx, "network_connectivity", sampling_tag="boundary:connectivity_vs_sim")

    # account lock vs password/reset
    lock_only = np.flatnonzero(
        texts.map(lambda t: bool(_LOCK_PATTERN.search(t)))
        & ~texts.map(lambda t: bool(_PASSWORD_PATTERN.search(t)))
    )
    pass_only = np.flatnonzero(
        texts.map(lambda t: bool(_PASSWORD_PATTERN.search(t)))
        & ~texts.map(lambda t: bool(_LOCK_PATTERN.search(t)))
    )
    for idx in _sample_indices(lock_only, 3, rng, used):
        add_row(idx, "account_access", sampling_tag="boundary:account_lock")
    for idx in _sample_indices(pass_only, 3, rng, used):
        add_row(idx, "account_access", sampling_tag="boundary:password_reset")

    # vague vs actionable
    for idx in _sample_indices(np.flatnonzero(vague_mask), 8, rng, used):
        add_row(idx, "other_or_unclear", sampling_tag="vague_or_unclear")

    # multi-problem
    for idx in _sample_indices(np.flatnonzero(multi_mask), 12, rng, used):
        also = matched_intents(str(df.iloc[idx]["customer_text"]))
        primary = also[0] if also else "other_or_unclear"
        add_row(idx, primary, sampling_tag="multi_problem")

    # Top up to target_size with remaining multi / residual other.
    if len(selected) < target_size:
        residual = np.flatnonzero(n_matched == 0)
        need = target_size - len(selected)
        for idx in _sample_indices(residual, need, rng, used):
            add_row(idx, "other_or_unclear", sampling_tag="residual_unmatched")

    # If still short, random fill from unused.
    if len(selected) < target_size:
        all_idx = np.arange(len(df))
        need = target_size - len(selected)
        for idx in _sample_indices(all_idx, need, rng, used):
            also = matched_intents(str(df.iloc[idx]["customer_text"]))
            primary = also[0] if also else "other_or_unclear"
            add_row(idx, primary, sampling_tag="random_fill")

    out = pd.DataFrame(selected)
    # Cap near target while keeping all boundary/multi if slightly over — trim random_fill first.
    if len(out) > target_size + 20:
        # keep structured samples; drop excess residual/random
        priority = out["_sampling_tag"].map(
            lambda t: 0
            if str(t).startswith(("boundary", "multi", "vague", "stratified"))
            else 1
        )
        out = out.assign(_pri=priority).sort_values(["_pri"]).head(target_size).drop(columns=["_pri"])

    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.insert(0, "example_id", [f"GS-{i:04d}" for i in range(1, len(out) + 1)])

    # Extract primary proposed intent for reporting (still not ground truth).
    def _primary(s: str) -> str:
        m = re.match(r"MACHINE_CANDIDATE:\s*([a-z0-9_]+)", str(s))
        return m.group(1) if m else "other_or_unclear"

    out["_proposed_primary"] = out["proposed_intent"].map(_primary)

    summary = {
        "n_examples": int(len(out)),
        "per_proposed_intent": out["_proposed_primary"].value_counts().to_dict(),
        "n_ambiguous_multi": int(out["_is_multi"].sum()),
        "n_vague_unclear_candidates": int(out["_is_vague_candidate"].sum()),
        "n_boundary_tagged": int(
            out["_sampling_tag"].astype(str).str.startswith("boundary").sum()
        ),
        "sampling_tag_counts": out["_sampling_tag"].value_counts().to_dict(),
    }

    export_cols = [
        "example_id",
        "customer_tweet_id",
        "agent_tweet_id",
        "customer_text",
        "agent_text",
        "proposed_intent",
        "annotator_label",
        "annotator_notes",
    ]
    return out[export_cols + ["_sampling_tag", "_is_multi", "_is_vague_candidate", "_proposed_primary"]], summary


def export_golden_set_candidates(
    out_path,
    pairs_path=None,
    target_size: int = 200,
    per_intent: int = 14,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build candidates and write the annotation CSV (without helper columns)."""
    from pathlib import Path

    pairs = None
    if pairs_path is not None:
        pairs = pd.read_csv(
            pairs_path,
            dtype={"customer_tweet_id": str, "agent_tweet_id": str},
        )
    full, summary = build_golden_set_candidates(
        pairs=pairs,
        target_size=target_size,
        per_intent=per_intent,
        seed=seed,
    )
    export_cols = [
        "example_id",
        "customer_tweet_id",
        "agent_tweet_id",
        "customer_text",
        "agent_text",
        "proposed_intent",
        "annotator_label",
        "annotator_notes",
    ]
    export = full[export_cols].copy()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(path, index=False)
    summary["out_path"] = str(path)
    return export, summary
