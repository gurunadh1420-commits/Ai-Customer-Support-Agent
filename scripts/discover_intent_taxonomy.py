"""
Temporary taxonomy-discovery sampling (NOT labeling, NOT training).

Proposes candidate intent buckets with stricter separation than the earlier
exploratory themes. Counts are approximate keyword candidates only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import PROCESSED_SUPPORT_PAIRS_PATH

# Stricter exploratory candidate patterns for taxonomy discovery.
CANDIDATES: dict[str, re.Pattern[str]] = {
    "battery_drain": re.compile(
        r"\b(battery\s*life|batter(?:y|ies)\s*(drain|draining|dies|dying|dead|low|issue|problem)|"
        r"drain(?:ing)?\s*(my\s*)?battery|killing\s*(my\s*)?battery|battery\s*performance|"
        r"runs?\s*down|dies?\s*fast|battery\s*%|percent\s*battery)\b",
        re.I,
    ),
    "charging_power": re.compile(
        r"\b(charg(?:e|ing|er)|won'?t\s*charge|not\s*charging|slow\s*charg|"
        r"lightning\s*cable|wireless\s*charg|plug(?:ged)?\s*in|usb[- ]?c\s*charg)\b",
        re.I,
    ),
    "ios_update_regression": re.compile(
        r"\b((ios|ipados|macos|watchos)\s*\d*|ios11|high\s*sierra|"
        r"after\s*(the\s*)?(update|upgrading|upgrade)|since\s*(the\s*)?(update|upgraded|updating)|"
        r"new\s*update|latest\s*update|software\s*update|updated\s*(my\s*)?(phone|iphone|ipad|mac))\b",
        re.I,
    ),
    "device_hardware_fault": re.compile(
        r"\b(touch\s*bar|screen\s*(crack|black|blank|flicker|freeze|frozen|unresponsive)|"
        r"display\s*(issue|problem|black)|speaker|microphone|camera\s*(not|won|'t|issue|problem)|"
        r"home\s*button|power\s*button|overheat|swollen|hardware|"
        r"won'?t\s*turn\s*on|boot\s*loop|frozen\s*screen)\b",
        re.I,
    ),
    "wifi_cellular_connectivity": re.compile(
        r"\b(wi[- ]?fi|wifi|bluetooth|cellular|lte|5g|4g|hotspot|airdrop|"
        r"no\s*internet|can'?t\s*connect|connection\s*(issue|problem|drop)|"
        r"network\s*(issue|problem)|keeps\s*disconnect)\b",
        re.I,
    ),
    "sim_carrier_service": re.compile(
        r"\b(sim\s*card|no\s*sim|invalid\s*sim|no\s*service|searching\s*for\s*network|"
        r"carrier\s*(settings|update)|mobile\s*data\s*(not|won)|signal\s*bar)\b",
        re.I,
    ),
    "apple_id_lock_recovery": re.compile(
        r"\b((apple\s*id|icloud|account)\s*(locked|hacked|stolen|disabled)|"
        r"locked\s*out|can'?t\s*(sign|log)\s*in|two[- ]factor|2fa|"
        r"verification\s*code|confirmation\s*code|trusted\s*phone|"
        r"account\s*recovery|security\s*lockout)\b",
        re.I,
    ),
    "password_reset_signin": re.compile(
        r"\b(reset\s*(my\s*)?(password|apple\s*id)|forgot\s*(my\s*)?password|"
        r"change\s*(my\s*)?password|password\s*(not\s*working|incorrect|wrong)|"
        r"sign[- ]?in\s*(issue|problem|page)|login\s*(issue|problem))\b",
        re.I,
    ),
    "app_store_purchase_download": re.compile(
        r"\b(app\s*store|download\s*(app|apps|from)|purchase\s*(app|apps)|"
        r"can'?t\s*download|unable\s*to\s*download|update\s*apps|"
        r"app\s*store\s*(not|won|crash|connect|loading))\b",
        re.I,
    ),
    "specific_app_bug": re.compile(
        r"\b(safari|messages|imessage|mail|photos|facetime|itunes|music\s*app|"
        r"podcast|notes|calendar|maps|camera\s*app|watch\s*app|"
        r"(app|apps)\s*(crash|crashes|crashing|freez|not\s*working|won'?t\s*open))\b",
        re.I,
    ),
    "payment_billing_refund": re.compile(
        r"\b(refund|charged\s*(me|twice|again)|unauthorized\s*charge|"
        r"payment\s*(method|declined|rejected|failed|issue)|billing|"
        r"want\s*(my\s*)?money\s*back|receipt|invoice)\b",
        re.I,
    ),
    "subscription_management": re.compile(
        r"\b(subscription|subscribe|renew(?:al)?|cancel\s*(my\s*)?(subscription|apple\s*music|icloud)|"
        r"manage\s*subscription|family\s*sharing|apple\s*music\s*(family|plan)|"
        r"icloud\+|apple\s*tv\+|apple\s*care|applecare)\b",
        re.I,
    ),
    "storage_full": re.compile(
        r"\b(storage\s*(full|almost\s*full|almost\s*full|space)|not\s*enough\s*storage|"
        r"iphone\s*storage|icloud\s*storage|out\s*of\s*storage|free\s*up\s*space|"
        r"other\s*storage|storage\s*used)\b",
        re.I,
    ),
    "howto_feature_question": re.compile(
        r"\b(how\s*(do|can|to)\s+i|where\s*(do|can)\s+i|is\s*it\s*possible|"
        r"how\s*to\s*(delete|turn|enable|disable|setup|set\s*up|find|connect))\b",
        re.I,
    ),
    "vague_or_unclear": re.compile(
        r"^\s*@?\w*\s*(help|please\s*help|fix\s*(this|it)?|this\s*sucks|"
        r"what('?s|\s+is)\s*wrong|not\s*working|doesn'?t\s*work|issue|problem)\s*[.!?]*\s*$",
        re.I,
    ),
}


def main() -> None:
    df = pd.read_csv(
        PROCESSED_SUPPORT_PAIRS_PATH,
        dtype={"customer_tweet_id": str, "agent_tweet_id": str},
    )
    text = df["customer_text"].astype(str)
    n = len(df)
    print(f"rows={n:,}")

    matches: dict[str, pd.Series] = {}
    for name, pat in CANDIDATES.items():
        matches[name] = text.map(lambda t, p=pat: bool(p.search(t)))
        print(f"{matches[name].sum():7d}  {name}")

    # Separation checks
    print("\n=== OVERLAP CHECKS ===")
    pairs = [
        ("battery_drain", "charging_power"),
        ("ios_update_regression", "device_hardware_fault"),
        ("apple_id_lock_recovery", "password_reset_signin"),
        ("app_store_purchase_download", "specific_app_bug"),
        ("wifi_cellular_connectivity", "sim_carrier_service"),
        ("payment_billing_refund", "subscription_management"),
    ]
    for a, b in pairs:
        both = (matches[a] & matches[b]).sum()
        print(f"overlap {a} ∩ {b}: {both}")

    # Exclusive-ish samples: match A, not B (for similar pairs)
    print("\n=== SEPARATION EXAMPLES ===")

    def show(title: str, mask: pd.Series, k: int = 5) -> None:
        sub = df[mask].copy()
        sub["len"] = sub["customer_text"].str.len()
        sub = sub[(sub["len"] >= 35) & (sub["len"] <= 200)]
        print(f"\n[{title}] n_mask={int(mask.sum()):,}")
        for _, row in sub.head(k).iterrows():
            print(f"- {row['customer_text']}")

    show("battery_drain NOT charging", matches["battery_drain"] & ~matches["charging_power"])
    show("charging NOT battery_drain", matches["charging_power"] & ~matches["battery_drain"])
    show("ios_update NOT hardware", matches["ios_update_regression"] & ~matches["device_hardware_fault"])
    show("hardware NOT ios_update", matches["device_hardware_fault"] & ~matches["ios_update_regression"])
    show("apple_id_lock NOT password_reset", matches["apple_id_lock_recovery"] & ~matches["password_reset_signin"])
    show("password_reset NOT apple_id_lock", matches["password_reset_signin"] & ~matches["apple_id_lock_recovery"])
    show("app_store NOT specific_app", matches["app_store_purchase_download"] & ~matches["specific_app_bug"])
    show("specific_app NOT app_store", matches["specific_app_bug"] & ~matches["app_store_purchase_download"])
    show("wifi/cellular NOT sim", matches["wifi_cellular_connectivity"] & ~matches["sim_carrier_service"])
    show("sim NOT wifi/cellular", matches["sim_carrier_service"] & ~matches["wifi_cellular_connectivity"])
    show("payment NOT subscription", matches["payment_billing_refund"] & ~matches["subscription_management"])
    show("subscription NOT payment", matches["subscription_management"] & ~matches["payment_billing_refund"])
    show("storage_full", matches["storage_full"])
    show("howto_feature_question", matches["howto_feature_question"])
    show("vague_or_unclear", matches["vague_or_unclear"])

    # Multi-candidate
    mat = pd.DataFrame(matches)
    multi = mat.sum(axis=1) >= 2
    print(f"\nmessages matching >=2 candidate patterns: {int(multi.sum()):,}")
    print("\n=== MULTI-CANDIDATE EXAMPLES ===")
    for _, row in df[multi].head(10).iterrows():
        hit = [k for k, s in matches.items() if bool(s.loc[row.name])]
        print(f"- {hit}")
        print(f"  {row['customer_text'][:220]}")


if __name__ == "__main__":
    main()
