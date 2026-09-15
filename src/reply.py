"""
Grounded reply drafting from retrieved historical AppleSupport responses.

Deterministic / template-based — no external LLM.
Drafts must stay faithful to retrieved agent_text evidence only.
"""

from __future__ import annotations

import re
from typing import Any

# Patterns that indicate a DM / private-channel redirect in historical replies.
_DM_RE = re.compile(
    r"\b(dm\b|direct\s*message|send\s*us\s*a\s*dm|join\s*us\s*in\s*a\s*dm|"
    r"meet\s*us\s*in\s*(a\s*)?dm|more\s*room)\b",
    re.I,
)
_LINK_RE = re.compile(r"https?://\S+", re.I)
_QUESTION_RE = re.compile(
    r"\b(which\s+(iphone|ipad|device|mac|version|ios)|what\s+(iphone|ipad|device|version|ios)|"
    r"ios\s*version|software\s*version|what\s+country|when\s+did|"
    r"have\s+you\s+tried|did\s+you|can\s+you\s+tell)\b",
    re.I,
)
# Instruction-ish cues commonly present in AppleSupport tweets.
_INSTRUCTION_RE = re.compile(
    r"\b(go\s+to|open\s+settings|settings\s*>|tap|select|follow\s+these\s+steps|"
    r"try\s+these|restart|reset|update\s+to|check\s+(from|here|out)|"
    r"back\s+up|force\s*quit|forget\s+the\s+network|slide\s+to)\b",
    re.I,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Soft boilerplate to strip so we do not parrot full historical tweets.
_BOILERPLATE_RE = re.compile(
    r"^\s*("
    r"we('?re| are)\s+(here|happy|glad|sorry).*?\.|"
    r"thanks?\s+for\s+reaching\s+out.*?\.|"
    r"we('?d| would)\s+(like|love|be\s+happy)\s+to\s+help.*?\.|"
    r"hello[!.,]?\s*|hi\s+there[!.,]?\s*"
    r")",
    re.I,
)


def _normalize_example(ex: dict[str, Any], idx: int) -> dict[str, Any]:
    """Normalize one retrieved example dict."""
    agent = str(ex.get("agent_text") or "").strip()
    cust_id = str(ex.get("customer_tweet_id") or f"unknown_customer_{idx}")
    agent_id = str(ex.get("agent_tweet_id") or f"unknown_agent_{idx}")
    return {
        "customer_tweet_id": cust_id,
        "agent_tweet_id": agent_id,
        "customer_text": str(ex.get("customer_text") or ""),
        "agent_text": agent,
        "similarity_score": float(ex.get("similarity_score") or 0.0),
        "has_dm": bool(_DM_RE.search(agent)),
        "has_link": bool(_LINK_RE.search(agent)),
        "has_question": bool(_QUESTION_RE.search(agent)),
        "has_instruction": bool(_INSTRUCTION_RE.search(agent)),
    }


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts


def _extract_instruction_snippets(agent_text: str, max_len: int = 160) -> list[str]:
    """Pull short instruction-bearing sentences; never return the full tweet as one blob."""
    snippets: list[str] = []
    for sent in _split_sentences(agent_text):
        cleaned = _BOILERPLATE_RE.sub("", sent).strip(" -–")
        if not cleaned or len(cleaned) < 12:
            continue
        if not _INSTRUCTION_RE.search(cleaned):
            continue
        # Drop pure DM sentences even if they mention "follow".
        if _DM_RE.search(cleaned) and not re.search(
            r"settings|restart|reset|tap|backup|force", cleaned, re.I
        ):
            continue
        if len(cleaned) > max_len:
            cleaned = cleaned[: max_len - 1].rstrip() + "…"
        if cleaned.lower() not in {s.lower() for s in snippets}:
            snippets.append(cleaned)
    return snippets


def _extract_clarifying_questions(agent_text: str, max_len: int = 140) -> list[str]:
    questions: list[str] = []
    for sent in _split_sentences(agent_text):
        if "?" not in sent and not _QUESTION_RE.search(sent):
            continue
        cleaned = _BOILERPLATE_RE.sub("", sent).strip()
        if not cleaned:
            continue
        if _DM_RE.search(cleaned) and "which" not in cleaned.lower() and "what" not in cleaned.lower():
            continue
        if len(cleaned) > max_len:
            cleaned = cleaned[: max_len - 1].rstrip() + "…"
        if cleaned.lower() not in {q.lower() for q in questions}:
            questions.append(cleaned)
    return questions


def _intent_phrase(predicted_intent: str) -> str:
    return str(predicted_intent or "your issue").replace("_", " ")


def _is_mostly_dm_redirects(examples: list[dict[str, Any]]) -> bool:
    """True when historical replies are mainly DM/link redirects without instructions."""
    if not examples:
        return True
    dm_like = 0
    instructive = 0
    for ex in examples:
        if ex["has_instruction"]:
            instructive += 1
        elif ex["has_dm"] or (ex["has_link"] and not ex["has_instruction"]):
            dm_like += 1
    if instructive == 0:
        return True
    return dm_like >= instructive and instructive <= max(1, len(examples) // 3)


def draft_reply(
    customer_text: str,
    predicted_intent: str,
    retrieved_examples: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """
    Draft a concise support reply grounded only in retrieved historical agent_text.

    Returns:
      reply_text: str
      evidence: list of {customer_tweet_id, agent_tweet_id, similarity_score?}
      groundedness_note: str
      draft_style: "instructional" | "dm_redirect" | "insufficient_evidence"
    """
    customer_text = str(customer_text or "").strip()
    predicted_intent = str(predicted_intent or "other_or_unclear").strip()
    raw_examples = list(retrieved_examples or [])

    examples = [_normalize_example(ex, i) for i, ex in enumerate(raw_examples)]
    examples = [ex for ex in examples if ex["agent_text"]]

    evidence = [
        {
            "customer_tweet_id": ex["customer_tweet_id"],
            "agent_tweet_id": ex["agent_tweet_id"],
            "similarity_score": ex["similarity_score"],
        }
        for ex in examples
    ]

    groundedness_note = (
        "This draft is template-based and grounded only in retrieved historical "
        "AppleSupport responses. It does not invent troubleshooting steps beyond "
        "that evidence, and it is not an LLM-generated answer."
    )

    if not examples:
        reply = (
            f"Thanks for reaching out about {_intent_phrase(predicted_intent)}. "
            "I do not have enough similar historical cases to suggest specific steps yet. "
            "Please share your device model and software version, or continue with a support specialist."
        )
        return {
            "reply_text": reply,
            "evidence": [],
            "groundedness_note": groundedness_note,
            "draft_style": "insufficient_evidence",
        }

    # Collect reusable fragments from evidence only.
    instruction_snippets: list[str] = []
    clarifying: list[str] = []
    for ex in examples:
        for snip in _extract_instruction_snippets(ex["agent_text"]):
            if snip.lower() not in {s.lower() for s in instruction_snippets}:
                instruction_snippets.append(snip)
        for q in _extract_clarifying_questions(ex["agent_text"]):
            if q.lower() not in {c.lower() for c in clarifying}:
                clarifying.append(q)

    used_ids = {
        (ex["customer_tweet_id"], ex["agent_tweet_id"]) for ex in examples
    }
    # Prefer evidence that actually contributed snippets when available.
    contributing = []
    for ex in examples:
        agent = ex["agent_text"]
        contributed = bool(
            _extract_instruction_snippets(agent)
            or _extract_clarifying_questions(agent)
            or ex["has_dm"]
        )
        if contributed:
            contributing.append(
                {
                    "customer_tweet_id": ex["customer_tweet_id"],
                    "agent_tweet_id": ex["agent_tweet_id"],
                    "similarity_score": ex["similarity_score"],
                }
            )
    if contributing:
        evidence = contributing
    else:
        # Fall back to all retrieved IDs if nothing parsed (still grounded on that set).
        evidence = [
            {
                "customer_tweet_id": ex["customer_tweet_id"],
                "agent_tweet_id": ex["agent_tweet_id"],
                "similarity_score": ex["similarity_score"],
            }
            for ex in examples
        ]
    _ = used_ids

    intent_bit = _intent_phrase(predicted_intent)

    if _is_mostly_dm_redirects(examples) and not instruction_snippets:
        # Transparent DM / continue-in-private draft — do not pretend the issue is solved.
        ask = "Could you share your device model and software version"
        if clarifying:
            # Rephrase first clarifying ask without pasting the whole historical tweet.
            first_q = clarifying[0].rstrip("?")
            if len(first_q) > 100:
                ask = "Could you share a bit more detail about what you are seeing"
            else:
                ask = first_q if first_q.lower().startswith(
                    ("which", "what", "when", "have", "did", "can", "could")
                ) else ask
        reply = (
            f"Thanks for writing in about {intent_bit}. "
            "In similar past cases, Apple Support usually continued privately over DM "
            "rather than finishing troubleshooting in public replies. "
            f"{ask}? "
            "Once we have that, a specialist can take the next step with you."
        )
        return {
            "reply_text": reply,
            "evidence": evidence,
            "groundedness_note": groundedness_note,
            "draft_style": "dm_redirect",
        }

    # Instructional draft grounded in extracted historical steps only.
    lines = [
        f"Thanks for reaching out about {intent_bit}.",
        "Based on how similar historical cases were handled, here is what support typically suggested next:",
    ]
    for snip in instruction_snippets[:3]:
        # Present as cited guidance, not a verbatim full reply copy.
        lines.append(f"- {snip}")

    if clarifying and len(instruction_snippets) < 2:
        q = clarifying[0].rstrip("?")
        if len(q) <= 120:
            lines.append(f"Also helpful to confirm: {q}?")

    if any(ex["has_dm"] for ex in examples):
        lines.append(
            "If this still does not resolve it, similar cases were often moved to DM for closer help."
        )

    # Guard: never equal any full historical agent_text.
    reply = "\n".join(lines)
    for ex in examples:
        if reply.strip() == ex["agent_text"].strip():
            reply = (
                f"Thanks for reaching out about {intent_bit}. "
                "Similar historical replies asked for more device details before next steps. "
                "Could you share your device model and software version?"
            )
            break

    return {
        "reply_text": reply,
        "evidence": evidence,
        "groundedness_note": groundedness_note,
        "draft_style": "instructional",
    }
