"""
Evaluate end-to-end SupportAgent on the 200-example Golden Set.

Saves detailed predictions and decisions to:
  data/processed/golden_set_agent_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"c:\Users\LENOVO\OneDrive\Desktop\hiver-sde-assignment")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.agent import SupportAgent
from src.config import GOLDEN_SET_FINAL_PATH, INTENT_TAXONOMY
from src.evaluate import load_golden_set

report_lines = []

def log(msg=""):
    report_lines.append(str(msg))

def main() -> None:
    golden_df = load_golden_set(GOLDEN_SET_FINAL_PATH)
    agent = SupportAgent()

    results = []
    for idx, row in golden_df.iterrows():
        res = agent.process_message(row["customer_text"])
        top_sim = (
            float(res["retrieved_evidence"][0]["similarity_score"])
            if res["retrieved_evidence"]
            else 0.0
        )
        results.append(
            {
                "example_id": row["example_id"],
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": row["customer_text"],
                "annotator_label": row["annotator_label"],
                "predicted_intent": res["predicted_intent"],
                "classifier_confidence": res["classifier_confidence"],
                "top_retrieval_similarity": top_sim,
                "reply_status": res["reply_status"],
                "reply_text": res["reply_text"],
                "escalation_decision": res["escalation_decision"],
                "escalation_reason_code": res["escalation_reason_code"],
                "escalation_reason": res["escalation_reason"],
            }
        )

    out_df = pd.DataFrame(results)
    out_path = ROOT / "data" / "processed" / "golden_set_agent_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    y_true = out_df["annotator_label"].tolist()
    y_pred = out_df["predicted_intent"].tolist()

    labels = [lab for lab in INTENT_TAXONOMY if lab in set(y_true) | set(y_pred)]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)

    auto_count = int((out_df["escalation_decision"] == "auto_handle").sum())
    esc_count = int((out_df["escalation_decision"] == "escalate").sum())

    reason_counts = out_df["escalation_reason_code"].value_counts().to_dict()

    # Escalation rate by predicted intent
    intent_esc = []
    for intent, group in out_df.groupby("predicted_intent"):
        total = len(group)
        e_cnt = (group["escalation_decision"] == "escalate").sum()
        rate = (e_cnt / total) * 100 if total > 0 else 0.0
        intent_esc.append(
            {
                "predicted_intent": intent,
                "total": total,
                "escalate_count": e_cnt,
                "escalation_rate_pct": rate,
            }
        )
    intent_esc_df = pd.DataFrame(intent_esc)

    log("=" * 60)
    log("GOLDEN SET SUPPORT AGENT EVALUATION REPORT")
    log("=" * 60)
    log(f"Total Golden Set Rows: {len(out_df)}")
    log(f"Saved Output File:     {out_path}")
    log(f"Intent Accuracy:       {acc:.4f}")
    log(f"Intent Macro-F1:       {macro_f1:.4f}")
    log(f"Auto-Handle Count:     {auto_count} ({(auto_count/len(out_df))*100:.1f}%)")
    log(f"Escalate Count:        {esc_count} ({(esc_count/len(out_df))*100:.1f}%)")
    log("-" * 60)
    log("Escalation Reason Code Counts:")
    for code, count in reason_counts.items():
        log(f"  {code:<40}: {count}")

    log("-" * 60)
    log("Escalation Rate by Predicted Intent:")
    log(f"{'Predicted Intent':<30} | {'Total':<6} | {'Escalated':<10} | {'Rate (%)':<8}")
    log("-" * 60)
    for _, r in intent_esc_df.iterrows():
        log(
            f"{r['predicted_intent']:<30} | {r['total']:<6} | {r['escalate_count']:<10} | {r['escalation_rate_pct']:<8.1f}"
        )

    log("\n10 Lowest-Confidence Examples:")
    lowest_conf = out_df.sort_values("classifier_confidence").head(10)
    for idx, r in lowest_conf.iterrows():
        log(
            f"  [Conf: {r['classifier_confidence']:.4f}] ID={r['example_id']} | True: {r['annotator_label']} | Pred: {r['predicted_intent']} | Decision: {r['escalation_decision']} ({r['escalation_reason_code']})"
        )
        log(f"    Text: {r['customer_text'][:100]}")

    log("\n10 Lowest-Retrieval-Similarity Examples:")
    lowest_sim = out_df.sort_values("top_retrieval_similarity").head(10)
    for idx, r in lowest_sim.iterrows():
        log(
            f"  [Sim: {r['top_retrieval_similarity']:.4f}] ID={r['example_id']} | True: {r['annotator_label']} | Pred: {r['predicted_intent']} | Decision: {r['escalation_decision']} ({r['escalation_reason_code']})"
        )
        log(f"    Text: {r['customer_text'][:100]}")

    out_file = Path(r"C:\Users\LENOVO\.gemini\antigravity\brain\d14208ce-d0db-428c-8c92-4446b1d22e75\scratch\agent_golden_set_report.txt")
    out_file.write_text("\n".join(report_lines), encoding="utf-8")
    print("Wrote agent golden set report to", out_file)


if __name__ == "__main__":
    main()
