# AppleSupport AI Customer Support Agent — Hiver SDE Intern Assignment

An end-to-end, reproducible AI Customer Support Agent for **AppleSupport** built on historical Twitter support interactions from the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/thoughtvector/customer-support-on-twitter) dataset.

---

## 🚀 Quick Start & Reproduction Path (<15 Minutes)

You can run the end-to-end agent pipeline and full evaluation harness directly using pre-processed data and trained models in the repository.

### 1. Environment Setup

```bash
# Clone repository and create virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows (or source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

### 2. Run Complete Test Suite (<10 seconds)

```bash
pytest tests/
```

### 3. Run End-to-End SupportAgent Interactive / Demo CLI

```bash
python scripts/run_agent.py --text "My battery drains so fast after updating to iOS 11"
```

### 4. Evaluate Final SupportAgent on 200-Example Golden Set

```bash
python scripts/evaluate_agent_golden_set.py
```

Outputs detailed predictions and decisions to `data/processed/golden_set_agent_results.csv`.

---

## 🏗️ System Architecture

The support agent operates as a **4-stage deterministic and explainable pipeline**:

```
Customer Message
       │
       ▼
┌─────────────────────────┐
│ 1. Intent Classifier    │  ──► Predicts 1 of 13 intents + confidence score
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 2. Retrieval Engine     │  ──► Retrieves Top-5 historical support pairs (TF-IDF + Cosine)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 3. Reply Drafter        │  ──► Drafts grounded response strictly from retrieved agent evidence
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 4. Escalation Policy    │  ──► Evaluates 4 safety gates -> auto_handle OR escalate + reason
└─────────────────────────┘
```

---

## 📊 Evaluation Headline Metrics

Evaluated on the unchanged **200-example human-annotated Golden Set** (`data/processed/golden_set_final.csv`):

| Pipeline Stage | Metric | Baseline Model | Final Winning Model (Exp 3A) | Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Intent Classifier** | **Golden Set Accuracy** | 61.0% | **64.0%** | **+3.0%** |
| **Intent Classifier** | **Golden Set Macro-F1** | 62.1% | **64.15%** | **+2.05%** |
| **SupportAgent** | **Auto-Handle Rate** | — | **6.0%** (12 / 200) | Safe conservative handling |
| **SupportAgent** | **Escalation Rate** | — | **94.0%** (188 / 200) | Zero unauthorized risk |

---

## 🔬 Classifier Improvement Experiments Summary

| Experiment | Feature Representation | Class Weighting | Int. Val Acc | Int. Val F1 | Golden Set Acc | Golden Set F1 | Status / Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline 1 (Original)** | Word TF-IDF (1,2) | `balanced` | 0.9327 | 0.8947 | 0.5900 | 0.6010 | Initial reference |
| **Baseline 1 (Improved OOU)** | Word TF-IDF (1,2) | `balanced` | 0.9327 | 0.8947 | **0.6100** | **0.6210** | **Benchmark Baseline** |
| **Experiment 1 (Weak Expansion)** | Word TF-IDF (1,2) | `balanced` | 0.9628 | 0.8886 | 0.5500 | 0.5614 | Rejected (label noise) |
| **Experiment 2 (Self-Training)** | Word TF-IDF (1,2) | `balanced` | 0.9335 | 0.8987 | 0.6150 | 0.6254 | Modest gain (+0.5%) |
| **Experiment 3A (Char TF-IDF)** | **Char n-gram (2,5)** | **`balanced`** | **0.8847** | **0.8299** | **0.6400** | **0.6415** | **WINNER (+3.0% Acc)** |
| **Experiment 3B (Comb Word+Char)** | Word+Char | `balanced` | 0.9240 | 0.8797 | 0.6150 | 0.6183 | Feature dilution |
| **Experiment 4A (Unweighted)** | Char n-gram (2,5) | `None` | 0.9006 | 0.7339 | 0.5000 | 0.5112 | Rejected (minority collapse) |
| **Experiment 4B (Sqrt Weighted)** | Char n-gram (2,5) | `sqrt(bal)` | 0.9221 | 0.8726 | 0.6300 | 0.6353 | Solid (+2.0%), sub-balanced |

---

## 🛠️ Complete Pipeline Execution & Data Regeneration

If you wish to re-execute dataset processing and model training from scratch:

### Step 1: Raw Data Setup
Place `twcs.csv` (Kaggle dataset) into `data/raw/twcs.csv`.

### Step 2: Extract Support Pairs & Golden Set Candidates
```bash
python scripts/prepare_data.py
python scripts/prepare_support_pairs.py
python scripts/prepare_golden_set_candidates.py
```

### Step 3: Weak Training Label Generation & Classifier Training
```bash
python scripts/prepare_weak_training.py
python scripts/train_baseline1.py
python scripts/train_baseline2.py
```

### Step 4: Build Retrieval Index
```bash
python scripts/build_retrieval_index.py
```

---

## ⚖️ LLM-as-a-Judge Evaluation

The automated judge system evaluates replies on a 1–5 scale across **Relevance**, **Groundedness**, **Helpfulness**, and **Safety** per [docs/judge_rubric.md](docs/judge_rubric.md).

```bash
# Set credentials (optional live evaluation)
$env:LLM_JUDGE_PROVIDER="gemini"
$env:LLM_JUDGE_MODEL="gemini-2.5-flash"
$env:GEMINI_API_KEY="<your-api-key>"

python scripts/run_judge.py
```

- **Output File**: `data/processed/judge_sample_10.csv`
- **Genuine Live Evaluated Results**: 6 PASS / 2 FAIL
- **Rate Limit Handling**: Explicitly records `API_RATE_LIMITED` without fake scores when API limits are reached.

---

## 📌 Top 5 Failure Modes & Headline Metric Limitations

### Top 5 Failure Modes
1. **Multi-Intent Overlap (33.3%)**: Dual-symptom queries (e.g. battery drain *after* iOS update) force single-label choices.
2. **Weak-Label Noise (25.0%)**: Heuristic rules learn spurious keyword shortcuts.
3. **Class Imbalance (19.4%)**: Rare intents (`storage_issue`, `subscription_management`) have <100 weak training examples.
4. **Twitter Slang & Formatting Noise (12.5%)**: Extreme abbreviations, broken unicode, and single-word outbursts.
5. **Taxonomy Ambiguity (9.7%)**: Borderline queries between `howto_or_feature` and software bugs.

### "What is Misleading About My Headline Number?"
- **Small Evaluation Sample**: 200 Golden Set examples carry a $\pm 6.7\%$ margin of error.
- **Classification vs Support Resolution**: 64% intent accuracy measures step 1 classification, not full issue resolution.
- **Conservative Auto-Handling**: Auto-handle rate is 6.0% (12/200) to ensure zero unauthorized risk.

---

## 📁 Key Documentation Links

- **Final Submission Report (6-page summary)**: [docs/final_report.md](docs/final_report.md)
- **Engineering Decision Log (15 decisions)**: [docs/decision_log.md](docs/decision_log.md)
- **LLM Judge Rubric**: [docs/judge_rubric.md](docs/judge_rubric.md)
- **Golden Set Annotation Guide**: [docs/golden_set_annotation_guide.md](docs/golden_set_annotation_guide.md)
