# Decision Log: AppleSupport Automated Agent & Evaluation Pipeline

This decision log documents 15 key non-obvious engineering decisions made during the design, implementation, and evaluation of the AppleSupport automated support system.

---

### Decision 1: Single-Brand Support Domain Focus (`AppleSupport`)
- **Alternative Considered**: Multi-brand dataset mixing Apple, Amazon, Uber, Spotify, and Delta support tweets.
- **Why Chosen**: AppleSupport represents the largest, cleanest, and most cohesive single brand corpus in TWCS (~106k customer-agent pairs), with consistent technical support terminology and high interaction volume.
- **Trade-off / Consequence**: The trained classifier and retrieval index are domain-specific to Apple products and cannot generalize to other brands without retraining.

---

### Decision 2: Support Pair Construction (Direct Customer-to-Agent Dialogues)
- **Alternative Considered**: Full multi-turn conversation trees with threaded customer replies.
- **Why Chosen**: Single-turn first customer message $\rightarrow$ first agent response captures the initial diagnostic state and intent clearly, preventing downstream dialogue loop noise.
- **Trade-off / Consequence**: Ignores follow-up customer clarifications in multi-turn threads.

---

### Decision 3: 13-Intent Taxonomy Structure
- **Alternative Considered**: Fine-grained 50+ intent taxonomy or coarse 3-intent taxonomy (hardware, software, billing).
- **Why Chosen**: 13 intents strikes the optimal balance between operational clarity for support escalation and distinct feature space for classification.
- **Trade-off / Consequence**: Some overlap remains between `software_update_issue`, `app_issue`, and `battery_drain` after major iOS updates.

---

### Decision 4: Dedicated `other_or_unclear` Intent Class
- **Alternative Considered**: Forcing every query into a technical intent or using a post-hoc probability threshold.
- **Why Chosen**: Real customer tweets frequently contain vague complaints ("fix this please!"), slang, or single-word frustrations. Explicitly modeling `other_or_unclear` allows the classifier to learn vague query patterns and trigger safe escalation.
- **Trade-off / Consequence**: `other_or_unclear` requires strict guard patterns so technical queries containing "fix" are not misclassified.

---

### Decision 5: Weak-Label Training Data Generation
- **Alternative Considered**: Hand-labeling a small 1,000-example training set or using fully unsupervised clustering.
- **Why Chosen**: Rule-based weak labeling leveraged ~13,223 clean historical pairs for training without expensive manual annotation, providing broad coverage across all 13 intents.
- **Trade-off / Consequence**: Introduces rule noise and keyword shortcut learning in the training set.

---

### Decision 6: Golden Set Exclusion Guarantee
- **Alternative Considered**: Including Golden Set candidates in weak-label training data and splitting 80/20.
- **Why Chosen**: Strict isolation of the 200 Golden Set customer tweet IDs prevents data leakage, guaranteeing that evaluation metrics reflect true out-of-sample generalization.
- **Trade-off / Consequence**: Reduces the available weak training set size by 200 rows (negligible impact).

---

### Decision 7: Character n-gram TF-IDF Representation (`char_wb`, 2–5)
- **Alternative Considered**: Word-level TF-IDF (Baseline 1 default) or Dense Word Embeddings (Word2Vec/GloVe).
- **Why Chosen**: Character subword n-grams (`char_wb`, n-grams 2–5) improved Golden Set accuracy from 61.0% to 64.0% (+3.0%) and Macro-F1 from 62.1% to 64.15% (+2.05%) by robustly capturing Twitter typos, device names (`iphone7`, `ios11`), hashtags, and noisy abbreviations.
- **Trade-off / Consequence**: Increases feature matrix vocabulary size (~15,000 features).

---

### Decision 8: Balanced Class Weighting (`class_weight='balanced'`)
- **Alternative Considered**: Unweighted Logistic Regression (`class_weight=None`) or aggressive oversampling (SMOTE).
- **Why Chosen**: Unweighted training collapsed minority intent recall (50.0% Golden Set accuracy). Balanced weighting in Logistic Regression penalizes minority class errors inversely proportional to class frequencies, preserving high Macro-F1 across all 13 intents.
- **Trade-off / Consequence**: Slightly increases false positive rates for rare intents.

---

### Decision 9: Rejection of Broad Weak-Label Expansion (Experiment 1)
- **Alternative Considered**: Adding loose single-word regex rules to label 21,894 training pairs (+8,671 examples).
- **Why Chosen**: Broad keywords like "update" or "ios" flooded the training set with 17,008 `software_update_issue` labels (77.7% of all data), degrading Golden Set accuracy from 61.0% to 55.0% (-6.0%).
- **Trade-off / Consequence**: Kept training set size smaller (13,223 rows) to preserve high label precision.

---

### Decision 10: High-Confidence Self-Training (Experiment 2, $\tau=0.95$)
- **Alternative Considered**: Unfiltered pseudo-labeling on all 90k unlabeled support pairs.
- **Why Chosen**: Filtering pseudo-labels with a strict confidence threshold ($\tau=0.95$) selected on the internal validation split added 142 clean pseudo-labels, improving Golden Set accuracy to 61.5% without introducing label noise.
- **Trade-off / Consequence**: Only 142 out of 93,223 unlabeled pairs met the strict $\tau=0.95$ threshold.

---

### Decision 11: Lexical TF-IDF Cosine Retrieval Index
- **Alternative Considered**: Fine-tuned Dense Semantic Embeddings (SBERT / Sentence-Transformers).
- **Why Chosen**: TF-IDF cosine retrieval over 106k historical support pairs executes in <5ms without GPU dependencies, providing exact keyword matching for specific Apple error codes and device names.
- **Trade-off / Consequence**: Fails on deep semantic queries where customer text shares no lexical terms with historical agent responses.

---

### Decision 12: Deterministic Grounded Reply Generation
- **Alternative Considered**: Unconstrained LLM text generation (GPT-4 / Claude).
- **Why Chosen**: Deterministic grounding directly extracts verified historical agent steps and DM handoff patterns from top retrieved evidence, eliminating hallucination risks, unauthorized financial promises, and security violations.
- **Trade-off / Consequence**: Replies remain structured and conservative rather than highly conversational.

---

### Decision 13: Four-Layer Conservative Escalation Policy
- **Alternative Considered**: Auto-handling all queries where classifier confidence $>0.50$.
- **Why Chosen**: Requiring all four safety gates (supported intent, classifier confidence $\ge 0.65$, top retrieval similarity $\ge 0.30$, and grounded reply status) ensures zero high-risk automated responses on ambiguous or severe complaints.
- **Trade-off / Consequence**: Caps auto-handle rate at 6.0% (12 out of 200), escalating 94.0% (188 out of 200) to human support agents.

---

### Decision 14: DM-Redirect Handoff Recognition in Evaluation
- **Alternative Considered**: Treating DM-redirect agent replies as "unhelpful" or "empty" answers.
- **Why Chosen**: In historical AppleSupport tweets, 68%+ of agent responses direct customers to private DMs for account privacy and diagnostic logs. Evaluating DM redirects as valid handoffs aligns with real-world enterprise support protocols.
- **Trade-off / Consequence**: DM-redirect replies yield safe handoffs but do not resolve issues publicly.

---

### Decision 15: Automated LLM-as-a-Judge Evaluation Rubric
- **Alternative Considered**: Relying solely on lexical overlap metrics (BLEU / ROUGE).
- **Why Chosen**: BLEU/ROUGE punish valid rephrasing and handoff replies. The LLM judge evaluates Relevance, Groundedness, Helpfulness, and Safety on a 1–5 scale using multi-dimensional prompt criteria and evidence citation verification.
- **Trade-off / Consequence**: API rate limits on free-tier LLM providers require graceful handling and explicit reporting of non-judged entries.
