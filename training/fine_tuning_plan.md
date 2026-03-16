# Fine-Tuning Plan: Social-Media-Radar Signal Classifier

> Implements Section: LLM Adjudicator Training Strategy  
> Targets: macro F1 ≥ 0.82 | ECE ≤ 0.05 | false-action rate ≤ 0.08

---

## 1. Base Model Selection

| Model | Cost (input/output per 1M tokens) | Latency (p50) | Context | Fine-tune? | Recommended |
|---|---|---|---|---|---|
| **GPT-4o-mini** | $0.15 / $0.60 | ~400ms | 128k | ✅ (OpenAI FT) | **✅ Primary** |
| Claude Haiku 3.5 | $0.25 / $1.25 | ~300ms | 200k | ✅ (Anthropic FT) | ✅ Backup |
| Llama 3.1 8B (local) | ~$0 (compute) | ~800ms | 128k | ✅ (LoRA/QLoRA) | ✅ Offline |
| GPT-4o | $2.50 / $10.00 | ~800ms | 128k | ✅ | ❌ Too expensive |

**Rationale for GPT-4o-mini as primary:**
- OpenAI's fine-tuning API supports JSONL chat-format (matches `signal_classification_dataset.jsonl`)
- 10-30× cheaper than GPT-4o post-fine-tuning on domain-specific tasks
- 128k context handles all social media post lengths
- Fine-tuned mini routinely matches frontier model accuracy on narrow classification tasks
- Llama 3.1 8B as offline fallback eliminates API dependency for air-gapped deployments

---

## 2. Data Preparation Pipeline

### 2.1 Raw Data → Labelled JSONL

```
Raw post text
    → [Preprocessing] strip HTML, normalise Unicode, truncate to 512 chars
    → [Translation] translate non-English to English (DeepL or NLLB-200)
    → [Annotation] dual human annotation using SMR annotation guidelines
    → [Agreement check] Cohen's κ ≥ 0.85 between annotators
    → [Adjudication] senior annotator resolves conflicts below κ threshold
    → [Format] convert to OpenAI JSONL: {"messages": [...], "signal_type": "..."}
```

### 2.2 Annotation Guidelines (summary)

- **Primary rule:** Annotate the *intent behind the post*, not its surface tone.
- **Sarcasm rule:** If the literal text contradicts obvious context, annotate the intended meaning.
- **Indirect intent rule:** Label churn_risk / alternative_seeking even when no product is explicitly named if the buying signal is clear from context.
- **Ambiguity rule:** If two annotators disagree on the primary type, mark as `unclear` unless κ between their top-2 choices ≥ 0.85.
- **Abstention rule:** Any post requiring knowledge of the parent thread to classify must be `unclear` + abstain.
- **Hard negatives:** Include ≥20% spam/noise/unclear examples to train the abstention classifier.

### 2.3 Inter-Annotator Agreement Target

**Cohen's κ ≥ 0.85** (substantial agreement). Measure per-batch before merging. Batches with κ < 0.75 are re-annotated.

---

## 3. Training Hyperparameters

### GPT-4o-mini (OpenAI Fine-Tuning API)

| Hyperparameter | Value | Rationale |
|---|---|---|
| Learning rate multiplier | `0.1` | Conservative; avoids catastrophic forgetting on narrow domain |
| Epochs | `4` | Standard for small datasets; monitor val loss for early stopping |
| Batch size | `8` | OpenAI default; balances gradient noise and memory |
| Train/val split | `80/20` | 62 train / 16 val from the 78-example dataset (grow to 90/10 at ≥500 examples) |
| Early stopping | Val loss plateau for 2 consecutive epochs | Prevents overfitting |
| Seed | `42` | Reproducibility |

### Llama 3.1 8B (LoRA local fine-tuning)

| Hyperparameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | `q_proj, v_proj` |
| Learning rate | `2e-4` |
| Warmup steps | 50 |
| Max sequence length | 512 |
| Quantization | 4-bit QLoRA (bitsandbytes) |

---

## 4. Evaluation Protocol

Run `app/evals/` suite on the held-out validation split after every training epoch:

| Metric | Tool | Target threshold |
|---|---|---|
| Macro F1 | `app/evals/classification_eval.py` | ≥ 0.82 |
| ECE (calibration) | `app/evals/calibration_eval.py` | ≤ 0.05 |
| NDCG@5 (signal ranking) | `app/evals/ranking_eval.py` | ≥ 0.80 |
| False-action rate | `app/evals/adversarial_eval.py` BUILT_IN_CASES | ≤ 0.08 |
| Abstention precision | `app/evals/adversarial_eval.py` | ≥ 0.90 |

**Critical gate:** The fine-tuned model MUST pass ALL five thresholds simultaneously before promotion to production. Partial passes (e.g., high F1 but poor ECE) are rejected — a well-calibrated classifier is non-negotiable for the abstention system.

---

## 5. Deployment Strategy

### 5.1 Integration with LLMAdjudicator

`app/intelligence/llm_adjudicator.py` uses `self.model_name` to route to the LLM router. To activate the fine-tuned model:

1. Register the fine-tuned model ID in `app/core/config.py`:
   ```python
   fine_tuned_model_id: str = "ft:gpt-4o-mini-2024-07-18:social-media-radar:signal-v1:XXXXX"
   ```
2. Update `LLMAdjudicator.__init__` to read `settings.fine_tuned_model_id` when `use_fine_tuned=True`.
3. Route high-stakes signal types (CHURN_RISK, LEGAL_RISK, SECURITY_CONCERN, REPUTATION_RISK) to the frontier model; all others to the fine-tuned model. Implement in `LLMRouter` via a `signal_type_tier` dispatch table.

### 5.2 Zero-Downtime Rollout

Use the existing `LLMRouter` with a `shadow_model` configuration:
- Deploy fine-tuned model in shadow mode (parallel calls, results logged but not served)
- After 48 hours of shadow traffic with ECE ≤ 0.05, promote to primary
- Keep frontier model as hot standby

### 5.3 Monitoring Post-Deployment

Wire `FeedbackCollector.get_approval_rate()` to a daily cron job. If approval rate drops below 0.75 for any signal type over a 7-day window, trigger an alert and pause routing to the fine-tuned model for that type.

---

## 6. Rollback Plan

If the fine-tuned model underperforms baseline on any evaluation metric:

1. **Immediate:** Revert `settings.fine_tuned_model_id` to the previous baseline model ID via environment variable (no code change required).
2. **Within 1 hour:** `LLMRouter` picks up the config change on next startup. All in-flight requests complete with the old model.
3. **Root cause analysis:** Compare per-signal-type F1 between fine-tuned and baseline. Identify signal types where fine-tuning degraded accuracy. Augment training data for those types and re-train.
4. **Data fix:** If κ < 0.85 was accepted in any batch, re-annotate and exclude bad examples. Re-train from clean data only.
5. **Re-evaluation gate:** Must pass all 5 thresholds in section 4 before re-attempting promotion.

