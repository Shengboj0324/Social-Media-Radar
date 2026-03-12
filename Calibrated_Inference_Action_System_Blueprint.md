# Calibrated Inference-and-Action System Blueprint

## Objective
Transform the current Social Media Radar codebase from a retrieval-and-summary system into a robust **AI/ML-driven calibrated inference and action engine** that can:

1. infer latent commercial/customer signals from messy real-world content,
2. abstain when confidence is weak,
3. rank opportunities by expected business value,
4. generate action recommendations and channel-specific drafts,
5. learn from downstream outcomes.

This blueprint is intentionally implementation-oriented.

---

## 1. Current gap

The present codebase still centers on:
- connector ingestion,
- normalization,
- vector search,
- clustering,
- summarization,
- personalization modules.

That stack is useful, but it is not yet a calibrated action system.

The missing core is a **decision architecture** with:
- typed signal inference,
- uncertainty modeling,
- learned ranking,
- policy-based workflow execution,
- outcome-driven learning,
- evaluation for adversarial and unpredictable inputs.

---

## 2. Target system architecture

```text
Raw content -> normalization/enrichment -> candidate signal retrieval ->
LLM structured adjudication -> confidence calibration ->
action ranking -> response planning -> draft generation ->
policy/safety checks -> operator queue -> outcome logging -> model updates
```

### New first-class objects

#### `NormalizedObservation`
Represents a single post/thread/comment with context.

Fields:
- id
- platform
- author/account metadata
- title
- body/raw_text
- quoted text
- thread context
- language
- translated_text
- entities
- competitor mentions
- time features
- engagement features
- embedding
- platform metadata

#### `SignalInference`
Represents model-level interpretation before promotion to an action object.

Fields:
- observation_id
- candidate_signal_types
- top_signal_type
- confidence_raw
- confidence_calibrated
- evidence_spans
- rationale
- ambiguity_flags
- requires_more_context
- abstain_reason
- risk_labels
- action_hypotheses

#### `ActionableSignal`
Represents an operational unit shown to the customer.

Fields:
- id
- signal_type
- priority_score
- opportunity_score
- urgency_score
- risk_score
- confidence_score
- recommended_action
- suggested_channel
- response_brief
- draft_variants
- status
- expires_at
- owner
- feedback/outcome

---

## 3. Signal taxonomy

The system needs a stable taxonomy. Recommended initial set:

- `lead_capture`
- `competitor_displacement`
- `renewal_risk`
- `product_confusion`
- `support_escalation`
- `feature_request_pattern`
- `trend_to_content`
- `creator_amplification`
- `misinformation_risk`
- `reputation_risk`
- `monitor_only`
- `abstain`

### Design rule
Use **multi-label inference first**, then policy-driven collapse to a primary label if needed.

This prevents brittle forced classification.

---

## 4. The inference pipeline

## Stage A: normalization and enrichment

### Build a new module
`app/intelligence/normalization.py`

Responsibilities:
- merge title/body/quoted text,
- infer language,
- run translation for non-English content,
- extract entities and competitor mentions,
- attach thread context,
- compute engagement/freshness features,
- generate embeddings.

### Output
`NormalizedObservation`

### Why
This removes scattered preprocessing logic and gives downstream models a consistent contract.

---

## Stage B: semantic candidate generation

### Build a candidate retriever
`app/intelligence/candidate_retrieval.py`

Inputs:
- normalized observation
- labeled exemplar bank
- taxonomy definitions

Methods:
- embedding similarity to canonical signal exemplars,
- lightweight classifier probabilities,
- entity-conditioned rules,
- platform-specific prior adjustments.

### Output
Top-k signal candidates with weak scores.

### Why
Do not let regex be the gatekeeper. Regex can remain as a low-weight feature only.

---

## Stage C: structured LLM adjudication

### Build
`app/intelligence/llm_adjudicator.py`

The LLM should not produce free text only. It must output a strict schema.

Example output schema:

```json
{
  "candidate_signal_types": ["competitor_displacement", "lead_capture"],
  "primary_signal_type": "competitor_displacement",
  "confidence": 0.78,
  "evidence_spans": [
    {"text": "Need a better alternative", "reason": "explicit replacement intent"}
  ],
  "rationale": "Author explicitly requests alternatives and names a competitor.",
  "requires_more_context": false,
  "abstain": false,
  "risk_labels": ["public_reply_safe"],
  "suggested_actions": ["reply_public", "prepare_dm_followup"]
}
```

### Design constraints
- use JSON schema validation,
- reject malformed outputs,
- retry with structured repair prompts,
- log all parse failures.

---

## Stage D: calibration

### Build
`app/intelligence/calibration.py`

Raw model confidence is not trustworthy. Add a calibration layer.

Recommended methods:
- Platt scaling,
- isotonic regression,
- temperature scaling,
- class-wise calibration curves.

### Inputs
- weak heuristic features,
- classifier logits/probabilities,
- LLM confidence,
- candidate agreement score,
- context completeness features.

### Outputs
- `confidence_calibrated`
- `abstain_probability`
- `requires_human_review`

### Decision rule
Only promote to `ActionableSignal` when calibrated confidence clears policy thresholds.

---

## 5. Replace heuristics with a hybrid ML stack

## 5.1 Classification layer

### Near-term
Use a supervised text classifier over the signal taxonomy.

Recommended baseline:
- sentence transformer embeddings + XGBoost/LightGBM,
- or DeBERTa-v3 fine-tuned classifier.

### Better production stack
- multilingual encoder for noisy social text,
- thread-context encoder,
- class imbalance handling,
- cost-sensitive loss,
- abstention-aware thresholding.

### Do not do this
Do not rely mainly on:
- keywords,
- regex match counts,
- phrase lookup confidence.

---

## 5.2 Ranking layer

### Build
`app/intelligence/ranker.py`

This should estimate business value, not topic popularity.

Recommended models:
- LambdaMART / LightGBM ranker,
- XGBoost classifier for reply-worthiness,
- contextual bandit later for action selection.

### Features
- semantic intent score,
- platform,
- author reach/quality,
- competitor mention strength,
- freshness,
- engagement velocity,
- thread depth,
- historical conversion outcome,
- account-brand fit,
- content risk,
- prior action success on similar signals.

### Outputs
- `priority_score`
- `opportunity_score`
- `replyworthiness_score`
- `risk_adjusted_action_score`

---

## 5.3 Response generation layer

### Build
`app/intelligence/action_generator.py`

Pipeline:
1. infer action intent,
2. create response brief,
3. generate 3 to 5 drafts,
4. critique each draft,
5. revise top drafts,
6. run safety/policy checks,
7. rank and return best 1 to 2.

### Critique dimensions
- relevance,
- specificity,
- platform fit,
- tone alignment,
- brand safety,
- hallucination risk,
- overclaiming,
- spamminess.

### Required output object
- response brief,
- public reply,
- DM follow-up,
- internal explanation,
- confidence,
- safe-to-send flag.

---

## 6. Workflow architecture refactor

Current systems like this usually become brittle because orchestration is string-switched. Replace that.

### Build
`app/workflows/contracts.py`

Define:
- `StepType`
- `ArtifactType`
- `ExecutionContext`
- `StepResult`
- `TransitionPolicy`

### Example enums
- `StepType.INFER_SIGNAL`
- `StepType.CALIBRATE`
- `StepType.RANK`
- `StepType.PLAN_ACTION`
- `StepType.GENERATE_RESPONSE`
- `StepType.POLICY_CHECK`
- `StepType.QUEUE`
- `StepType.LOG_OUTCOME`

### Handler contract
Every handler must declare:
- required input artifacts,
- optional inputs,
- output artifacts,
- failure behavior,
- retry behavior.

This prevents silent fallback behavior.

---

## 7. Data needed to make this truly ML-driven

You need labeled training data. No way around it.

## 7.1 Build a signal annotation dataset

Create `data/signal_annotations/` with rows like:
- text/thread,
- platform,
- signal labels,
- urgency,
- risk,
- engage/do-not-engage,
- recommended action,
- gold response quality note.

### Minimum volume
- 1,500 to 3,000 labeled examples for a usable v1
- at least 150+ per major signal class
- include hard negatives and abstain examples

## 7.2 Add hard case categories

The dataset must include:
- sarcasm,
- indirect buying intent,
- multilingual posts,
- code-switching,
- adversarial bait,
- vague dissatisfaction,
- noisy scraped text,
- very short posts,
- long thread-dependent posts,
- policy-sensitive content.

Without this, the system will look strong in demos and fail in production.

---

## 8. Evaluation framework

### Build
`app/evals/`

Modules:
- `classification_eval.py`
- `ranking_eval.py`
- `response_eval.py`
- `calibration_eval.py`
- `adversarial_eval.py`

### Metrics to track

#### Classification
- macro F1,
- per-class precision/recall,
- abstain precision,
- false-action rate.

#### Calibration
- Expected Calibration Error (ECE),
- Brier score,
- reliability diagrams,
- overconfidence rate.

#### Ranking
- NDCG@k,
- precision@k,
- opportunity hit rate,
- median rank of acted signals.

#### Response generation
- approval rate,
- edit distance to human-edited final,
- hallucination rate,
- unsafe draft rate,
- user acceptance rate.

#### End-to-end
- time-to-action,
- conversion on surfaced signals,
- missed-opportunity rate,
- analyst override frequency.

---

## 9. Trust architecture

User trust will not come from model size. It comes from system behavior.

### Requirements
Every surfaced signal should include:
- why the system flagged it,
- evidence spans,
- confidence score,
- risk classification,
- reason for suggested action,
- whether thread context was complete.

### UI-level deliverable
For each queue item, show:
- source post/thread,
- inferred signal type,
- calibrated confidence,
- 2 to 3 evidence bullets,
- recommended action,
- optional draft,
- “why not higher confidence” note if applicable.

This is critical. Black-box actioning destroys trust.

---

## 10. Concrete code changes by area

## 10.1 Fix current contract issues first

### `app/api/routes/search.py`
Problems:
- constructs `ContentItem` with mismatched fields,
- references `engagement_score` that is not in the Pydantic model,
- omits required `media_type`.

### Action
- align DB and Pydantic schemas,
- either add a structured `engagement` field to `ContentItem` or remove it from route construction,
- enforce model validation in API tests.

### `app/intelligence/__init__.py`
Problem:
- eager imports create unnecessary dependency coupling.

### Action
- remove eager imports,
- export lazily,
- avoid importing DB-heavy modules at package import time.

## 10.2 Introduce new packages

Recommended new package tree:

```text
app/
  intelligence/
    normalization.py
    candidate_retrieval.py
    llm_adjudicator.py
    calibration.py
    ranker.py
    action_generator.py
    policy_engine.py
    schemas.py
  workflows/
    contracts.py
    engine.py
    handlers/
      infer_signal.py
      calibrate.py
      rank.py
      plan_action.py
      generate_response.py
      policy_check.py
      queue_signal.py
      log_outcome.py
  evals/
    classification_eval.py
    ranking_eval.py
    calibration_eval.py
    response_eval.py
    adversarial_eval.py
```

---

## 11. Model roadmap

## Phase 1: strong pragmatic baseline
- sentence-transformer embeddings,
- LightGBM/XGBoost signal classifier,
- LightGBM ranking model,
- LLM adjudicator for hard cases,
- isotonic calibration,
- LLM response generator with critique pass.

This is the highest ROI starting point.

## Phase 2: stronger supervised models
- fine-tuned DeBERTa or multilingual encoder,
- thread-aware classifier,
- explicit risk classifier,
- judge model for response quality.

## Phase 3: online learning
- contextual bandit for action selection,
- outcome-aware ranking updates,
- user/team-specific policy adaptation,
- active learning for uncertain examples.

---

## 12. Implementation roadmap

## Sprint 1: hardening and schemas
- fix route/schema mismatches,
- define `NormalizedObservation`, `SignalInference`, `ActionableSignal`,
- add strict validation,
- decouple imports,
- add compile/import CI checks.

## Sprint 2: inference v1
- build normalization service,
- build candidate retrieval,
- implement structured LLM adjudicator,
- ship abstain support,
- store inference artifacts.

## Sprint 3: ranking v1
- create labeled dataset,
- build ranking features,
- train first LightGBM ranker,
- add policy thresholds.

## Sprint 4: action generation v1
- response brief generator,
- multi-draft generator,
- critique/revision stage,
- policy checker,
- operator queue wiring.

## Sprint 5: evaluation and calibration
- calibration module,
- adversarial test suite,
- dashboard metrics,
- confidence tuning,
- rollout gating.

## Sprint 6: learning loop
- capture outcomes,
- train outcome prediction,
- introduce contextual bandit,
- active learning queue for uncertain examples.

---

## 13. Final product form

The product should appear as a **queue-first operator console**, not a dashboard-first analytics toy.

### Main surfaces

#### 1. Opportunity Queue
A ranked list of actionable signals.

Each item contains:
- signal type,
- priority score,
- confidence,
- source post/thread,
- evidence,
- recommended action,
- suggested response.

#### 2. Signal Detail View
A drill-down page with:
- full thread context,
- evidence spans,
- inferred labels,
- confidence decomposition,
- draft variants,
- risk and policy notes.

#### 3. Team Action Console
Allows:
- approve/edit/send,
- dismiss,
- assign owner,
- mark outcome,
- feed result back into learning loop.

#### 4. Evaluation/Admin Surface
Internal only.
Shows:
- confidence reliability,
- false positive clusters,
- missed signal classes,
- response acceptance,
- drift.

---

## 14. Non-negotiable principles

1. **No silent confidence theater**
   Confidence must be calibrated, not invented.

2. **No regex gatekeeping as primary intelligence**
   Use heuristics as weak signals only.

3. **Abstention is a feature, not a bug**
   Wrong action is worse than no action.

4. **Thread context matters**
   Single-post classification is not enough.

5. **Action value beats topic novelty**
   The objective is business outcome, not interesting summaries.

6. **Every surfaced action needs traceable justification**
   Trust requires visible reasoning artifacts.

---

## 15. Immediate next moves

1. Repair schema and import correctness issues.
2. Create the three core schemas: `NormalizedObservation`, `SignalInference`, `ActionableSignal`.
3. Build normalization + candidate retrieval.
4. Add a structured LLM adjudicator with abstain support.
5. Start a labeled dataset immediately.
6. Train baseline classifier and ranker.
7. Replace brittle workflow wiring with typed contracts.
8. Add evaluation and calibration before widening feature scope.

---

## Bottom line

The next step is not “more features.”
It is **turning the codebase into a disciplined inference system**:
- robust contracts,
- hybrid ML + LLM decisioning,
- calibrated confidence,
- explicit abstention,
- learned ranking,
- action generation with critique,
- continuous evaluation.

That is the path from impressive prototype to real operator-grade AI product.
