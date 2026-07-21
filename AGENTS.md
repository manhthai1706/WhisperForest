# WhisperForest — Agent Guide

## What this is

Causal Decision Intelligence Engine with multi-tier reasoning:

```
                    Data
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
 Predictive Branch          Causal Discovery
 (RandomForest)              DAG Learning
        │                           │
        └─────────────┬─────────────┘
                      ▼
               WhisperTrace RCA
                      │
         Per-patient Causal Trace
                      │
                      ▼
          CATE + Intervention Effects
                      │
                      ▼
      Knowledge / Experience Memory
                      │
                      ▼
      SCM Reasoning Engine (multi-hypothesis)
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
 Counterfactual Simulation    Policy Generator
          │                        │
          └───────────┬────────────┘
                      ▼
             Causal Auditor
          (Consistency Voting)
```

Entry point: `src/whisper_forest.py` → `WhisperForest` class. Each branch is a subpackage under `src/`.

## Run the test

```powershell
$env:PYTHONIOENCODING='utf-8'; python test_pipeline.py
```

No CI, no linter, no formatter configured.

## Key dependencies

- `pandas`, `numpy`, `scikit-learn` (core)
- `causal-learn` (PC algorithm)
- `lingam` (DirectLiNGAM — optional)

## Code structure

```
src/
  whisper_forest.py              # Main orchestrator — 5 branches + memory + reasoner
  memory/                        # NEW: Long-term knowledge layer
    causal_trace.py              #   CausalTrace dataclass (per-patient causal profile)
    knowledge_memory.py          #   KnowledgeMemory — store/query causal profiles
    reasoner.py                  #   SCMReasoner — multi-hypothesis intervention planning
    hypothesis.py                #   HypothesisGenerator + Hypothesis dataclass
  causal/
    discovery/                   # DAG discovery (PC + DirectLiNGAM ensemble)
    estimation/                  # ATE/CATE estimation (EconML DML / T-Learner)
    auditor.py                   # Cross-layer consistency audit
  predictive/                    # RandomForest classifier/regressor
  rca/
    engine/                      # SCMEngine + MixtureOfSCMEngine (EM-MoSCM)
    attribution/                 # Per-row root cause attribution
  optimization/                  # Counterfactual optimization
  policy/                        # Policy memory + recommendation
heart.csv                        # Sample dataset (303 rows, 14 columns)
test_pipeline.py                 # Integration test — all branches + reasoning engine
```

## Key conventions

- **No comments in code** — repo convention.
- **Docstrings in Vietnamese** — code/variable names in English.
- `wf.trace` is an alias for `wf.rca` (RCABranch).
- SCM must be fitted before `simulate()`/`counterfactual()` (auto-fits on first call).
- `heart.csv` primary dataset; `target` column is outcome.

## New components (5th branch)

### CausalTrace
Per-patient causal profile stored in Knowledge Memory. Contains features, RCA scores, CATE estimates, interventions tried. Supports:
- `top_causes(k)` — top-K features by attribution
- `causal_profile` — numpy vector of (RCA + CATE) for similarity search
- `best_intervention` — best intervention by delta improvement
- `get_modifiable()` — features that can be intervened on (excludes age/sex by default)
- `modifiable_features` — optional override field

### KnowledgeMemory
Long-term memory that stores CausalTraces and retrieves by **causal profile similarity** (not feature vectors). Methods:
- `store(trace)` — save trace
- `find_similar_by_causal_profile(trace, k)` — nearest causal cases
- `find_similar_by_features(trace, k)` — nearest feature vectors (for comparison)
- `get_success_rate(pattern)` — aggregate success stats
- `get_most_effective_interventions(k)` — best plans across all traces

### HypothesisGenerator
Generates multiple intervention hypotheses with rationales:
- **RCA-driven** — single-feature hypotheses targeting top root causes
- **Combined** — multi-feature hypotheses for synergistic effects
- **Knowledge-guided** — patterns from similar causal cases in memory
Each `Hypothesis` has a `.plan`, `.rationale` (human-readable story), and `.source`.

### SCMReasoner (refactored)
Multi-step reasoning pipeline:
1. `retrieve_similar_cases(trace)` — case retrieval from memory
2. `generate_hypotheses(trace)` — hypothesis generation (delegates to HypothesisGenerator)
3. `simulate_hypotheses(patient, hypotheses)` — counterfactual simulation
4. `rank_hypotheses(results)` — plan ranking by delta
5. `reason(trace)` — full pipeline

### PolicyBranch (updated)
Two modes:
- **Tree-based** (legacy): `distill_policy_from_experiments()` — decision tree on trial logs
- **Memory-based** (new): `learn_from_traces()` + `recommend_from_trace()` — learns from full (features, RCA, CATE, intervention, outcome, success) chain stored in KnowledgeMemory

## Usage flow

```python
wf = WhisperForest(data=df, target="target")
wf.predictive.fit()
dag = wf.causal.get_dag(constraints=...)
rca = wf.rca.analyze_anomaly(anomaly, baseline)

# Multi-hypothesis reasoning
ranked_results, trace, recommendations = wf.reason(
    rca_report=rca,
    patient=anomaly,
    cate_series=...,         # optional, uses RCA if not available
    n_plans=8,
)
# ranked_results: List[Tuple[Hypothesis, InterventionRecord]]
# trace: CausalTrace (stored in memory automatically)
# recommendations: ranked plans from policy

# Policy from full chain
wf.policy.learn_from_traces(wf.memory.traces)
best = wf.policy.recommend_from_trace(trace)

# Auditor
wf.audit_consistency(patient)

# Generate trace directly
trace = wf.counterfactual_trace(patient_df=patient, outcome=0)
```

## SCMReasoner detail

The `reason()` method in SCMReasoner runs this pipeline:

```
Patient Trace
    ↓
Step 1: Case Retrieval — find similar causal profiles in Experience Bank
    ↓
Step 2: Hypothesis Generation — create multiple candidate plans:
  ├─ RCA-driven: single-feature plans from root causes
  ├─ Combined: multi-feature plans for synergistic effects
  └─ Knowledge-guided: patterns from similar historical cases
    ↓
Step 3: Counterfactual Simulation — simulate each through SCM
    ↓
Step 4: Plan Ranking — sort by expected outcome improvement
    ↓
Step 5: Policy Recommendation — aggregate from successful patterns
```

## Gotchas

- No packaging — import as `from src.whisper_forest import WhisperForest` (run from repo root).
- `.gitignore` excludes `*.h5` and `*.png`.
- CATE requires a `treatment` variable; without it, reasoner falls back to RCA scores for guidance.
- Bootstrap runs (`bootstrap_runs=10`) affects DAG quality vs runtime.
- MoSCM EM loop (`n_iterations=5`) — no convergence criterion, fixed iterations.
- Console encoding on Windows: use `$env:PYTHONIOENCODING='utf-8'`.
