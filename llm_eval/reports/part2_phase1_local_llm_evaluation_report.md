# Part II – Phase 1 Report
## Local LLM Evaluation for Conversational UFCE

## 1. Executive conclusion

Phase 1 is successful: we now have a clear, evidence-based local model selection result for the **constraint-extraction layer** of Part II.

The most important conclusion is:

- **`qwen3-14b Q4_K_M` is the best practical winner**
- **`gpt-oss-20b` is the best fast alternative**
- **`gemma-3-12b` is the best conservative backup**

At the same time, the benchmark also exposed an important system-level insight:

> **Almost all local models were strong enough on clean extraction (Group A), some were good on partial extraction (Group B), but nearly all remained weak on contradiction/ambiguity semantics (Group C).**

That result supports the architectural decision that:

- the **LLM should not own final recourse logic**
- the **UFCE / validator / negotiation module must remain deterministic**
- the LLM’s role should be constrained to **structured extraction, clarification support, and explanation rendering**

The final comparison in this report is compiled from the per-run benchmark summaries and the failure-pattern review from the uploaded artifacts.

---

## 2. Context: why Part II needs a local LLM at all

Part II is not trying to make the LLM generate counterfactuals directly.

The core scientific and computational engine remains:

- **UFCE / deterministic recourse logic**
- feature constraints
- feasibility checks
- validator rules
- negotiation logic when the request is infeasible

The problem is that real users do not speak in feature vectors or JSON. They speak in natural language, for example:

- “I can try to raise my income a little.”
- “I don’t want to change my job.”
- “I want minimal changes.”
- “I can accept online banking, but not risky products.”

So Part II needs an interface layer that can convert **natural language → structured CF request**.

That is the local LLM’s role.

### Why local instead of cloud/API model

Choosing a **local model** is justified for both practical and thesis reasons:

### 2.1 Privacy and data control
Even in the benchmark stage, the target use case resembles financial-profile interaction. A local model avoids sending potentially sensitive structured preference data to an external provider.

### 2.2 Reproducibility
A local model is much easier to freeze for the thesis:
- exact model family
- exact quantization
- exact prompt version
- exact runtime environment

That is much better for a research artifact than relying on a changing hosted API.

### 2.3 Cost and iteration speed
You needed many repeated runs:
- multiple models
- multiple quantizations
- 22 benchmark cases
- repeated trials

That is much more practical locally.

### 2.4 Agent-mode feasibility
Later in Part II, the model may be used inside an interactive loop:
- parse user request
- receive validator feedback
- ask clarification
- produce explanation

That favors a model you can run repeatedly, cheaply, and under your own control.

### 2.5 Architectural discipline
The LLM is not the decision-maker. It is a **controlled parser / interaction assistant**.

---

## 3. Part II architecture and where the LLM fits

The architecture for Part II should be framed like this:

### 3.1 User-facing flow
1. User describes desired changes in natural language  
2. Local LLM converts that into a **strict CF JSON request**  
3. Validator checks:
   - schema validity
   - typed fields
   - missing values
   - contradictions
4. Deterministic UFCE layer evaluates feasibility  
5. If infeasible:
   - validator / recourse layer returns reason codes
   - negotiation module decides what can be relaxed
6. LLM may then:
   - ask a clarification question
   - explain the reason for infeasibility
   - render the final accepted structured request back to natural language

### 3.2 The LLM’s bounded responsibility
The benchmark was designed around this assumption:

**LLM responsibility**
- extract fields
- preserve hard constraints
- avoid hallucinating values
- detect missing/contradictory information
- produce valid structured JSON

**Not the LLM’s responsibility**
- solve recourse optimization
- decide final feasible intervention path
- overrule constraint logic
- invent policy on contradictions

That boundary is important, and the benchmark results validate it.

---

## 4. Why the benchmark was reduced to the bank CF parser task

This was the right scope reduction.

Instead of benchmarking a full conversational agent end-to-end too early, we evaluated the most critical first function:

> **Natural language → bank-domain CF JSON request**

Canonical target fields:

- `Income`
- `CCAvg`
- `Family`
- `Education`
- `Mortgage`
- `CDAccount`
- `Online`
- `SecuritiesAccount`
- `CreditCard`

This gives a benchmark that is:

- small enough to run repeatedly
- strict enough to expose model weaknesses
- close to the real Part II pipeline
- interpretable at field level

---

## 5. How the benchmark was designed from Part II needs

The benchmark was not random prompt testing. It was derived from the needs of the future MVP.

### 5.1 Benchmark objective
The benchmark needed to answer:

1. Can the model produce **valid JSON**?
2. Can it follow the **exact schema contract**?
3. Can it extract values without **hallucinating** missing fields?
4. Can it distinguish:
   - complete request
   - partial request
   - needs clarification
   - conflict
5. Can it remain reasonably **stable** across repeats?

### 5.2 Output contract
The output schema was designed as a strict wrapper, not just a raw CF object:

- `task`
- `status`
- `cf_request`
- `missing_fields`
- `conflicts`
- `notes`

This matters because Part II later needs not only values, but also:
- detection of uncertainty
- detection of contradiction
- explicit representation of incompleteness

### 5.3 Three benchmark groups

#### Group A — complete extraction
These are fully specified user requests.

Purpose:
- test easy parser correctness
- check whether the model can map direct user language into the exact CF object

#### Group B — partial / underspecified extraction
These are realistic user requests where some values are omitted.

Purpose:
- test conservative behavior
- ensure the model does **not invent** unspecified fields
- ensure missing information is surfaced explicitly

This group is especially important for Part II, because real users often speak like this.

#### Group C — conflict / ambiguity
These are contradictory or ambiguous requests.

Purpose:
- test the model’s ability to represent uncertainty correctly
- check whether it uses `conflicts`, `missing_fields`, and `status` properly

This group turned out to be the hardest for almost every model.

### 5.4 Scoring logic
The harness evaluated:

- valid JSON rate
- schema-valid rate
- exact-match rate
- field accuracy
- status accuracy
- missing-fields accuracy
- conflict accuracy
- stability across repeats
- latency / throughput

That scoring setup was appropriate because the task is not open-ended QA; it is **contract obedience under structured extraction**.

---

## 6. What the raw/parsed-output review taught us

Beyond the aggregate summaries, the behavior patterns were highly informative.

### 6.1 Common failure family: prompt/schema contamination
Several models produced JSON, but often returned:

- `schema_reference`
- `allowed_status_values`
- `feature_dictionary`

as part of the answer object.

That means the model was “aware of the schema” but not properly aligned to return only the target output object.

This affected:
- `gpt-oss-20b`
- `qwen3.5-9b`
- Phi runs
- GLM run

### 6.2 Common failure family: overfilling missing values
Some models tried to be too helpful:
- filled unknown fields with `0`
- filled unknown fields with `null`
- completed partial cases aggressively

This hurt:
- `ministral-3-14b-reasoning`
- `gemma-3-27b-it Q2`
- `qwen3-30b-a3b-instruct-2507 Q2`

### 6.3 Stronger models did better on Group B than on Group C
This is the key system insight.

The best models:
- were excellent on **Group A**
- strong on **Group B**
- still weak on **Group C exact-match**

So the benchmark says:

> use the LLM for extraction and clarification support, but keep contradiction resolution deterministic.

---

## 7. Final comparison table of tested models

To keep the final table clean and useful, this table lists the **best or most representative retained run/config** for each model/configuration family.

| Model / Config | Quant | Schema Valid | Exact Match | Field Accuracy | Status Accuracy | Stability | Avg Latency | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **qwen3-14b** | **Q4_K_M** | **0.9697** | **0.6212** | **0.9966** | **0.8788** | **0.5909** | **16.7s** | **Best practical winner** |
| qwen3-14b | Q6_K (retest) | 0.9848 | 0.6061 | 0.9966 | 0.8182 | 0.5455 | 13.5s | Near-tie with Q4_K_M |
| qwen3.5-35b-a3b | Q2_K_XL | 1.0000 | 0.6061 | 1.0000 | 0.8636 | 0.5455 | 41.7s | Strong but too slow |
| gemma-3-12b | — | 0.8182 | 0.5606 | 0.9848 | 0.7273 | **0.8636** | 10.0s | Best conservative backup |
| gpt-oss-20b | best retained run | 0.8485 | 0.5000 | 0.9444 | 0.6515 | 0.3636 | **1.21s** | Fastest serious candidate |
| mistralai/ministral-3-14b-reasoning | — | 0.7879 | 0.5303 | 0.8451 | 0.7576 | 0.6364 | 4.87s | Strong A, weaker B/C |
| qwen3-14b-128k | Q4_1 | 0.8939 | 0.5909 | 0.9916 | 0.8182 | 0.5455 | 21.3s | Good, but below main 14B variants |
| qwen3-30b-a3b-instruct-2507 | Q2_K_XL | 0.8182 | 0.5000 | 0.7256 | 0.6667 | 0.6818 | **0.64s** | Very fast, but too weak on B/C |
| gemma-3-27b-it | Q2_K_XL | 0.4091 | 0.4091 | 0.5859 | 0.7273 | **1.0000** | 3.57s | Stable but poor strict extraction |
| qwen3.5-9b | — | 0.5758 | 0.3182 | 0.7306 | 0.4848 | 0.1364 | 31.5s | Weak baseline |
| phi-4-reasoning-vision-15b | Q4_K_M | 0.0606 | 0.0303 | 0.4949 | 0.0606 | 0.0909 | 2.92s | Not suitable |
| phi-4-reasoning-vision-15b | Q6_K | 0.0758 | 0.0303 | 0.4747 | 0.0606 | 0.0909 | 5.50s | Not suitable |
| glm-z1-9b-0414 | Q8_K_XL | 0.2273 | 0.1515 | 0.6061 | 0.3030 | 0.0909 | 19.7s | Not suitable |

---

## 8. Final top 3 picks

### 8.1 Rank 1 — `qwen3-14b Q4_K_M`
This is the best final pick.

Why:
- highest practical exact-match performance
- near-perfect field accuracy
- strong schema obedience
- best balance of quality, speed, and deployment realism
- clearly better than larger Q2 models for this task

### 8.2 Rank 2 — `gpt-oss-20b`
This should be the second choice.

Why:
- by far the fastest serious candidate
- strong field-level extraction ability
- good enough overall to remain interesting
- errors are often **systematic wrapper problems**, not semantic collapse

### 8.3 Rank 3 — `gemma-3-12b`
This is the best backup / conservative fallback.

Why:
- clean behavior
- strong stability
- strong parser discipline
- less flashy, but more predictable than several alternatives

---

## 9. Strengths, weaknesses, and mitigations for the final top 3

## 9.1 `qwen3-14b Q4_K_M`

### Strengths
- best overall practical benchmark result
- excellent on complete extraction
- very strong on partial extraction
- low hallucination in A/B
- good schema-validity
- good enough latency for local use

### Weaknesses
- still weak on conflict/ambiguity exact-match
- moderate stability, not perfect
- Group C remains unsolved
- may drift between `partial` and `needs_clarification` in edge cases

### Risk for agent mode
In later agent mode, if the model is asked to do too much, it may:
- over-interpret contradictions
- return semantically plausible but benchmark-inconsistent conflict objects
- produce unstable clarification semantics

### Mitigation
Use it in a **two-stage constrained agent pattern**:

#### Stage 1 — extraction-only prompt
- JSON only
- no explanation
- no negotiation
- no reasoning text surfaced
- temperature 0

#### Stage 2 — deterministic validator
- reject malformed outputs
- compute reason codes
- determine whether the case is:
  - complete
  - partial
  - contradiction
  - infeasible under UFCE

#### Stage 3 — controlled clarification
Only if needed, call the LLM again with:
- validator reason code
- one allowed question type
- one missing-field target

This keeps Qwen in its strongest zone.

## 9.2 `gpt-oss-20b`

### Strengths
- fastest serious candidate by far
- strong raw parser ability
- high throughput
- good field extraction even when exact-match is lower
- attractive for iterative agent loops

### Weaknesses
- repeated `schema_reference` wrapping
- low stability
- weaker exact-match than top Qwen
- poor Group C behavior

### Risk for agent mode
In agent mode, this model may:
- package the answer in a schema-meta wrapper
- drift in outer object shape
- be too unstable if prompts are long or overloaded

### Mitigation
This model needs **prompt simplification + safe post-unwrapping**.

#### Mitigation A — remove schema-like prompt clutter
Avoid putting too much schema documentation inline.

#### Mitigation B — keep output format ultra-minimal
Give only:
- required output object
- field list
- strict “return one JSON object only”

#### Mitigation C — safe wrapper recovery
In the actual agent system, add a narrow repair rule:
- if the only top-level extra key is `schema_reference`
- and the nested object clearly contains the intended target structure
- unwrap it deterministically

#### Mitigation D — use GPT-OSS as a fast first-pass router
One attractive design later is:
- GPT-OSS first-pass parser
- validator checks result
- fallback to Qwen or Gemma if invalid

## 9.3 `gemma-3-12b`

### Strengths
- high stability
- clean behavior
- strong exact field extraction
- very conservative output style
- good fallback personality for structured tasks

### Weaknesses
- slower than GPT-OSS
- weaker than Qwen on top-line quality
- conflict handling is still weak
- wrapper discipline around `conflicts` can break in hard cases

### Risk for agent mode
In later agent mode, Gemma may:
- stay too conservative
- under-specify conflict representation
- underperform if prompt complexity grows

### Mitigation
Use Gemma as a **backup parser or confirmation model**, not the main innovation layer.

Good use cases:
- final-stage benchmark confirmation
- sanity-check baseline
- fallback when the system wants a conservative parse from a partial request

---

## 10. Main research finding from Phase 1

The biggest takeaway is not just “Qwen won.”

The bigger finding is:

> **For local structured recourse extraction, medium-size models with healthy quantization often outperform larger models with harsher quantization, and nearly all tested local models remain weak on contradiction-heavy semantics under strict output contracts.**

This is a very useful thesis result.

It supports:
- choosing a local model rationally
- keeping the LLM role narrow
- building Part II around deterministic validation and negotiation logic

---

## 11. Final model decision for Part II

I recommend freezing the decision like this:

### Primary model
**`qwen3-14b Q4_K_M`**

### Secondary / speed-focused alternative
**`gpt-oss-20b`**

### Backup conservative baseline
**`gemma-3-12b`**

This matches the current preference and is well supported by the evidence.

---

## 12. What this means for Phase 2

Phase 2 should now move from **model selection** to **system integration**.

Recommended next focus:

### 12.1 Freeze the parser contract
- final prompt
- final schema
- final validator behavior
- final repair policy

### 12.2 Integrate chosen model into the Part II pipeline
- user utterance
- parser
- validator
- UFCE core
- reason-code output
- clarification turn
- final explanation

### 12.3 Build the negotiation benchmark
Now that the parser model is chosen, the next experiments should evaluate:
- clarification success
- infeasibility recovery
- number of turns
- preservation of hard constraints

### 12.4 Keep Group C deterministic-first
The benchmark strongly suggests:
- do not let the LLM “solve” contradiction logic alone
- let the validator and reason-code system drive it

---

## 13. Final wrap-up statement

Phase 1 of Part II is complete.

You now have:
- a justified reason to use a **local LLM**
- a well-designed parser benchmark derived from the real needs of Part II
- a clear ranking across tested local models and quantizations
- and a final deployment decision with backup options

Final wrap-up:

> **Qwen3-14B Q4_K_M is the best practical local parser for Part II. GPT-OSS-20B is the fastest high-potential alternative, and Gemma-3-12B remains the safest conservative backup. The benchmark also shows that contradiction-heavy semantics should remain validator-driven rather than LLM-driven.**
