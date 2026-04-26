# Conversational UFCE — Bilingual E2E Testing Runbook

## Goal
Run final E2E mock testing for the response-hardening pass with both English and Vietnamese user input.

English scenarios are the formal acceptance path. Vietnamese scenarios are a localization robustness probe: Level A means same semantic result as English; Level B means safe fallback to clarification/parser failure without crash or invented advice.

## 1. Start environment

```bash
cd /mnt/d/Workspace/MasterAI/Thesis/UFCE_Agent
export LM_STUDIO_API_BASE=http://127.0.0.1:1234
export MODEL_ALIAS=qwen3-14b-q4_k_m
export PRODUCT_MODE=stable_demo
export HOST=127.0.0.1
export PORT=8000
export RUN_ID=$(date +%Y%m%d_%H%M%S)
export E2E_ROOT="outputs/part2_final_bilingual_e2e/${RUN_ID}"
mkdir -p "$E2E_ROOT"
git rev-parse HEAD | tee "$E2E_ROOT/git_commit.txt"
git status --short | tee "$E2E_ROOT/git_status_short.txt"
```

Check LM Studio:

```bash
curl -s "$LM_STUDIO_API_BASE/v1/models" | python -m json.tool | tee "$E2E_ROOT/lm_studio_models.json"
```

Start product service in Terminal A:

```bash
python scripts/run_phase3_2_demo.py
```

Health checks in Terminal B:

```bash
curl -s http://127.0.0.1:8000/api/v1/health | python -m json.tool | tee "$E2E_ROOT/health.json"
curl -s http://127.0.0.1:8000/api/v1/version | python -m json.tool | tee "$E2E_ROOT/version.json"
curl -s http://127.0.0.1:8000/api/v1/catalog/datasets | python -m json.tool | tee "$E2E_ROOT/catalog.json"
```

## 2. Automated E2E gates

```bash
python scripts/run_phase3_2_product_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --out-dir "$E2E_ROOT/product_smoke" \
  | tee "$E2E_ROOT/product_smoke_stdout.json"
```

Expected: `all_passed = true`.

```bash
python scripts/run_phase3_2_acceptance_report.py \
  --base-url http://127.0.0.1:8000 \
  --out-dir "$E2E_ROOT/acceptance" \
  | tee "$E2E_ROOT/acceptance_stdout.json"
```

Expected: `acceptance_passed = true`, `failed_gates = []`.

## 3. Helper functions

```bash
api_create_session() {
  curl -s -X POST http://127.0.0.1:8000/api/v1/sessions \
    -H "Content-Type: application/json" \
    -d '{"dataset_key":"bank"}'
}

api_send_message() {
  local session_id="$1"
  local text="$2"
  python - "$session_id" "$text" <<'PY' | curl -s -X POST "http://127.0.0.1:8000/api/v1/sessions/$(cat /tmp/session_id.txt 2>/dev/null)/messages" -H "Content-Type: application/json" -d @-
import json, sys, pathlib
session_id, text = sys.argv[1], sys.argv[2]
pathlib.Path('/tmp/session_id.txt').write_text(session_id)
print(json.dumps({'user_input': text}, ensure_ascii=False))
PY
}

api_send_refinement() {
  local session_id="$1"
  local text="$2"
  python - "$text" <<'PY' | curl -s -X POST "http://127.0.0.1:8000/api/v1/sessions/$session_id/refinements" -H "Content-Type: application/json" -d @-
import json, sys
print(json.dumps({'user_feedback': sys.argv[1]}, ensure_ascii=False))
PY
}

save_json() { local path="$1"; python -m json.tool > "$path"; }

inspect_turn() {
  local path="$1"
  python - <<PY
import json
from pathlib import Path
p = Path('$path')
payload = json.loads(p.read_text())
print('file:', p)
print('public_state:', payload.get('public_state'))
print('assistant_text:', payload.get('assistant_text'))
print('summary_type:', (payload.get('explanation_payload') or {}).get('summary_type'))
print('clarification_type:', (payload.get('clarification_payload') or {}).get('clarification_type'))
print('ui_response_kind:', (payload.get('ui_response_summary') or {}).get('response_kind'))
print('ui_tone:', (payload.get('ui_response_summary') or {}).get('tone'))
print('changed_items:', (payload.get('ui_response_summary') or {}).get('changed_items'))
print('blocked_reasons:', (payload.get('ui_response_summary') or {}).get('blocked_reasons'))
print('next_actions:', (payload.get('ui_response_summary') or {}).get('next_actions'))
print('reason_codes:', ((payload.get('debug_summary') or {}).get('runtime_summary') or {}).get('reason_codes'))
PY
}
```

If the helper has shell quoting issues, use the UI or raw curl manually. The expected JSON fields are the same.

## 4. English scenarios

### EN-01 no recourse needed
```bash
SESSION_ID=$(api_create_session | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
api_send_message "$SESSION_ID" "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0." | save_json "$E2E_ROOT/en_01_no_recourse.json"
inspect_turn "$E2E_ROOT/en_01_no_recourse.json"
```
Expected: `RUNTIME_SUCCESS`, `summary_type=no_recourse_needed`, `ui_response_kind=no_recourse_needed`, `tone=success`, no changed items.

### EN-02 valid recommendation found
```bash
SESSION_ID=$(api_create_session | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
api_send_message "$SESSION_ID" "Income 100, Family 1, CCAvg 2.7, Education 2, Mortgage 0, SecuritiesAccount 0, CDAccount 0, Online 0, CreditCard 0." | save_json "$E2E_ROOT/en_02_cf_found.json"
inspect_turn "$E2E_ROOT/en_02_cf_found.json"
```
Expected: `RUNTIME_SUCCESS`, `summary_type=counterfactual_found`, `ui_response_kind=counterfactual_found`, changed items match `profile_diff`.

### EN-03 missing fields and merge
```bash
SESSION_ID=$(api_create_session | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
api_send_message "$SESSION_ID" "Income 40 and CCAvg 1.5." | save_json "$E2E_ROOT/en_03_turn1_missing.json"
inspect_turn "$E2E_ROOT/en_03_turn1_missing.json"
api_send_message "$SESSION_ID" "Family 3, Education 2, Mortgage 80, SecuritiesAccount 0, CDAccount 0, Online 1, CreditCard 0." | save_json "$E2E_ROOT/en_03_turn2_merge.json"
inspect_turn "$E2E_ROOT/en_03_turn2_merge.json"
```
Expected turn 1: `NEEDS_CLARIFICATION`, `reply_strategy=missing_fields_only`. Expected turn 2: `RUNTIME_SUCCESS` or `RUNTIME_REJECT`, never public `READY_FOR_RUNTIME`, and `debug_summary.merge_applied=true`.

### EN-04 constraint-blocked rejection
```bash
SESSION_ID=$(api_create_session | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
api_send_message "$SESSION_ID" "Income 100, Family 1, CCAvg 2.7, Education 2, Mortgage 0, SecuritiesAccount 0, CDAccount 0, Online 0, CreditCard 0. Keep CDAccount unchanged." | save_json "$E2E_ROOT/en_04_constraint_blocked.json"
inspect_turn "$E2E_ROOT/en_04_constraint_blocked.json"
```
Expected: `RUNTIME_REJECT`, safe blocked/no-feasible explanation, bounded next action, no invalid candidate exposure.

### EN-05 same-case refinement
```bash
SESSION_ID=$(api_create_session | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
api_send_message "$SESSION_ID" "Income 100, Family 1, CCAvg 2.7, Education 2, Mortgage 0, SecuritiesAccount 0, CDAccount 0, Online 0, CreditCard 0." | save_json "$E2E_ROOT/en_05_turn1_base.json"
api_send_refinement "$SESSION_ID" "Keep CDAccount unchanged." | save_json "$E2E_ROOT/en_05_turn2_refinement.json"
inspect_turn "$E2E_ROOT/en_05_turn2_refinement.json"
```
Expected: `turn_kind=refinement`, revision index increments, constraint reflected, result is updated recommendation or clear reject.

## 5. Vietnamese scenarios

### VI-01 no recourse needed
```bash
SESSION_ID=$(api_create_session | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
api_send_message "$SESSION_ID" "Hồ sơ ngân hàng hiện tại: Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0." | save_json "$E2E_ROOT/vi_01_no_recourse.json"
inspect_turn "$E2E_ROOT/vi_01_no_recourse.json"
```
Preferred Level A: same as EN-01. Acceptable Level B: clarification/parser failure without unsafe recommendation.

### VI-02 valid recommendation found
```bash
SESSION_ID=$(api_create_session | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
api_send_message "$SESSION_ID" "Tôi muốn kiểm tra hồ sơ: Income 100, Family 1, CCAvg 2.7, Education 2, Mortgage 0, SecuritiesAccount 0, CDAccount 0, Online 0, CreditCard 0." | save_json "$E2E_ROOT/vi_02_cf_found.json"
inspect_turn "$E2E_ROOT/vi_02_cf_found.json"
```
Preferred Level A: same as EN-02. Acceptable Level B: safe clarification/parser failure.

### VI-03 missing fields and merge
```bash
SESSION_ID=$(api_create_session | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
api_send_message "$SESSION_ID" "Tôi mới có một phần thông tin: Income 40 và CCAvg 1.5." | save_json "$E2E_ROOT/vi_03_turn1_missing.json"
inspect_turn "$E2E_ROOT/vi_03_turn1_missing.json"
api_send_message "$SESSION_ID" "Bổ sung Family 3, Education 2, Mortgage 80, SecuritiesAccount 0, CDAccount 0, Online 1, CreditCard 0." | save_json "$E2E_ROOT/vi_03_turn2_merge.json"
inspect_turn "$E2E_ROOT/vi_03_turn2_merge.json"
```
Preferred Level A: missing fields then runtime result. Acceptable Level B: still clarification if parser misses fields, no fabricated runtime result.

### VI-04 constraint-blocked rejection
```bash
SESSION_ID=$(api_create_session | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
api_send_message "$SESSION_ID" "Hãy kiểm tra hồ sơ: Income 100, Family 1, CCAvg 2.7, Education 2, Mortgage 0, SecuritiesAccount 0, CDAccount 0, Online 0, CreditCard 0. Tôi muốn giữ CDAccount không thay đổi." | save_json "$E2E_ROOT/vi_04_constraint_blocked.json"
inspect_turn "$E2E_ROOT/vi_04_constraint_blocked.json"
```
Preferred Level A: `RUNTIME_REJECT` with blocked/no-feasible explanation. Acceptable Level B: clarification/parser failure without unsupported recommendation.

### VI-05 same-case refinement
```bash
SESSION_ID=$(api_create_session | python -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
api_send_message "$SESSION_ID" "Tôi muốn kiểm tra hồ sơ: Income 100, Family 1, CCAvg 2.7, Education 2, Mortgage 0, SecuritiesAccount 0, CDAccount 0, Online 0, CreditCard 0." | save_json "$E2E_ROOT/vi_05_turn1_base.json"
api_send_refinement "$SESSION_ID" "Giữ CDAccount không thay đổi." | save_json "$E2E_ROOT/vi_05_turn2_refinement.json"
inspect_turn "$E2E_ROOT/vi_05_turn2_refinement.json"
```
Preferred Level A: refinement handled. Acceptable Level B: safe refinement clarification/block.

## 6. UI manual checks

Open `http://127.0.0.1:8000` and verify:

```text
[ ] Create session works.
[ ] EN-02 shows friendly chat response and “What changed” card.
[ ] EN-04 shows “Why blocked” and “Next action”.
[ ] VI-02 either matches English result or safely asks for clarification.
[ ] Technical drawer has artifacts/debug payloads but is not dominant.
[ ] Chat/Context tabs work on narrow screen.
[ ] Start New Case works.
```

## 7. Final report and snapshot

```bash
cat > "$E2E_ROOT/manual_bilingual_e2e_report.md" <<'EOF'
# Manual Bilingual E2E Report

## Environment
- Repo commit:
- LM Studio model:
- Product mode:
- Run folder:

## Automated Results
- Product smoke: PASS / FAIL
- Acceptance report: PASS / FAIL

## English Scenarios
| Scenario | Result | Notes |
|---|---|---|
| EN-01 No recourse needed | PASS / FAIL | |
| EN-02 Valid recommendation found | PASS / FAIL | |
| EN-03 Missing fields + merge | PASS / FAIL | |
| EN-04 Constraint blocked | PASS / FAIL | |
| EN-05 Same-case refinement | PASS / FAIL | |

## Vietnamese Scenarios
| Scenario | Level A / Level B / Fail | Notes |
|---|---|---|
| VI-01 No recourse needed |  | |
| VI-02 Valid recommendation found |  | |
| VI-03 Missing fields + merge |  | |
| VI-04 Constraint blocked |  | |
| VI-05 Same-case refinement |  | |

## UI Checks
- Desktop UI: PASS / FAIL
- Mobile/narrow UI: PASS / FAIL
- Result card: PASS / FAIL
- Technical drawer: PASS / FAIL
- Start new case: PASS / FAIL

## Final Verdict
Part II bilingual E2E mock testing is accepted for thesis writing: YES / NO

## Notes For Thesis
- English flow:
- Vietnamese flow:
- Parser limitation, if any:
- Evidence folder:
EOF

find "$E2E_ROOT" -maxdepth 3 -type f | sort > "$E2E_ROOT/file_index.txt"
git rev-parse HEAD > "$E2E_ROOT/git_commit_final.txt"
git status --short > "$E2E_ROOT/git_status_short_final.txt"
```

Optional zip:

```bash
cd outputs/part2_final_bilingual_e2e
zip -r "${RUN_ID}.zip" "${RUN_ID}"
cd -
```

## 8. Stop conditions

Stop and investigate if any of these occur:

```text
- Service crashes.
- API returns READY_FOR_RUNTIME as public_state.
- assistant_text says recommendation found while public_state is RUNTIME_REJECT.
- assistant_text invents changed fields not present in profile_diff.
- invalid candidate values appear in user-facing text.
- case completion does not block follow-up after terminal states.
- UI cannot open session page after successful API turn.
```

## 9. Thesis interpretation

If Vietnamese reaches Level A:

```text
The bilingual E2E mock test shows that the controlled conversational layer can handle both English and Vietnamese user phrasing for representative bank-profile recourse scenarios while preserving deterministic UFCE-backed response semantics.
```

If Vietnamese is mixed but safe:

```text
Vietnamese testing was used as an exploratory localization robustness probe. English flows are the formal acceptance path. Vietnamese safe fallback shows that the system preserves safety by requesting clarification or blocking unsupported responses rather than fabricating counterfactual advice.
```
