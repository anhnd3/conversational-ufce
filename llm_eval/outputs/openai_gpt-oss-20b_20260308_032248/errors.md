# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 12
- Exact-match failures: 33

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 12x Unexpected top-level keys: schema_reference
- 12x task must equal 'extract_cf_request'
- 12x status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- 9x Missing top-level keys: task, status
- 3x Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes
- 3x cf_request must be an object.
- 3x missing_fields must be an array of strings.
- 3x conflicts must be an array of strings.
- 3x notes must be an array of strings.

## Representative failed runs

### A01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields specified. Status complete. No missing_fields. notes empty.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"task\":\"ex
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_f

### A02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields: Income=40, CCAvg=1.5, Family=3, Education=2, Mortgage=80. All binary fields 1. So complete. No missing.", "type": "reasoning"}, {"co
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_f

### A02 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "Need to fill cf_request with given values. All fields present. status complete. missing_fields empty. conflicts none. notes empty.", "type": "reasoning"
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_f

### A04 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "Need to extract values. All fields present. So status complete. No missing. Notes empty.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"t
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":65,"CCAvg":3.0,"Family":2,"Education":1,"Mortgage":0,"CDAccount":1,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fi

### A08 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields specified. Binary mapping: no CD account -> 0; yes online->1; no securities account->0; no credit card->0. status complete.", "type":
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":30,"CCAvg":0.8,"Family":2,"Education":1,"Mortgage":0,"CDAccount":0,"Online":1,"SecuritiesAccount":0,"CreditCard":0},"missing_fi

### B01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields except binary ones: CDAccount, Online, SecuritiesAccount, CreditCard. So partial status, missing_fields list those four. No conflicts
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80},"missing_fields":["CDAccount","Online","SecuritiesAccount","CreditCard"],"

### B03 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have fields: Income=50, Mortgage=100, Online=1, SecuritiesAccount=0. Others missing. So partial status with missing_fields list. No conflicts. Notes 
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"partial"},"cf_request":{"Income":50,"Mortgage":100,"Online":1,"SecuritiesAccount":0},"missing_fields":["CCAvg","Family","Education","CDAccount","CreditCard"],"co

### B03 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have Income=50, Mortgage=100, Online=1, SecuritiesAccount=0. Others unspecified => partial. missing_fields list: Family, Education, CCAvg, CDAccount,
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"partial","cf_request":{"Income":50,"Mortgage":100,"Online":1,"SecuritiesAccount":0},"missing_fields":["Family","Education","CCAvg","CDAccount","CreditCard"],"con

### B05 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "Need Income=45. Others missing => partial, missing_fields list all other fields.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"task\":\"
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"partial"},"cf_request":{"Income":45},"missing_fields":["CCAvg","Family","Education","Mortgage","CDAccount","Online","SecuritiesAccount","CreditCard"],"conflicts"

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We need extract fields: Income? None. CCAvg? None. Family=3. Education? none. Mortgage: \"should stay low\" ambiguous numeric not given -> missing. CDAc
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Family":3,"CDAccount":1,"Online":1},"missing_fields":["Income","CCAvg","Education","Mortgage","SecuritiesAccount","CreditCard"],"conflicts":[],"notes":["Mort
