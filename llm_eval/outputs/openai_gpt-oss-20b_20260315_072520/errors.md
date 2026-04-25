# Error Summary

- API errors: 0
- Invalid JSON runs: 1
- Schema-invalid runs: 21
- Exact-match failures: 36

## Top API errors

- none

## Top parse errors

- 1x Expecting ',' delimiter (line 1, column 189)

## Top validation errors

- 18x Unexpected top-level keys: schema_reference
- 18x task must equal 'extract_cf_request'
- 18x status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- 12x conflicts must be an array of strings.
- 12x notes must be an array of strings.
- 11x Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes
- 11x cf_request must be an object.
- 11x missing_fields must be an array of strings.
- 7x Missing top-level keys: task, status
- 1x Missing top-level keys: conflicts, notes

## Representative failed runs

### A02 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields specified. Return complete. No conflicts. Notes empty.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"task\":\"extract
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete","cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_fi

### A02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We need extract values. All present. So complete.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"task\":\"extract_cf_request\",\"status\"
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_f

### A03 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "All fields specified. Binary negatives =>0. Complete.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"task\":\"extract_cf_request\",\"stat
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":55,"CCAvg":2.2,"Family":4,"Education":3,"Mortgage":120,"CDAccount":0,"Online":0,"SecuritiesAccount":0,"CreditCard":0},"missing_

### A03 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We fill fields. All specified. So complete. No notes.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"task\":\"extract_cf_request\",\"stat
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":55,"CCAvg":2.2,"Family":4,"Education":3,"Mortgage":120,"CDAccount":0,"Online":0,"SecuritiesAccount":0,"CreditCard":0},"missing_

### A04 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "All fields provided. status complete.", "type": "reasoning"}, {"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"st
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": [       "complete"     ],     "cf_request": {       "Income": 65,       "CCAvg": 3.0,       "Family": 2,       "Education": 1,       "Mortgage": 0

### A05 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "All fields provided. Status complete.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"task\":\"extract_cf_request\",\"status\":[\"complete
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":["complete"],"cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_

### A05 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "All fields provided numeric or binary yes. So status complete.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"task\":\"extract_cf_request
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete","cf_request":{"Income":40.0,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80.0,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missin

### A06 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields specified. So status complete. No missing. no conflicts. notes empty.", "type": "reasoning"}, {"content": "<|channel|>final <|constra
- final_message_excerpt: <|channel|>final <|constrain|>JSON<|message|>{"schema_reference":{"task":"extract_cf_request","status":"complete","cf_request":{"Income":72,"CCAvg":4.8,"Family":1,"Education":2,"Mortgage":200,"CDAccount":1,"Online":0,"Se

### A08 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields specified: Income=30, CCAvg=0.8, Family=2, Education=1, Mortgage=0, CDAccount=no ->0, Online=yes->1, SecuritiesAccount=no->0, CreditC
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":30,"CCAvg":0.8,"Family":2,"Education":1,"Mortgage":0,"CDAccount":0,"Online":1,"SecuritiesAccount":0,"CreditCard":0},"missing_fi

### B04 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "Need extract Income=62, CCAvg=2.7, CreditCard=1, Online=1. Others unspecified -> partial. missing_fields list those. No conflicts. notes empty.", "type"
- final_message_excerpt: <|channel|>final <|constrain|>json<|message|>{"task":"extract_cf_request","status":"partial","cf_request":{"Income":62,"CCAvg":2.7},"missing_fields":["Family","Education","Mortgage","CDAccount","SecuritiesAccount"],"conf
