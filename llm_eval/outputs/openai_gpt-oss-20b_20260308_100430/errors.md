# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 10
- Exact-match failures: 33

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 10x Unexpected top-level keys: schema_reference
- 10x task must equal 'extract_cf_request'
- 10x status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- 8x Missing top-level keys: task, status
- 2x Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes
- 2x cf_request must be an object.
- 2x missing_fields must be an array of strings.
- 2x conflicts must be an array of strings.
- 2x notes must be an array of strings.

## Representative failed runs

### A02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields specified. So status complete. No missing. No conflicts. notes empty.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"t
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_f

### A04 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields present. So status complete. No missing. No conflicts. notes empty.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"tas
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":65,"CCAvg":3.0,"Family":2,"Education":1,"Mortgage":0,"CDAccount":1,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fi

### A04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields provided. status complete. No missing. notes empty.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"task\":\"extract_cf
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":65,"CCAvg":3.0,"Family":2,"Education":1,"Mortgage":0,"CDAccount":1,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fi

### A05 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields provided. So status complete. No missing_fields. notes empty.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"task\":\"
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":40.0,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80.0,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missi

### A06 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields provided. So status complete, missing_fields empty. notes empty.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"task\"
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":72,"CCAvg":4.8,"Family":1,"Education":2,"Mortgage":200,"CDAccount":1,"Online":0,"SecuritiesAccount":1,"CreditCard":0},"missing_

### A06 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields present. So status complete. No missing_fields, conflicts empty. notes empty.", "type": "reasoning"}, {"content": "{\"schema_referenc
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":72,"CCAvg":4.8,"Family":1,"Education":2,"Mortgage":200,"CDAccount":1,"Online":0,"SecuritiesAccount":1,"CreditCard":0},"missing_

### A08 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields specified. Binary values: CDAccount no =>0; Online yes=>1; SecuritiesAccount no=>0; CreditCard no=>0. Status complete. No missing_fie
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete","cf_request":{"Income":30,"CCAvg":0.8,"Family":2,"Education":1,"Mortgage":0,"CDAccount":0,"Online":1,"SecuritiesAccount":0,"CreditCard":0},"missing_fie

### B04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have Income=62, CCAvg=2.7, CreditCard=1, Online=1. Others missing: Family, Education, Mortgage, CDAccount, SecuritiesAccount. So status partial, miss
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"partial"},"cf_request":{"Income":62,"CCAvg":2.7,"CreditCard":1,"Online":1},"missing_fields":["Family","Education","Mortgage","CDAccount","SecuritiesAccount"],"co

### B05 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "Need to extract Income=45. Others unspecified. So partial, missing_fields list all other keys. No conflicts.", "type": "reasoning"}, {"content": "{\"sch
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"partial"},"cf_request":{"Income":45},"missing_fields":["CCAvg","Family","Education","Mortgage","CDAccount","Online","SecuritiesAccount","CreditCard"],"conflicts"

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have Income, CCAvg, Education missing. So partial. Provide cf_request with known: Online=1, CDAccount=1, Family=3. Mortgage unspecified but \"should 
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Family":3,"Online":1,"CDAccount":1},"missing_fields":["Income","CCAvg","Education","Mortgage","SecuritiesAccount","CreditCard"],"conflicts":[],"notes":[]}
