# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 19
- Exact-match failures: 37

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 15x Unexpected top-level keys: schema_reference
- 15x task must equal 'extract_cf_request'
- 15x status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- 10x Missing top-level keys: task, status
- 5x Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes
- 5x cf_request must be an object.
- 5x missing_fields must be an array of strings.
- 5x conflicts must be an array of strings.
- 5x notes must be an array of strings.
- 4x Education must be an integer.

## Representative failed runs

### A01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields specified. All values numeric or binary. Convert \"yes\" to 1. Status complete.", "type": "reasoning"}, {"content": "{\"schema_refere
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete","cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_fi

### A02 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields specified. So status complete. No missing. No conflicts. Notes empty.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"t
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_f

### A02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields provided. So status complete. No missing fields. Notes empty.\n\nReturn JSON.", "type": "reasoning"}, {"content": "{\"schema_referenc
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_f

### A03 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We need fill all fields. All specified. No conflict. So status complete. Provide cf_request with numbers: Income 55 (number), CCAvg 2.2, Family 4, Educa
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete","cf_request":{"Income":55,"CCAvg":2.2,"Family":4,"Education":3,"Mortgage":120,"CDAccount":0,"Online":0,"SecuritiesAccount":0,"CreditCard":0},"missing_f

### A03 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We can extract all fields: Income 55, CCAvg 2.2, Family 4, Education 3, Mortgage 120, CDAccount 0, Online 0, SecuritiesAccount 0, CreditCard 0. All spec
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":55,"CCAvg":2.2,"Family":4,"Education":3,"Mortgage":120,"CDAccount":0,"Online":0,"SecuritiesAccount":0,"CreditCard":0},"missing_

### A04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields specified. So status complete. Provide cf_request with numbers and binaries. Note: CDAccount yes -> 1, Online yes->1, SecuritiesAccou
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete","cf_request":{"Income":65,"CCAvg":3.0,"Family":2,"Education":1,"Mortgage":0,"CDAccount":1,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fie

### A05 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields specified: Income=40.0; CCAvg=1.5; Family=3; Education=2; Mortgage=80.0; CDAccount=on -> 1; Online=on->1; SecuritiesAccount=on->1; Cr
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":40.0,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80.0,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missi

### A06 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have all fields present. So status complete. No missing. No conflicts. Notes empty.", "type": "reasoning"}, {"content": "{\"schema_reference\":{\"tas
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":72,"CCAvg":4.8,"Family":1,"Education":2,"Mortgage":200,"CDAccount":1,"Online":0,"SecuritiesAccount":1,"CreditCard":0},"missing_

### A07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "Need to extract fields: Income=90, CCAvg=5.5, Family=5, Education=3, Mortgage=250, CDAccount=0, Online=1, SecuritiesAccount=0, CreditCard=1. All present
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete"},"cf_request":{"Income":90,"CCAvg":5.5,"Family":5,"Education":3,"Mortgage":250,"CDAccount":0,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_

### B03 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "openai/gpt-oss-20b", "output": [{"content": "We have missing fields: CCAvg, Family, Education, CDAccount, CreditCard. So partial. Provide notes? Maybe note that unspecified fields omitted. No confl
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"partial"},"cf_request":{"Income":50,"Mortgage":100,"Online":1},"missing_fields":["CCAvg","Family","Education","CDAccount","CreditCard"],"conflicts":[],"notes":[]
