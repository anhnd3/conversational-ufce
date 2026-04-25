# Error Summary

- API errors: 0
- Invalid JSON runs: 1
- Schema-invalid runs: 4
- Exact-match failures: 27

## Top API errors

- none

## Top parse errors

- 1x Expecting ',' delimiter (line 1, column 236)

## Top validation errors

- 3x Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes
- 3x Unexpected top-level keys: schema_reference
- 3x task must equal 'extract_cf_request'
- 3x status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- 3x cf_request must be an object.
- 3x missing_fields must be an array of strings.
- 3x conflicts must be an array of strings.
- 3x notes must be an array of strings.
- 1x No parsed JSON object available.

## Representative failed runs

### A04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to produce a JSON object matching schema_reference. Extract fields.\n\nInput: \"Target outcome: Income 65, CCAvg 3.0, Family 2, Education 1, Mo
- final_message_excerpt: {"task":"extract_cf_request","status":"complete","cf_request":{"Income":65,"CCAvg":3.0,"Family":2,"Education":1,"Mortgage":0,"CDAccount":1,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fields":[],"conflicts":

### A05 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We must produce a JSON object with structure per schema reference.\n\nInput gives all fields: Income 40.0, CCAvg 1.50, Family 3, Education 2, Mortgage 
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40.0,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80.0,       "

### A07 repeat 1
- api_error: none
- parse_error: Expecting ',' delimiter (line 1, column 236)
- validation_errors: No parsed JSON object available.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to produce JSON object per schema reference.\n\nUser: \"Return exactly one JSON object that matches the schema reference.\" We must parse input
- final_message_excerpt: {"task":"extract_cf_request","status":"complete","cf_request":{"Income":90,"CCAvg":5.5,"Family":5,"Education":3,"Mortgage":250,"CDAccount":0,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fields":[],"conflicts

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to produce JSON with structure: top-level keys: task, status, cf_request (only explicitly inferable values), missing_fields list, conflicts lis
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Online":1,"CDAccount":1,"Family":3},"missing_fields":["Income","CCAvg","Mortgage","Education","SecuritiesAccount","CreditCard"],"conflicts":[],"notes":[]}

### B07 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to parse the user's request: \"I want online banking and a CD account, and my family size is 3. My mortgage should stay low.\" The case payload
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "CDAccount": 1,     "Online": 1   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage",     "

### B07 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to extract cf_request fields. Input: \"I want online banking and a CD account, and my family size is 3. My mortgage should stay low.\"\n\nField
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Online":1,"CDAccount":1,"Family":3},"missing_fields":["Income","CCAvg","Mortgage","Education","SecuritiesAccount","CreditCard"],"conflicts":[],"notes":["Mort

### B08 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need output JSON with top-level keys: task, status, cf_request, missing_fields, conflicts, notes.\n\nGiven input: \"I want income 48, card average 1
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Income":48,"CCAvg":1.2,"Family":2,"CreditCard":1},"missing_fields":["Education","CDAccount","Online","SecuritiesAccount"],"conflicts":[],"notes":[]}

### B08 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to produce JSON object matching schema.\n\nThe user input: \"I want income 48, card average 1.2, family 2, and I studied at the undergraduate l
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Income":48,"CCAvg":1.2,"Family":2,"CreditCard":1},"missing_fields":["Education","Mortgage","CDAccount","Online","SecuritiesAccount"],"conflicts":[],"notes":[

### B08 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need produce JSON with required structure: task=\"extract_cf_request\", status accordingly. We parse input: \"I want income 48, card average 1.2, fa
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Income":48,"CCAvg":1.2,"Family":2,"CreditCard":1},"missing_fields":["Education","Mortgage","CDAccount","Online","SecuritiesAccount"],"conflicts":[],"notes":[

### C01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need produce JSON with top-level keys: task, status, cf_request, missing_fields, conflicts, notes.\n\nThe input text: \"I want CDAccount yes and no 
- final_message_excerpt: {"task":"extract_cf_request","status":"conflict","cf_request":{"Income":40,"Online":1},"missing_fields":["CCAvg","Family","Education","Mortgage","SecuritiesAccount","CreditCard"],"conflicts":["CDAccount specified as both
