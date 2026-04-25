# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 35
- Exact-match failures: 47

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 34x status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- 34x conflicts must be an array of strings.
- 34x notes must be an array of strings.
- 33x Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes
- 33x task must equal 'extract_cf_request'
- 33x cf_request must be an object.
- 33x missing_fields must be an array of strings.
- 18x Unexpected top-level keys: schema_reference
- 10x Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference
- 5x Unexpected top-level keys: feature_dictionary, schema_reference

## Representative failed runs

### A01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to produce a JSON object with the top-level keys: schema_reference (but in example they use \"schema_reference\" inside? Actually output struct
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,       "CDAc

### A02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to output JSON only, following schema reference.\n\nThe \"case payload\" includes inputs:\n\n- mortgage of 80\n- family size 3\n- income 40\n- 
- final_message_excerpt: {   "task": "extract_cf_request",   "status": ["complete"],   "cf_request": {     "Income": 40,     "CCAvg": 1.5,     "Family": 3,     "Education": 2,     "Mortgage": 80,     "CDAccount": 1,     "Online": 1,     "Securit

### A02 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We have to produce JSON with keys: schema_reference, feature_dictionary, allowed_status_values? Wait the instructions say \"Return exactly one JSON obj
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,       "CDAc

### A03 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to produce JSON with top-level keys: \"schema_reference\" containing task, status, cf_request fields etc. The user wants extraction of the requ
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 55,       "CCAvg": 2.2,       "Family": 4,       "Education": 3,       "Mortgage": 120,       "CDA

### A03 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to produce JSON with keys: schema_reference object containing task, status, cf_request, missing_fields, conflicts, notes. Must match the specif
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 55,       "CCAvg": 2.2,       "Family": 4,       "Education": 3,       "Mortgage": 120,       "CDA

### A04 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to produce JSON object with schema_reference and feature_dictionary etc? Wait the user says: \"Return exactly one JSON object that matches the 
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete","cf_request":{"Income":65,"CCAvg":3.0,"Family":2,"Education":1,"Mortgage":0,"CDAccount":1,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fie

### A04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to produce JSON object with fields: schema_reference containing task, status, cf_request, missing_fields, conflicts, notes. The input string is
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 65,       "CCAvg": 3.0,       "Family": 2,       "Education": 1,       "Mortgage": 0,       "CDAcc

### A04 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We need to produce JSON object per schema.\n\nInput:\n\n\"Target outcome: Income 65, CCAvg 3.0, Family 2, Education 1, Mortgage 0. Keep CD account yes,
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete","cf_request":{"Income":65,"CCAvg":3.0,"Family":2,"Education":1,"Mortgage":0,"CDAccount":1,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fie

### A05 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We have to produce a JSON object that matches schema reference.\n\nInput: \"I need income 40.0, card average spending 1.50, family 3, education code 2,
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,       "CDAc

### A05 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "unsloth/gpt-oss-20b", "output": [{"content": "We must produce JSON object with top-level keys: schema_reference? Actually output should match schema reference? The user says \"Return exactly one JS
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": ["complete"],     "cf_request": {       "Income": 40.0,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80.0,
