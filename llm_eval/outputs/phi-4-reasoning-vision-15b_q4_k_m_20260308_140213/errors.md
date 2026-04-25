# Error Summary

- API errors: 0
- Invalid JSON runs: 4
- Schema-invalid runs: 62
- Exact-match failures: 64

## Top API errors

- none

## Top parse errors

- 3x Expecting property name enclosed in double quotes (line 1, column 2)
- 1x Expecting value (line 1, column 1)

## Top validation errors

- 55x notes must be an array of strings.
- 53x Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes
- 53x task must equal 'extract_cf_request'
- 53x status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- 53x cf_request must be an object.
- 53x missing_fields must be an array of strings.
- 53x conflicts must be an array of strings.
- 42x Unexpected top-level keys: schema_reference
- 10x Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference
- 4x No parsed JSON object available.

## Representative failed runs

### A01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q4_k_m", "output": [{"content": "{\"schema_reference\":{\"task\":\"extract_cf_request\",\"status\":\"complete\",\"cf_request\":{\"Income\":40,\"CCAvg\":1.5,\"Family\":3,\
- final_message_excerpt: {"schema_reference":{"task":"extract_cf_request","status":"complete","cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_fi

### A01 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: notes, notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q4_k_m", "output": [{"content": "{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"complete\",\n  \"cf_request\": {\n    \"Income\": 40,\n    \"CCAvg\": 1.5,\n    \
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 40,     "CCAvg": 1.5,     "Family": 3,     "Education": 2,     "Mortgage": 80,     "CDAccount": 1,     "Online": 1,     "Securitie

### A02 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q4_k_m", "output": [{"content": "{\n    \"schema_reference\": {\n        \"task\": \"extract_cf_request\",\n        \"status\": \"complete\",\n        \"cf_request\": {\n
- final_message_excerpt: {     "schema_reference": {         "task": "extract_cf_request",         "status": "complete",         "cf_request": {             "Income": 40,             "CCAvg": 1.5,             "Family": 3,             "Education"

### A02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q4_k_m", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,       "CDAc

### A02 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q4_k_m", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,       "CDAc

### A03 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q4_k_m", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 55,       "CCAvg": 2.2,       "Family": 4,       "Education": 3,       "Mortgage": 120,       "CDA

### A03 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q4_k_m", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 55,       "CCAvg": 2.2,       "Family": 4,       "Education": 3,       "Mortgage": 120,       "CDA

### A03 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q4_k_m", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 55,       "CCAvg": 2.2,       "Family": 4,       "Education": 3,       "Mortgage": 120,       "CDA

### A04 repeat 1
- api_error: none
- parse_error: Expecting property name enclosed in double quotes (line 1, column 2)
- validation_errors: No parsed JSON object available.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q4_k_m", "output": [{"content": "{...}", "type": "message"}], "response_id": "resp_7c2a51af8e0551e7e511605840449ebe88d249538fd9c4cb", "stats": {"input_tokens": 812, "reas
- final_message_excerpt: {...}

### A04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q4_k_m", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 65,       "CCAvg": 3.0,       "Family": 2,       "Education": 1,       "Mortgage": 0,       "CDAcc
