# Error Summary

- API errors: 0
- Invalid JSON runs: 5
- Schema-invalid runs: 61
- Exact-match failures: 64

## Top API errors

- none

## Top parse errors

- 3x Expecting property name enclosed in double quotes (line 1, column 2)
- 1x Expecting value (line 1, column 1)
- 1x Expecting property name enclosed in double quotes (line 1, column 3)

## Top validation errors

- 56x Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes
- 56x task must equal 'extract_cf_request'
- 56x status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- 56x cf_request must be an object.
- 56x missing_fields must be an array of strings.
- 56x conflicts must be an array of strings.
- 56x notes must be an array of strings.
- 46x Unexpected top-level keys: schema_reference
- 10x Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference
- 5x No parsed JSON object available.

## Representative failed runs

### A01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q6_k", "output": [{"content": "{  \n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income
- final_message_excerpt: {     "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,       "CD

### A01 repeat 2
- api_error: none
- parse_error: Expecting value (line 1, column 1)
- validation_errors: No parsed JSON object available.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q6_k", "output": [{"content": "<response>", "type": "message"}], "response_id": "resp_06c1b462493021d3ace906ee7cdef05d84eb833d1aefc382", "stats": {"input_tokens": 814, "r
- final_message_excerpt: <response>

### A01 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q6_k", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\"
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,       "CDAc

### A02 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q6_k", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\"
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,       "CDAc

### A02 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q6_k", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"partial\",\n    \"cf_request\": {\n      \"Income\":
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "partial",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,       "CDAcc

### A03 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q6_k", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\"
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 55,       "CCAvg": 2.2,       "Family": 4,       "Education": 3,       "Mortgage": 120,       "CDA

### A03 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q6_k", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\"
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 55,       "CCAvg": 2.2,       "Family": 4,       "Education": 3,       "Mortgage": 120,       "CDA

### A03 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q6_k", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\"
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 55,       "CCAvg": 2.2,       "Family": 4,       "Education": 3,       "Mortgage": 120,       "CDA

### A04 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q6_k", "output": [{"content": "```json\n{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \
- final_message_excerpt: ```json {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 65,       "CCAvg": 3.0,       "Family": 2,       "Education": 1,       "Mortgage": 0,     

### A04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "phi-4-reasoning-vision-15b@q6_k", "output": [{"content": "{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\"
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 65,       "CCAvg": 3.0,       "Family": 2,       "Education": 1,       "Mortgage": 0,       "CDAcc
