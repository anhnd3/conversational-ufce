# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 66
- Exact-match failures: 66

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 66x Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes
- 66x task must equal 'extract_cf_request'
- 66x status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- 66x cf_request must be an object.
- 66x missing_fields must be an array of strings.
- 66x conflicts must be an array of strings.
- 66x notes must be an array of strings.
- 58x Unexpected top-level keys: schema_reference
- 8x Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference

## Representative failed runs

### A01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "baidu/ernie-4.5-21b-a3b", "output": [{"content": "```json\n{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\
- final_message_excerpt: ```json {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40.0,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80.0,

### A01 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "baidu/ernie-4.5-21b-a3b", "output": [{"content": "```json\n{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\
- final_message_excerpt: ```json {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,    

### A01 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "baidu/ernie-4.5-21b-a3b", "output": [{"content": "```json\n{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\
- final_message_excerpt: ```json {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,    

### A02 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "baidu/ernie-4.5-21b-a3b", "output": [{"content": "```json\n{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\
- final_message_excerpt: ```json {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40.0,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": null,

### A02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "baidu/ernie-4.5-21b-a3b", "output": [{"content": "```json\n{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"partial\",\n    \"cf_request\": {\n      \"Income\"
- final_message_excerpt: ```json {   "schema_reference": {     "task": "extract_cf_request",     "status": "partial",     "cf_request": {       "Income": null,       "CCAvg": null,       "Family": 3,       "Education": 2,       "Mortgage": null,

### A02 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "baidu/ernie-4.5-21b-a3b", "output": [{"content": "```json\n{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\
- final_message_excerpt: ```json {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40.0,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": null,

### A03 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "baidu/ernie-4.5-21b-a3b", "output": [{"content": "```json\n{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\
- final_message_excerpt: ```json {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 55,       "CCAvg": 2.2,       "Family": 4,       "Education": 3,       "Mortgage": 120,   

### A03 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "baidu/ernie-4.5-21b-a3b", "output": [{"content": "```json\n{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\
- final_message_excerpt: ```json {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 55,       "CCAvg": 2.2,       "Family": 4,       "Education": 3,       "Mortgage": 120,   

### A03 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "baidu/ernie-4.5-21b-a3b", "output": [{"content": "```json\n{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\
- final_message_excerpt: ```json {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 55,       "CCAvg": 2.2,       "Family": 4,       "Education": 3,       "Mortgage": 120,   

### A04 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "baidu/ernie-4.5-21b-a3b", "output": [{"content": "```json\n{\n  \"schema_reference\": {\n    \"task\": \"extract_cf_request\",\n    \"status\": \"complete\",\n    \"cf_request\": {\n      \"Income\
- final_message_excerpt: ```json {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 65.0,       "CCAvg": 3.0,       "Family": 2,       "Education": 1,       "Mortgage": 0.0,
