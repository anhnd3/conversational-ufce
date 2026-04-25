# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 51
- Exact-match failures: 56

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 46x conflicts must be an array of strings.
- 46x notes must be an array of strings.
- 45x status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- 44x missing_fields must be an array of strings.
- 43x task must equal 'extract_cf_request'
- 43x cf_request must be an object.
- 42x Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes
- 38x Unexpected top-level keys: schema_reference
- 4x Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference
- 2x Missing top-level keys: conflicts, notes

## Representative failed runs

### A01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "glm-z1-9b-0414", "output": [{"content": "Okay, let's tackle this step by step. The user provided a natural language request that needs to be converted into the specified JSON format.\n\nFirst, I'll
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": [       "complete",       "partial",       "needs_clarification",       "conflict"     ],     "cf_request": {       "Income": 40,       "CCAvg": 1

### A01 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "glm-z1-9b-0414", "output": [{"content": "Okay, let's tackle this step by step. The user wants me to convert their natural-language request into a specific JSON object following the given schema.\n\
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": [       "complete",       "partial",       "needs_clarification",       "conflict"     ],     "cf_request": {       "Income": 40,       "CCAvg": 1

### A01 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "glm-z1-9b-0414", "output": [{"content": "Okay, let's tackle this step by step. The user provided a case payload where they want to extract a counterfactual request according to the given schema.\n\
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": [       "complete",       "partial",       "needs_clarification",       "conflict"     ],     "cf_request": {       "Income": 40.0,       "CCAvg":

### A02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "glm-z1-9b-0414", "output": [{"content": "Okay, let's tackle this step by step. The user wants me to convert their natural-language request into a JSON object following specific rules.\n\nFirst, I'l
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": [       "complete",       "partial",       "needs_clarification",       "conflict"     ],     "cf_request": {       "Income": 40,       "CCAvg": 1

### A02 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "glm-z1-9b-0414", "output": [{"content": "Okay, let me tackle this step by step. The user wants me to convert their natural-language request into a JSON object following specific rules.\n\nFirst, th
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": [       "complete",       "partial",       "needs_clarification",       "conflict"     ],     "cf_request": {       "Income": 40,       "CCAvg": 1

### A03 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "glm-z1-9b-0414", "output": [{"content": "Okay, let's start by breaking down the user's request. The task is to convert their natural language input into a specific JSON object following the given s
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": [       "complete",       "partial",       "needs_clarification",       "conflict"     ],     "cf_request": {       "Income": 55,       "CCAvg": 2

### A04 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "glm-z1-9b-0414", "output": [{"content": "Okay, let's tackle this problem step by step. The user provided a natural-language request and wants me to convert it into a specific JSON structure followi
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": [       "complete",       "partial",       "needs_clarification",       "conflict"     ],     "cf_request": {       "Income": 65,       "CCAvg": 3

### A04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "glm-z1-9b-0414", "output": [{"content": "Okay, let's tackle this query step by step. The user wants me to convert their natural-language request into a JSON object following specific rules.\n\nFirs
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": ["complete", "partial", "needs_clarification", "conflict"],     "cf_request": {       "Income": 65,       "CCAvg": 3.0,       "Family": 2,       "

### A05 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "glm-z1-9b-0414", "output": [{"content": "Okay, let's tackle this user request step by step. The user wants me to convert their natural-language query into a specific JSON object following the given
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": ["complete", "partial", "needs_clarification", "conflict"],     "cf_request": {       "Income": 40.0,       "CCAvg": 1.5,       "Family": 3,      

### A05 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "glm-z1-9b-0414", "output": [{"content": "Okay, let's tackle this step by step. The user provided a case payload and wants me to generate the correct JSON according to the schema. First, I need to p
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": [       "complete",       "partial",       "needs_clarification",       "conflict"     ],     "cf_request": {       "Income": 40.0,       "CCAvg":
