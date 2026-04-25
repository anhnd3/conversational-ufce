# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 28
- Exact-match failures: 45

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 25x Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes
- 25x task must equal 'extract_cf_request'
- 25x status must be one of 'complete', 'partial', 'needs_clarification', 'conflict'
- 25x cf_request must be an object.
- 25x missing_fields must be an array of strings.
- 25x conflicts must be an array of strings.
- 25x notes must be an array of strings.
- 13x Unexpected top-level keys: schema_reference
- 12x Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference
- 3x Income must be numeric.

## Representative failed runs

### A01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user wants me to convert their natural language request into a JSON object according to the schema.\n\nLet me analyze the input:\n- \"My target profile
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 40,     "CCAvg": 1.5,     "Family": 3,     "Education": 2,     "Mortgage": 80,     "CDAccount": 1,     "Online": 1,     "Securitie

### A01 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user wants me to extract a counterfactual request from the natural language input and convert it into a JSON object that matches the provided schema.\n
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,       "CDAc

### A02 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "Looking at this input, I need to extract a counterfactual request from the natural language statement.\n\nLet me parse the input:\n- \"I want a mortgage of
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80,       "CDAc

### A04 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Role: Structured extraction assistant for bank counterfactual requests.\n    *   Input: A JSON o
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 65,       "CCAvg": 3.0,       "Family": 2,       "Education": 1,       "Mortgage": 0,       "CDAcc

### A04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "Okay, I need to convert the user's request into a JSON object according to the schema provided. Let me start by understanding the input. The user says:\n\n
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 65,       "CCAvg": 3.0,       "Family": 2,       "Education": 1,       "Mortgage": 0,       "CDAcc

### A04 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "Looking at the case payload, I need to extract a counterfactual request from the user's natural language request.\n\nThe input says:\n- \"Target outcome: I
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 65,       "CCAvg": 3.0,       "Family": 2,       "Education": 1,       "Mortgage": 0,       "CDAcc

### A05 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user wants me to extract counterfactual request information from their natural language input and convert it into the specified JSON format.\n\nLet me 
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 40.0,       "CCAvg": 1.5,       "Family": 3,       "Education": 2,       "Mortgage": 80.0,       "

### A06 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "Looking at this case payload, I need to extract a counterfactual request from the input string.\n\nThe input contains:\n- Income=72\n- CCAvg=4.8\n- Family=
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 72,       "CCAvg": 4.8,       "Family": 1,       "Education": 2,       "Mortgage": 200,       "CDA

### A06 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: allowed_status_values, feature_dictionary, schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user wants me to extract a counterfactual request from the given input. Let me analyze the input:\n\n\"Income=72; CCAvg=4.8; Family=1; Education=2; Mor
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 72,       "CCAvg": 4.8,       "Family": 1,       "Education": 2,       "Mortgage": 200,       "CDA

### A08 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: task, status, cf_request, missing_fields, conflicts, notes, Unexpected top-level keys: schema_reference, task must equal 'extract_cf_request', status must be one of 'complete', 'partial', 'needs_clarification', 'conflict', cf_request must be an object., missing_fields must be an array of strings., conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user wants me to extract a counterfactual request from the given input text. Let me analyze the input:\n\n\"Set my requested profile to income 30, CCAv
- final_message_excerpt: {   "schema_reference": {     "task": "extract_cf_request",     "status": "complete",     "cf_request": {       "Income": 30,       "CCAvg": 0.8,       "Family": 2,       "Education": 1,       "Mortgage": 0,       "CDAcc
