# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 7
- Exact-match failures: 27

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 4x notes must be an array of strings.
- 3x conflicts must be an array of strings.
- 3x Missing top-level keys: notes
- 2x Missing top-level keys: conflicts
- 1x Missing top-level keys: conflicts, notes
- 1x Missing top-level keys: missing_fields
- 1x missing_fields must be an array of strings.

## Representative failed runs

### B03 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-14b-128k", "output": [{"content": "\nOkay, let's tackle this user query. The input is: \"Please set income to 50, mortgage to 100, and make online banking available. No securities account.\" 
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 50,     "Mortgage": 100,     "Online": 1,     "SecuritiesAccount": 0   },   "missing_fields": [     "CCAvg",     "Family",     "Edu

### B03 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-14b-128k", "output": [{"content": "\nOkay, let's tackle this user request. They want me to convert their natural language into a JSON object following the given schema.\n\nFirst, I need to pa
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 50,     "Mortgage": 100,     "Online": 1,     "SecuritiesAccount": 0   },   "missing_fields": ["Family", "Education", "CCAvg", "CDA

### B04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: conflicts, notes, conflicts must be an array of strings., notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-14b-128k", "output": [{"content": "\nOkay, let's see. The user wants me to convert their input into a JSON object based on the given schema. The task is to extract a counterfactual request.\n
- final_message_excerpt: {"task": "extract_cf_request", "status": "partial", "cf_request": {"Income": 62, "CCAvg": 2.7, "CreditCard": 1, "Online": 1}, "missing_fields": ["Family", "Education", "Mortgage", "CDAccount", "SecuritiesAccount"]}

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-14b-128k", "output": [{"content": "\nOkay, let's break down the user's input to create the JSON object as per the schema.\n\nThe user provided the case payload with input: \"I want online ban
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "CDAccount": 1,     "Online": 1   },   "missing_fields": ["Income", "CCAvg", "Education", "Mortgage", "SecuritiesAccount", "

### B07 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-14b-128k", "output": [{"content": "\nOkay, let's tackle this user query. The input is: \"I want online banking and a CD account, and my family size is 3. My mortgage should stay low.\" \n\nFi
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "CDAccount": 1,     "Online": 1   },   "missing_fields": ["Income", "CCAvg", "Education", "Mortgage", "SecuritiesAccount", "

### B07 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-14b-128k", "output": [{"content": "\nOkay, let's see. The user wants to convert their input into a JSON object based on the given schema. First, I need to extract the information from the cas
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Online": 1,     "CDAccount": 1,     "Family": 3   },   "missing_fields": ["Income", "CCAvg", "Mortgage", "Education", "SecuritiesAccount", "

### B08 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: conflicts, conflicts must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-14b-128k", "output": [{"content": "\nOkay, let me start by looking at the user's input. They mentioned wanting income 48, which is a number, so that maps to \"Income\": 48.0. Then they said \
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48.0,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": ["Education", "Mortgage", "CDAccount", "Onlin

### B08 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-14b-128k", "output": [{"content": "\nOkay, let's see what the user is asking for here. They provided a case payload with specific input fields and want me to convert that into a JSON object b
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "Online": 1,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortga

### B08 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: conflicts, conflicts must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-14b-128k", "output": [{"content": "\nOkay, let me process this user request step by step. The user wants to create a JSON object based on the provided schema. First, I need to check which fie
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": ["Education", "Mortgage", "CDAccount", "Online"

### C01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-14b-128k", "output": [{"content": "\nOkay, let's see what the user is asking here. They provided a case payload where the input says, \"I want CDAccount yes and no at the same time. Set incom
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "conflict",   "cf_request": {     "Income": 40,     "Online": 1   },   "missing_fields": ["CCAvg", "Family", "Education", "Mortgage", "CDAccount", "SecuritiesAccount", "Credi
