# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 2
- Exact-match failures: 25

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 2x Missing top-level keys: notes
- 2x notes must be an array of strings.

## Representative failed runs

### B02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see. The user wants me to extract a counterfactual request from their input. The input is: \"I want CD account yes, online yes, securities acc
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "needs_clarification",   "cf_request": {     "CDAccount": 1,     "Online": 1,     "SecuritiesAccount": 0,     "CreditCard": 1   },   "missing_fields": [     "Income",     "CC

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this query step by step. The user wants to generate a JSON object based on the provided schema and case payload.\n\nFirst, I need to pa
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "needs_clarification",   "cf_request": {     "Online": 1,     "CDAccount": 1,     "Family": 3   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mort

### B07 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this user query. The user wants to create a counterfactual request based on their input. First, I need to parse the input carefully.\n\
- final_message_excerpt: {"task": "extract_cf_request", "status": "partial", "cf_request": {"Family": 3, "CDAccount": 1, "Online": 1}, "missing_fields": ["Income", "CCAvg", "Education", "Mortgage", "SecuritiesAccount", "CreditCard"], "conflicts"

### B07 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see. The user wants me to convert their input into a JSON object based on the given schema. Let me start by looking at the case payload.\n\nTh
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Online": 1,     "CDAccount": 1,     "Family": 3   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage",     "

### B08 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this user query. The user wants me to convert their natural language request into a JSON object based on the provided schema.\n\nFirst,
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education"   ],   "conflicts": [],   "no

### B08 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's process this user input step by step. The user wants to create a counterfactual request based on the provided schema. First, I need to parse t
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "needs_clarification",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortgage", 

### B08 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see what the user is asking for here. They provided a case payload with specific input details. My job is to convert that into a JSON object b
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortgage",     "CDAccou

### C01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this user request. The user wants a JSON object based on the schema provided. First, I need to parse their input: \"I want CDAccount ye
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "conflict",   "cf_request": {     "Income": 40,     "Online": 1   },   "missing_fields": [     "CDAccount",     "CCAvg",     "Family",     "Education",     "Mortgage",     "S

### C01 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Missing top-level keys: notes, notes must be an array of strings.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see what the user is asking for here. They provided a case payload where they want both CDAccount to be yes and no at the same time, which see
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "conflict",   "cf_request": {     "Income": 40,     "Online": 1   },   "missing_fields": [     "CCAvg",     "Family",     "Education",     "Mortgage",     "CDAccount",     "S

### C01 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see what the user is asking for here. They provided a case payload with a specific input: \"I want CDAccount yes and no at the same time. Set 
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "conflict",   "cf_request": {     "Income": 40,     "Online": 1   },   "missing_fields": [     "CCAvg",     "Family",     "Education",     "Mortgage",     "CDAccount",     "S
