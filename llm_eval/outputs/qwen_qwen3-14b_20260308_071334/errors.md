# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 2
- Exact-match failures: 26

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 1x Missing top-level keys: missing_fields, notes
- 1x missing_fields must be an array of strings.
- 1x notes must be an array of strings.
- 1x Missing top-level keys: conflicts
- 1x conflicts must be an array of strings.

## Representative failed runs

### A02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see what the user is asking for here. They provided a case payload with input text that needs to be converted into a JSON object according to 
- final_message_excerpt: {"task":"extract_cf_request","status":"complete","cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_fields":[],"conflicts"

### B02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see. The user wants me to convert their input into a JSON object based on the given schema. The task is to extract a counterfactual request.\n
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "CDAccount": 1,     "Online": 1,     "SecuritiesAccount": 0,     "CreditCard": 1   },   "missing_fields": [],   "conflicts": [],   "notes": 

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see. The user wants to extract a counterfactual request from the input provided. The input is: \"I want online banking and a CD account, and m
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "needs_clarification",   "cf_request": {     "Family": 3,     "CDAccount": 1,     "Online": 1   },   "missing_fields": ["Income", "CCAvg", "Education", "Mortgage", "Securitie

### B07 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see. The user wants to extract a counterfactual request from their input. The input is: \"I want online banking and a CD account, and my famil
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "CDAccount": 1,     "Online": 1   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage",     "

### B07 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\n", "type": "reasoning"}, {"content": "\n\n{\"task\":\"extract_cf_request\",\"status\":\"partial\",\"cf_request\":{\"Family\":3,\"CDAccount\":1,\"Online\":
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Family":3,"CDAccount":1,"Online":1},"missing_fields":["Income","CCAvg","Education","Mortgage","SecuritiesAccount","CreditCard"],"conflicts":[],"notes":[]}

### B08 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this user request. The user provided a case payload with specific fields. My job is to convert that into the required JSON structure ba
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortgage",     "CDAccou

### B08 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's break down the user's input step by step. The task is to extract a counterfactual request into a JSON object based on the provided schema.\n\n
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortgage",     "CDAccou

### B08 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see. The user wants me to convert their input into a JSON object following the given schema. First, I need to parse the input they provided.\n
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortgage",     "CDAccou

### C01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\n", "type": "reasoning"}, {"content": "\n\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"conflict\",\n  \"cf_request\": {\n    \"Income\": 40,\n 
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "conflict",   "cf_request": {     "Income": 40,     "CDAccount": 1,     "Online": 1   },   "missing_fields": [     "CCAvg",     "Family",     "Education",     "Mortgage",    

### C01 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this user query. The input says, \"I want CDAccount yes and no at the same time. Set income to 40 and online to yes.\" \n\nFirst, I nee
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "conflict",   "cf_request": {     "Income": 40,     "Online": 1   },   "missing_fields": ["CCAvg", "Family", "Education", "Mortgage", "CDAccount", "SecuritiesAccount", "Credi
