# Error Summary

- API errors: 2
- Invalid JSON runs: 2
- Schema-invalid runs: 3
- Exact-match failures: 30

## Top API errors

- 2x ReadTimeout: HTTPConnectionPool(host='localhost', port=1234): Read timed out. (read timeout=300.0)

## Top parse errors

- none

## Top validation errors

- 1x Missing top-level keys: notes
- 1x notes must be an array of strings.

## Representative failed runs

### A01 repeat 3
- api_error: ReadTimeout: HTTPConnectionPool(host='localhost', port=1234): Read timed out. (read timeout=300.0)
- parse_error: none
- validation_errors: none
- exact_match: False
- final_message_excerpt: (empty)

### A02 repeat 2
- api_error: ReadTimeout: HTTPConnectionPool(host='localhost', port=1234): Read timed out. (read timeout=300.0)
- parse_error: none
- validation_errors: none
- exact_match: False
- final_message_excerpt: (empty)

### A03 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this user query. The user wants me to convert their input into a JSON object based on the provided schema.\n\nFirst, I need to parse th
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 55,     "CCAvg": 2.2,     "Family": 4,     "Education": 3,     "Mortgage": 120,     "CDAccount": 0,     "Online": 0,     "Securiti

### B03 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see. The user wants me to extract a counterfactual request from their input. The input is: \"Please set income to 50, mortgage to 100, and mak
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Income":50,"Mortgage":100,"Online":1,"SecuritiesAccount":0},"missing_fields":["CCAvg","Family","Education","CDAccount","CreditCard"],"conflicts":[],"notes":[

### B03 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see. The user wants me to convert their input into a JSON object based on the given schema. First, I need to parse the input carefully.\n\nThe
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 50,     "Mortgage": 100,     "Online": 1,     "SecuritiesAccount": 0   },   "missing_fields": [     "CDAccount",     "Family",     

### B06 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see. The user wants me to extract a counterfactual request from their input. The task is to convert the natural language into a JSON object ba
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 4,     "Education": 2,     "CDAccount": 1   },   "missing_fields": [     "Income",     "CCAvg",     "Mortgage",     "Online",     "

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see what the user needs here. They provided a case payload with input text and want me to extract the counterfactual request into JSON accordi
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Online": 1,     "CDAccount": 1,     "Family": 3   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage",     "

### B07 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this user request. The user wants to generate a JSON object based on the provided schema. First, I need to parse the input carefully.\n
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Online": 1,     "CDAccount": 1,     "Family": 3   },   "missing_fields": [     "Income",     "CCAvg",     "Mortgage",     "Education"   ],  

### B07 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this user request. The user wants to create a counterfactual request based on their input. First, I need to parse the input sentence ca
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "needs_clarification",   "cf_request": {     "CDAccount": 1,     "Online": 1,     "Family": 3   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mort

### B08 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see what the user is asking for here. They provided a case payload with specific input values. My job is to convert that into the JSON object 
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "needs_clarification",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Mortgage",     "CDAccount",
