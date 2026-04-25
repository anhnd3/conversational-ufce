# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 1
- Exact-match failures: 26

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 1x Missing top-level keys: notes
- 1x notes must be an array of strings.

## Representative failed runs

### A07 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this user request step by step. The user wants to convert their natural language input into a JSON object according to the provided sch
- final_message_excerpt: {"task":"extract_cf_request","status":"complete","cf_request":{"Income":90,"CCAvg":5.5,"Family":5,"Education":3,"Mortgage":250,"CDAccount":0,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fields":[],"conflicts

### B05 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see. The user provided a case where the input is \"The only thing I know is that my target income should be 45.\" So, the task is to extract a
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Income":45},"missing_fields":["CCAvg","Family","Education","Mortgage","CDAccount","Online","SecuritiesAccount","CreditCard"],"conflicts":[],"notes":["Only In

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this query. The user wants to convert their input into a JSON object following the given schema. First, I need to parse the input: \"I 
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "Online": 1,     "CDAccount": 1   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage"   ],  

### B07 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this query step by step. The user wants to convert their input into a JSON object based on the given schema.\n\nFirst, I need to parse 
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Online": 1,     "CDAccount": 1,     "Family": 3   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage",     "

### B07 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's process this user input step by step. The task is to extract the counterfactual request into the specified JSON format.\n\nFirst, I need to id
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Online": 1,     "CDAccount": 1,     "Family": 3   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage",     "

### B08 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's process this user input step by step. The user provided a case with specific details, and I need to extract the counterfactual request into th
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education"   ],   "conflicts": [],   "no

### B08 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's tackle this user request step by step. The user wants me to convert their natural language input into a JSON object following specific rules. 
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Income":48,"CCAvg":1.2,"Family":2,"CreditCard":1},"missing_fields":["Education","Mortgage","CDAccount","Online","SecuritiesAccount"],"conflicts":[],"notes":[

### B08 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's process this user input step by step. The user provided a case payload with specific details. First, I need to extract the information from th
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortgage",     "CDAccou

### C01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see what the user is asking for here. The input says, \"I want CDAccount yes and no at the same time. Set income to 40 and online to yes.\" \n
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "conflict",   "cf_request": {     "Income": 40,     "Online": 1   },   "missing_fields": [     "CCAvg",     "Family",     "Education",     "Mortgage",     "CDAccount",     "S

### C01 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3-14b", "output": [{"content": "\nOkay, let's see what the user is asking for here. The input says, \"I want CDAccount yes and no at the same time. Set income to 40 and online to yes.\" \n
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "conflict",   "cf_request": {     "Income": 40,     "Online": 1   },   "missing_fields": [     "CCAvg",     "Family",     "Education",     "Mortgage",     "CDAccount",     "S
