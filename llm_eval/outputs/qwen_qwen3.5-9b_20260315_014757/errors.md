# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 13
- Exact-match failures: 36

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 12x Education must be an integer.
- 11x Mortgage must be numeric.
- 11x SecuritiesAccount must be binary 0 or 1.
- 9x CCAvg must be numeric.
- 9x CDAccount must be binary 0 or 1.
- 8x Family must be an integer.
- 7x CreditCard must be binary 0 or 1.
- 5x Online must be binary 0 or 1.
- 4x Income must be numeric.

## Representative failed runs

### A01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user wants me to extract a counterfactual request from the natural language input and return it in the specified JSON format.\n\nLooking at the input: 
- final_message_excerpt: {"task":"extract_cf_request","status":"complete","cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},"missing_fields":[],"conflicts"

### A05 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user wants me to extract counterfactual request information from the natural language input into a JSON object following the specified schema.\n\nLet m
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 40.0,     "CCAvg": 1.5,     "Family": 3,     "Education": 2,     "Mortgage": 80.0,     "CDAccount": 1,     "Online": 1,     "Secur

### B02 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Income must be numeric., CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user wants me to extract information from a natural language request into a JSON object following a specific schema.\n\nLooking at the input:\n\"I want
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": null,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": 1,     "Online": 1,     

### B02 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A natural language request from a user (\"I want CD account yes, online yes, securities a
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "CDAccount": 1,     "Online": 1,     "SecuritiesAccount": 0,     "CreditCard": 1   },   "missing_fields": [     "Income",     "CCAvg",     "F

### B03 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A natural language request (\"Please set income to 50, mortgage to 100, and make online b
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 50,     "Mortgage": 100,     "Online": 1,     "SecuritiesAccount": 0   },   "missing_fields": [     "CCAvg",     "Family",     "Edu

### B04 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Family must be an integer., Education must be an integer., Mortgage must be numeric., CDAccount must be binary 0 or 1., SecuritiesAccount must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user wants me to extract a counterfactual request from the given input text.\n\nInput: \"I can aim for income 62 and card spending around 2.7. I also w
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 62,     "CCAvg": 2.7,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": null,     "Online": 1,     

### B04 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Family must be an integer., Education must be an integer., Mortgage must be numeric., CDAccount must be binary 0 or 1., SecuritiesAccount must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user wants me to extract counterfactual request information from this natural language input.\n\nLet me analyze the input:\n\"I can aim for income 62 a
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 62,     "CCAvg": 2.7,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": null,     "Online": 1,     

### B05 repeat 2
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric., CDAccount must be binary 0 or 1., Online must be binary 0 or 1., SecuritiesAccount must be binary 0 or 1., CreditCard must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user wants me to extract a counterfactual request from natural language text into a JSON object following the schema.\n\nLooking at the input: \"The on
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 45,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": null,     "Online": null, 

### B05 repeat 3
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric., CDAccount must be binary 0 or 1., Online must be binary 0 or 1., SecuritiesAccount must be binary 0 or 1., CreditCard must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "The user has provided a natural language request about their target income being 45.\n\nLooking at the schema, I need to extract:\n- Income (number) - ment
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 45,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": null,     "Online": null, 

### B06 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen/qwen3.5-9b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A natural language request from a user (\"Use family size 4 and education 2. I want CD ac
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 4,     "Education": 2,     "CDAccount": 1   },   "missing_fields": [     "Income",     "CCAvg",     "Mortgage",     "Online",     "
