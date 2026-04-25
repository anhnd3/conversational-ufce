# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 14
- Exact-match failures: 31

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 12x Family must be an integer.
- 11x Education must be an integer.
- 9x CCAvg must be numeric.
- 9x CDAccount must be binary 0 or 1.
- 9x SecuritiesAccount must be binary 0 or 1.
- 8x Mortgage must be numeric.
- 8x Online must be binary 0 or 1.
- 7x CreditCard must be binary 0 or 1.
- 6x Income must be numeric.

## Representative failed runs

### B01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "mistralai/ministral-3-14b-reasoning", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"complete\",\n  \"cf_request\": {\n    \"Income\": 40,\n    \"CCAvg\": 
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 40,     "CCAvg": 1.5,     "Family": 3,     "Education": 2,     "Mortgage": 80   },   "missing_fields": [     "CDAccount", 

### B01 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "mistralai/ministral-3-14b-reasoning", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"complete\",\n  \"cf_request\": {\n    \"Income\": 40,\n    \"CCAvg\": 
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 40,     "CCAvg": 1.5,     "Family": 3,     "Education": 2,     "Mortgage": 80   },   "missing_fields": [     "CDAccount", 

### B01 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "mistralai/ministral-3-14b-reasoning", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"complete\",\n  \"cf_request\": {\n    \"Income\": 40,\n    \"CCAvg\": 
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 40,     "CCAvg": 1.5,     "Family": 3,     "Education": 2,     "Mortgage": 80   },   "missing_fields": [     "CDAccount", 

### B02 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "mistralai/ministral-3-14b-reasoning", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"complete\",\n  \"cf_request\": {\n    \"CDAccount\": 1,\n    \"Online\
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "CDAccount": 1,     "Online": 1,     "SecuritiesAccount": 0,     "CreditCard": 1   },   "missing_fields": [     "Income",     "CCAvg

### B05 repeat 1
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric., CDAccount must be binary 0 or 1., Online must be binary 0 or 1., SecuritiesAccount must be binary 0 or 1., CreditCard must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "mistralai/ministral-3-14b-reasoning", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 45.0,\n    \"CCAvg\":
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 45.0,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": null,     "Onlin

### B05 repeat 2
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric., CDAccount must be binary 0 or 1., Online must be binary 0 or 1., SecuritiesAccount must be binary 0 or 1., CreditCard must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "mistralai/ministral-3-14b-reasoning", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 45.0,\n    \"CCAvg\":
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 45.0,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": null,     "Onlin

### B05 repeat 3
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric., CDAccount must be binary 0 or 1., Online must be binary 0 or 1., SecuritiesAccount must be binary 0 or 1., CreditCard must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "mistralai/ministral-3-14b-reasoning", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 45.0,\n    \"CCAvg\":
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 45.0,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": null,     "Onlin

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "mistralai/ministral-3-14b-reasoning", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Family\": 3,\n    \"Online\": 1
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "Online": 1,     "CDAccount": 1,     "Mortgage": 0.0   },   "missing_fields": [     "Income",     "CCAvg",     "Educ

### B07 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "mistralai/ministral-3-14b-reasoning", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Family\": 3,\n    \"Online\": 1
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "Online": 1,     "CDAccount": 1,     "Mortgage": 0.0   },   "missing_fields": [     "Income",     "CCAvg",     "Educ

### B07 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "mistralai/ministral-3-14b-reasoning", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Family\": 3,\n    \"Online\": 1
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "Online": 1,     "CDAccount": 1,     "Mortgage": 0.0   },   "missing_fields": [     "Income",     "CCAvg",     "Educ
