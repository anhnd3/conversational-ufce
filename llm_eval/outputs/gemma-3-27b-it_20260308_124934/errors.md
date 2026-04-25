# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 39
- Exact-match failures: 39

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 30x CCAvg must be numeric.
- 30x Education must be an integer.
- 30x CDAccount must be binary 0 or 1.
- 30x SecuritiesAccount must be binary 0 or 1.
- 27x Family must be an integer.
- 27x Mortgage must be numeric.
- 27x CreditCard must be binary 0 or 1.
- 18x Online must be binary 0 or 1.
- 12x Income must be numeric.

## Representative failed runs

### B02 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Income must be numeric., CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "gemma-3-27b-it", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": null,\n    \"CCAvg\": null,\n    \"Family\
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": null,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": 1,     "Online":

### B02 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Income must be numeric., CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "gemma-3-27b-it", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": null,\n    \"CCAvg\": null,\n    \"Family\
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": null,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": 1,     "Online":

### B02 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Income must be numeric., CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "gemma-3-27b-it", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": null,\n    \"CCAvg\": null,\n    \"Family\
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": null,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": 1,     "Online":

### B03 repeat 1
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., CDAccount must be binary 0 or 1., CreditCard must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "gemma-3-27b-it", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 50.0,\n    \"CCAvg\": null,\n    \"Family\
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 50.0,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": 100.0,     "CDAccount": null,     "Onli

### B03 repeat 2
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., CDAccount must be binary 0 or 1., CreditCard must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "gemma-3-27b-it", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 50.0,\n    \"CCAvg\": null,\n    \"Family\
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 50.0,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": 100.0,     "CDAccount": null,     "Onli

### B03 repeat 3
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., CDAccount must be binary 0 or 1., CreditCard must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "gemma-3-27b-it", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 50.0,\n    \"CCAvg\": null,\n    \"Family\
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 50.0,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": 100.0,     "CDAccount": null,     "Onli

### B04 repeat 1
- api_error: none
- parse_error: none
- validation_errors: Family must be an integer., Education must be an integer., Mortgage must be numeric., CDAccount must be binary 0 or 1., SecuritiesAccount must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "gemma-3-27b-it", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 62.0,\n    \"CCAvg\": 2.7,\n    \"Family\"
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 62.0,     "CCAvg": 2.7,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": null,     "Online

### B04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: Family must be an integer., Education must be an integer., Mortgage must be numeric., CDAccount must be binary 0 or 1., SecuritiesAccount must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "gemma-3-27b-it", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 62.0,\n    \"CCAvg\": 2.7,\n    \"Family\"
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 62.0,     "CCAvg": 2.7,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": null,     "Online

### B04 repeat 3
- api_error: none
- parse_error: none
- validation_errors: Family must be an integer., Education must be an integer., Mortgage must be numeric., CDAccount must be binary 0 or 1., SecuritiesAccount must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "gemma-3-27b-it", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 62.0,\n    \"CCAvg\": 2.7,\n    \"Family\"
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 62.0,     "CCAvg": 2.7,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": null,     "Online

### B05 repeat 1
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric., CDAccount must be binary 0 or 1., Online must be binary 0 or 1., SecuritiesAccount must be binary 0 or 1., CreditCard must be binary 0 or 1.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "gemma-3-27b-it", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 45.0,\n    \"CCAvg\": null,\n    \"Family\
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 45.0,     "CCAvg": null,     "Family": null,     "Education": null,     "Mortgage": null,     "CDAccount": null,     "Onlin
