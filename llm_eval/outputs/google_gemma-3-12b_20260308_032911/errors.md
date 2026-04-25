# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 12
- Exact-match failures: 29

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 12x conflicts must be an array of strings.
- 3x Missing top-level keys: conflicts

## Representative failed runs

### A08 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "google/gemma-3-12b", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"complete\",\n  \"cf_request\": {\n    \"Income\": 30.0,\n    \"CCAvg\": 0.8,\n    \"Fam
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 30.0,     "CCAvg": 0.8,     "Family": 2,     "Education": 1,     "Mortgage": 0.0,     "CDAccount": 0,     "Online": 1,    

### A08 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "google/gemma-3-12b", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"complete\",\n  \"cf_request\": {\n    \"Income\": 30.0,\n    \"CCAvg\": 0.8,\n    \"Fam
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 30.0,     "CCAvg": 0.8,     "Family": 2,     "Education": 1,     "Mortgage": 0.0,     "CDAccount": 0,     "Online": 1,    

### B01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "google/gemma-3-12b", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"complete\",\n  \"cf_request\": {\n    \"Income\": 40.0,\n    \"CCAvg\": 1.5,\n    \"Fam
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 40.0,     "CCAvg": 1.5,     "Family": 3,     "Education": 2,     "Mortgage": 80.0   },   "missing_fields": [],   "conflict

### B01 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "google/gemma-3-12b", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"complete\",\n  \"cf_request\": {\n    \"Income\": 40.0,\n    \"CCAvg\": 1.5,\n    \"Fam
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 40.0,     "CCAvg": 1.5,     "Family": 3,     "Education": 2,     "Mortgage": 80.0   },   "missing_fields": [],   "conflict

### B01 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "google/gemma-3-12b", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"complete\",\n  \"cf_request\": {\n    \"Income\": 40.0,\n    \"CCAvg\": 1.5,\n    \"Fam
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "complete",   "cf_request": {     "Income": 40.0,     "CCAvg": 1.5,     "Family": 3,     "Education": 2,     "Mortgage": 80.0   },   "missing_fields": [],   "conflict

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "google/gemma-3-12b", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Family\": 3,\n    \"Online\": 1,\n    \"CDAccoun
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "Online": 1,     "CDAccount": 1   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage

### B07 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "google/gemma-3-12b", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Family\": 3,\n    \"Online\": 1,\n    \"CDAccoun
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "Online": 1,     "CDAccount": 1   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage

### B07 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "google/gemma-3-12b", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Family\": 3,\n    \"Online\": 1,\n    \"CDAccoun
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "Online": 1,     "CDAccount": 1   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage

### B08 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "google/gemma-3-12b", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 48.0,\n    \"CCAvg\": 1.2,\n    \"Fami
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48.0,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortgage",   

### B08 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "google/gemma-3-12b", "output": [{"content": "```json\n{\n  \"task\": \"extract_cf_request\",\n  \"status\": \"partial\",\n  \"cf_request\": {\n    \"Income\": 48.0,\n    \"CCAvg\": 1.2,\n    \"Fami
- final_message_excerpt: ```json {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48.0,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortgage",
