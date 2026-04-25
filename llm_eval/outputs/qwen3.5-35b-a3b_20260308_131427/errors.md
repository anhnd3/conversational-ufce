# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 0
- Exact-match failures: 26

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- none

## Representative failed runs

### B03 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3.5-35b-a3b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A user request (\"Please set income to 50, mortgage to 100, and make online banking avail
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 50,     "Mortgage": 100,     "Online": 1,     "SecuritiesAccount": 0   },   "missing_fields": [     "CCAvg",     "CDAccount",     "

### B06 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3.5-35b-a3b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A user request string (\"Use family size 4 and education 2. I want CD account yes. I am n
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 4,     "Education": 2,     "CDAccount": 1   },   "missing_fields": [     "Income",     "Mortgage",     "CCAvg",     "Online",     "

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3.5-35b-a3b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A user request (\"I want online banking and a CD account, and my family size is 3. My mor
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "CDAccount": 1,     "Online": 1   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage",     "

### B07 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3.5-35b-a3b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A user request (\"I want online banking and a CD account, and my family size is 3. My mor
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "CDAccount": 1,     "Online": 1   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage",     "

### B07 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3.5-35b-a3b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A user request (\"I want online banking and a CD account, and my family size is 3. My mor
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Family": 3,     "CDAccount": 1,     "Online": 1   },   "missing_fields": [     "Income",     "CCAvg",     "Education",     "Mortgage",     "

### B08 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3.5-35b-a3b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A natural language request (\"I want income 48, card average 1.2, family 2, and I studied
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortgage",     "CDAccou

### B08 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3.5-35b-a3b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A user request string (\"I want income 48, card average 1.2, family 2, and I studied at t
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "needs_clarification",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortgage", 

### B08 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3.5-35b-a3b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A natural language request (\"I want income 48, card average 1.2, family 2, and I studied
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "partial",   "cf_request": {     "Income": 48,     "CCAvg": 1.2,     "Family": 2,     "CreditCard": 1   },   "missing_fields": [     "Education",     "Mortgage",     "CDAccou

### C01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3.5-35b-a3b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A user request string (\"I want CDAccount yes and no at the same time. Set income to 40 a
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "conflict",   "cf_request": {     "Income": 40,     "Online": 1   },   "missing_fields": [     "CCAvg",     "Family",     "Education",     "Mortgage",     "SecuritiesAccount"

### C01 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3.5-35b-a3b", "output": [{"content": "Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: A user request string (\"I want CDAccount yes and no at the same time. Set income to 40 a
- final_message_excerpt: {   "task": "extract_cf_request",   "status": "conflict",   "cf_request": {     "Income": 40,     "Online": 1   },   "missing_fields": [],   "conflicts": [     "CDAccount requested as both yes and no simultaneously"   ],
