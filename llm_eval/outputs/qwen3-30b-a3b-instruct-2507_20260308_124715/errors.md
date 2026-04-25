# Error Summary

- API errors: 0
- Invalid JSON runs: 0
- Schema-invalid runs: 12
- Exact-match failures: 33

## Top API errors

- none

## Top parse errors

- none

## Top validation errors

- 12x Family must be an integer.
- 10x CCAvg must be numeric.
- 10x Education must be an integer.
- 9x Mortgage must be numeric.
- 4x SecuritiesAccount must be binary 0 or 1.
- 4x CreditCard must be binary 0 or 1.
- 4x Income must be numeric.
- 3x CDAccount must be binary 0 or 1.
- 1x Online must be binary 0 or 1.

## Representative failed runs

### B01 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-30b-a3b-instruct-2507", "output": [{"content": "{\"task\":\"extract_cf_request\",\"status\":\"partial\",\"cf_request\":{\"Income\":40,\"CCAvg\":1.5,\"Family\":3,\"Education\":2,\"Mortgage\":8
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":0,"Online":0,"SecuritiesAccount":0,"CreditCard":0},"missing_fields":["CDAccount","O

### B01 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-30b-a3b-instruct-2507", "output": [{"content": "{\"task\":\"extract_cf_request\",\"status\":\"partial\",\"cf_request\":{\"Income\":40,\"CCAvg\":1.5,\"Family\":3,\"Education\":2,\"Mortgage\":8
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80,"CDAccount":0,"Online":0,"SecuritiesAccount":0,"CreditCard":0},"missing_fields":["CDAccount","O

### B03 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-30b-a3b-instruct-2507", "output": [{"content": "{\"task\":\"extract_cf_request\",\"status\":\"complete\",\"cf_request\":{\"Income\":50,\"CCAvg\":0,\"Family\":0,\"Education\":0,\"Mortgage\":10
- final_message_excerpt: {"task":"extract_cf_request","status":"complete","cf_request":{"Income":50,"CCAvg":0,"Family":0,"Education":0,"Mortgage":100,"CDAccount":0,"Online":1,"SecuritiesAccount":0,"CreditCard":0},"missing_fields":[],"conflicts":

### B04 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-30b-a3b-instruct-2507", "output": [{"content": "{\"task\":\"extract_cf_request\",\"status\":\"complete\",\"cf_request\":{\"Income\":62,\"CCAvg\":2.7,\"Family\":0,\"Education\":0,\"Mortgage\":
- final_message_excerpt: {"task":"extract_cf_request","status":"complete","cf_request":{"Income":62,"CCAvg":2.7,"Family":0,"Education":0,"Mortgage":0,"CDAccount":0,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fields":[],"conflicts":

### B04 repeat 2
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-30b-a3b-instruct-2507", "output": [{"content": "{\"task\":\"extract_cf_request\",\"status\":\"complete\",\"cf_request\":{\"Income\":62,\"CCAvg\":2.7,\"Family\":0,\"Education\":0,\"Mortgage\":
- final_message_excerpt: {"task":"extract_cf_request","status":"complete","cf_request":{"Income":62,"CCAvg":2.7,"Family":0,"Education":0,"Mortgage":0,"CDAccount":0,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fields":[],"conflicts":

### B04 repeat 3
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-30b-a3b-instruct-2507", "output": [{"content": "{\"task\":\"extract_cf_request\",\"status\":\"complete\",\"cf_request\":{\"Income\":62,\"CCAvg\":2.7,\"Family\":0,\"Education\":0,\"Mortgage\":
- final_message_excerpt: {"task":"extract_cf_request","status":"complete","cf_request":{"Income":62,"CCAvg":2.7,"Family":0,"Education":0,"Mortgage":0,"CDAccount":0,"Online":1,"SecuritiesAccount":0,"CreditCard":1},"missing_fields":[],"conflicts":

### B05 repeat 1
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-30b-a3b-instruct-2507", "output": [{"content": "{\"task\":\"extract_cf_request\",\"status\":\"partial\",\"cf_request\":{\"Income\":45,\"CCAvg\":null,\"Family\":null,\"Education\":null,\"Mortg
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Income":45,"CCAvg":null,"Family":null,"Education":null,"Mortgage":null,"CDAccount":0,"Online":0,"SecuritiesAccount":0,"CreditCard":0},"missing_fields":["CCAv

### B05 repeat 2
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-30b-a3b-instruct-2507", "output": [{"content": "{\"task\":\"extract_cf_request\",\"status\":\"partial\",\"cf_request\":{\"Income\":45,\"CCAvg\":null,\"Family\":null,\"Education\":null,\"Mortg
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Income":45,"CCAvg":null,"Family":null,"Education":null,"Mortgage":null,"CDAccount":0,"Online":0,"SecuritiesAccount":0,"CreditCard":0},"missing_fields":["CCAv

### B05 repeat 3
- api_error: none
- parse_error: none
- validation_errors: CCAvg must be numeric., Family must be an integer., Education must be an integer., Mortgage must be numeric.
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-30b-a3b-instruct-2507", "output": [{"content": "{\"task\":\"extract_cf_request\",\"status\":\"partial\",\"cf_request\":{\"Income\":45,\"CCAvg\":null,\"Family\":null,\"Education\":null,\"Mortg
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Income":45,"CCAvg":null,"Family":null,"Education":null,"Mortgage":null,"CDAccount":0,"Online":0,"SecuritiesAccount":0,"CreditCard":0},"missing_fields":["CCAv

### B07 repeat 1
- api_error: none
- parse_error: none
- validation_errors: none
- exact_match: False
- api_response_excerpt: {"model_instance_id": "qwen3-30b-a3b-instruct-2507", "output": [{"content": "{\"task\":\"extract_cf_request\",\"status\":\"partial\",\"cf_request\":{\"Family\":3,\"Online\":1,\"CDAccount\":1},\"missing_fields\":[\"Income
- final_message_excerpt: {"task":"extract_cf_request","status":"partial","cf_request":{"Family":3,"Online":1,"CDAccount":1},"missing_fields":["Income","CCAvg","Education","Mortgage","SecuritiesAccount","CreditCard"],"conflicts":[],"notes":[]}
