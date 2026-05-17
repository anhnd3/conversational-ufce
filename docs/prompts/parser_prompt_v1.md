# Parser Prompt v1

## Purpose

Convert a natural-language bank counterfactual request into exactly one canonical JSON object.

## System Prompt Artifact

- File: `llm/prompts/parser_system_prompt_v1.txt`
- Contract:
  - return JSON only
  - no markdown fences
  - no explanation text
  - use canonical field names and allowed top-level keys only
  - use status values `complete`, `partial`, `needs_clarification`, `conflict`

## User Prompt Template

- File: `llm/prompts/parser_user_template_v1.txt`
- Inputs:
  - schema and field contract JSON
  - case payload JSON

## Validation Backstop

- Contract YAML: `llm/config/parser_contract_v1.yaml`
- JSON Schema: `llm/config/parser_schema_v1.json`
- Runtime validator: `llm/src/validation/schema_validator.py`
