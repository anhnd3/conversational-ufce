"""Evaluation harness for bank CF parser benchmarking."""

"""
Copy/paste commands:

.venv/bin/python -m llm_eval.scripts.run_bank_cf_llm_eval \
  --benchmark llm_eval/benchmarks/ufce_bank_cf_parser_benchmark_v1.yaml \
  --model_alias qwen/qwen3.5-9b \
  --out_dir llm_eval/outputs

.venv/bin/python -m llm_eval.scripts.run_bank_cf_llm_eval \
  --benchmark llm_eval/benchmarks/ufce_bank_cf_parser_benchmark_v1.yaml \
  --model_alias google/gemma-3-12b \
  --out_dir llm_eval/outputs

.venv/bin/python -m llm_eval.scripts.run_bank_cf_llm_eval \
  --benchmark llm_eval/benchmarks/ufce_bank_cf_parser_benchmark_v1.yaml \
  --model_alias openai/gpt-oss-20b \
  --out_dir llm_eval/outputs

.venv/bin/python -m llm_eval.scripts.run_bank_cf_llm_eval \
  --benchmark llm_eval/benchmarks/ufce_bank_cf_parser_benchmark_v1.yaml \
  --model_alias mistralai/ministral-3-14b-reasoning \
  --out_dir llm_eval/outputs
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
