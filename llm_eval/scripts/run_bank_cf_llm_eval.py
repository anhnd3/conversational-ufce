from __future__ import annotations

from llm_eval.config import build_arg_parser, config_from_args
from llm_eval.runner import run_evaluation


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    result = run_evaluation(config)
    print(f"Run complete: {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
