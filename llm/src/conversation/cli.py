from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm.src.conversation.orchestrator import BankConversationOrchestrator
from llm.src.conversation.parser_adapter import (
    DEFAULT_API_BASE,
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_SYSTEM_PROMPT_PATH,
    DEFAULT_TIMEOUT_S,
    LiveLmStudioParserAdapter,
)
from llm.src.conversation.session import (
    create_interactive_session_state,
    handle_session_turn,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bank conversational UFCE MVP from the CLI.")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--text", default="", help="One-shot user input text.")
    mode.add_argument("--interactive", action="store_true", help="Run an interactive loop with one pending clarification seam.")
    parser.add_argument("--model-alias", default=DEFAULT_MODEL_ALIAS)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT_PATH))
    parser.add_argument("--out-dir", default="outputs/conversations")
    parser.add_argument("--scenario-slug", default="", help="Optional stable folder slug.")
    parser.add_argument("--debug-trace", action="store_true", help="Persist runtime debug trace artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    adapter = LiveLmStudioParserAdapter(
        model_alias=args.model_alias,
        api_base=args.api_base,
        timeout_s=args.timeout_s,
        benchmark_path=Path(args.benchmark),
        system_prompt_path=Path(args.system_prompt),
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=adapter.load_benchmark(),
        benchmark_path=Path(args.benchmark),
        system_prompt_path=Path(args.system_prompt),
        output_root=Path(args.out_dir),
        model_alias=args.model_alias,
    )

    if args.interactive:
        return run_interactive(orchestrator, args=args)
    if not args.text.strip():
        parser.error("Provide --text for one-shot mode or use --interactive.")
    result = orchestrator.run_turn(
        user_input=args.text,
        save_artifacts=True,
        scenario_slug=args.scenario_slug or None,
        debug_trace_enabled=args.debug_trace,
        command=shell_command(argv),
    )
    print(result.response_text)
    if result.artifact_record is not None:
        print(f"[OUT] {result.artifact_record.output_dir}")
    return 0


def run_interactive(orchestrator: BankConversationOrchestrator, *, args) -> int:
    print("Bank conversational UFCE MVP. Type 'quit' or 'exit' to stop.")
    session_state = create_interactive_session_state()
    while True:
        try:
            user_text = input("bank> ").strip()
        except EOFError:
            print()
            return 0
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            return 0
        result = handle_session_turn(
            orchestrator,
            session_state,
            user_input=user_text,
            save_artifacts=True,
            scenario_slug=None,
            debug_trace_enabled=args.debug_trace,
            command=shell_command(None),
        )
        print(result.response_text)
        if session_state.pending_clarification is not None:
            print("You may reply with only the missing fields for this clarification turn.")
        if result.artifact_record is not None:
            print(f"[OUT] {result.artifact_record.output_dir}")


def shell_command(argv: list[str] | None) -> str:
    active = argv if argv is not None else sys.argv[1:]
    if not active:
        return "python -m llm.src.conversation.cli"
    return "python -m llm.src.conversation.cli " + " ".join(active)
