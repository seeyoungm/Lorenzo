from __future__ import annotations

import argparse

from lorenzo.config import load_config
from lorenzo.orchestrator import LorenzoOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lorenzo v1 CLI demo")
    parser.add_argument("--config", type=str, default=None, help="Path to TOML config")
    parser.add_argument(
        "--once",
        type=str,
        default=None,
        help="Run one turn and print response",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    orchestrator = LorenzoOrchestrator.from_config(config)

    if args.once:
        result = orchestrator.run_turn(args.once)
        print(result.response)
        return

    print("Lorenzo v1 CLI")
    print("종료: /exit, 저장된 메모리 수 확인: /count")

    while True:
        user_input = input("you> ").strip()
        if not user_input:
            continue

        if user_input in {"/exit", "exit", "quit"}:
            print("세션을 종료합니다.")
            break

        if user_input == "/count":
            count = orchestrator.modules.memory_store.count()
            print(f"assistant> memory items: {count}")
            continue

        result = orchestrator.run_turn(user_input)
        print(f"assistant> {result.response}")


if __name__ == "__main__":
    main()
