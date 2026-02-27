#!/usr/bin/env python3
"""
Minimal baseline CLI agent scaffold for TerminalBench/Harbor experiments.

This baseline intentionally uses simple heuristics so you can validate
end-to-end harness wiring before integrating a learned policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from seed_utils import set_global_seed

@dataclass
class AgentState:
    cwd: str = ""
    stdout: str = ""
    stderr: str = ""
    last_command: str = ""


def decide_command(state: AgentState) -> str:
    """
    Cheap heuristic baseline:
    - If stderr is non-empty, inspect files and retry.
    - Otherwise, list files first.
    """
    if state.stderr.strip():
        return "ls -la && pwd"
    if "No such file or directory" in state.stdout:
        return "pwd && ls -la"
    return "ls -la"


def parse_state(raw: str) -> AgentState:
    """
    Accept either JSON lines or raw text state.
    JSON keys expected: cwd, stdout, stderr, last_command.
    """
    try:
        payload = json.loads(raw)
        return AgentState(
            cwd=payload.get("cwd", ""),
            stdout=payload.get("stdout", ""),
            stderr=payload.get("stderr", ""),
            last_command=payload.get("last_command", ""),
        )
    except json.JSONDecodeError:
        return AgentState(stdout=raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline terminal agent scaffold")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "--state",
        default="",
        help="State payload as JSON or text. If omitted, read from stdin.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    raw_state = args.state or sys.stdin.read()
    state = parse_state(raw_state)
    command = decide_command(state)
    print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
