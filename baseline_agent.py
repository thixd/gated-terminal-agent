#!/usr/bin/env python3
"""
Minimal baseline agent for TerminalBench/Harbor experiments.

This baseline intentionally uses simple heuristics so you can validate
end-to-end harness wiring before integrating a learned policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from seed_utils import set_global_seed

try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError:  # pragma: no cover - allows local CLI usage before install
    BaseAgent = object
    BaseEnvironment = object
    AgentContext = object

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover - allows local CLI usage before install
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

@dataclass
class AgentState:
    cwd: str = ""
    stdout: str = ""
    stderr: str = ""
    last_command: str = ""


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SAFE_FALLBACK_COMMAND = "pwd && ls -la"
LIKELY_SHELL_PREFIXES = {
    "ls",
    "pwd",
    "cat",
    "echo",
    "printf",
    "grep",
    "find",
    "head",
    "tail",
    "sed",
    "awk",
    "cut",
    "wc",
    "sort",
    "uniq",
    "cp",
    "mv",
    "rm",
    "mkdir",
    "touch",
    "chmod",
    "python",
    "python3",
    "pip",
    "gcc",
    "make",
    "git",
    "curl",
    "wget",
    "test",
    "bash",
    "sh",
    "true",
    "false",
    "cd",
}
PROSE_MARKERS = (
    "follow these steps",
    "here is",
    "to create",
    "you can",
    "this command",
    "the following",
)


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


def build_prompt(instruction: str, state: AgentState) -> str:
    return f"""You are a terminal agent operating in a Linux container.
Return exactly one executable bash command and nothing else.
Do not explain your reasoning.
Do not use markdown.
Do not use code fences.
Do not write prose.
Do not say "here is" or "follow these steps".
If the task requires creating or editing files, output a shell command that does that directly.
Valid examples:
ls -la
pwd && ls -la
cat > /app/hello.txt <<'EOF'
hello
EOF
Invalid examples:
To solve this task, first inspect the files:
Here is the command:
```bash
ls -la
```

Task:
{instruction.strip()}

Current working directory:
{state.cwd or "(unknown)"}

Last command:
{state.last_command or "(none)"}

stdout:
{state.stdout[-4000:] if state.stdout else "(empty)"}

stderr:
{state.stderr[-2000:] if state.stderr else "(empty)"}
"""


def looks_like_shell_command(candidate: str) -> bool:
    stripped = candidate.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(marker in lowered for marker in PROSE_MARKERS):
        return False
    if "\n" in stripped and "EOF" not in stripped:
        return False
    if stripped.endswith(":"):
        return False

    first_token = stripped.split()[0]
    if first_token in LIKELY_SHELL_PREFIXES:
        return True
    if first_token.startswith(("./", "/", "../")):
        return True
    if "=" in first_token and len(stripped.split()) > 1:
        return True
    return False


def clean_command(raw_text: str) -> str:
    text = raw_text.strip()
    if "```" in text:
        parts = [part.strip() for part in text.split("```") if part.strip()]
        if parts:
            text = parts[-1]
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return SAFE_FALLBACK_COMMAND

    if text.lower().startswith("bash\n"):
        text = text.split("\n", 1)[1].strip()
    elif text.lower().startswith("bash "):
        text = text[4:].strip()

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return SAFE_FALLBACK_COMMAND

    if "EOF" in text:
        candidate = text
    else:
        candidate = lines[0].strip()

    if not looks_like_shell_command(candidate):
        return SAFE_FALLBACK_COMMAND
    return candidate


class QwenCommandPolicy:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        max_new_tokens: int = 64,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._load_error: str | None = None

    def _resolve_model_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if torch is None:
            return kwargs

        if self.device == "auto":
            if torch.cuda.is_available():
                kwargs["torch_dtype"] = torch.float16
                kwargs["device_map"] = "auto"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float32
        return kwargs

    def load(self) -> None:
        if self._model is not None or self._load_error is not None:
            return
        if AutoTokenizer is None or AutoModelForCausalLM is None or torch is None:
            self._load_error = "transformers/torch are not installed"
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **self._resolve_model_kwargs(),
            )
            if (
                self.device == "auto"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                self._model = self._model.to("mps")
            elif self.device not in {"auto", "cpu"}:
                self._model = self._model.to(self.device)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - depends on local runtime
            self._load_error = str(exc)

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def generate_command(self, instruction: str, state: AgentState) -> str:
        self.load()
        if self._model is None or self._tokenizer is None:
            return decide_command(state)

        prompt = build_prompt(instruction, state)
        messages = [{"role": "user", "content": prompt}]
        rendered_prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer(rendered_prompt, return_tensors="pt")

        if hasattr(self._model, "device"):
            model_inputs = {
                key: value.to(self._model.device) for key, value in model_inputs.items()
            }

        with torch.no_grad():
            output = self._model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output[0][model_inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return clean_command(text)


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


class BaselineHarborAgent(BaseAgent):
    """
    Minimal Harbor-compatible baseline agent.

    The purpose is not benchmark performance; it is to validate the custom agent
    integration path and capture a reproducible baseline trial.
    """

    @staticmethod
    def name() -> str:
        return "baseline-harbor-agent"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        seed: int = 42,
        command_timeout_sec: int = 30,
        max_steps: int = 3,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._seed = seed
        self._command_timeout_sec = command_timeout_sec
        self._max_steps = max_steps
        self._policy = QwenCommandPolicy(model_name=model_name or DEFAULT_MODEL_NAME)

    def version(self) -> str:
        return "0.2.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        set_global_seed(self._seed)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        set_global_seed(self._seed)

        cwd_result = await environment.exec("pwd", timeout_sec=self._command_timeout_sec)
        cwd = (cwd_result.stdout or "").strip()

        commands: list[str] = []
        observations: list[dict[str, Any]] = []
        state = AgentState(cwd=cwd)

        for _ in range(self._max_steps):
            command = self._policy.generate_command(instruction, state)
            result = await environment.exec(command, timeout_sec=self._command_timeout_sec)
            commands.append(command)
            observations.append(
                {
                    "command": command,
                    "return_code": result.return_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            state = AgentState(
                cwd=cwd,
                stdout=(result.stdout or ""),
                stderr=(result.stderr or ""),
                last_command=command,
            )

        log_path = self.logs_dir / "baseline_agent.json"
        log_path.write_text(
            json.dumps(
                {
                    "instruction": instruction,
                    "seed": self._seed,
                    "model_name": self._policy.model_name,
                    "model_load_error": self._policy.load_error,
                    "cwd": cwd,
                    "commands": commands,
                    "observations": observations,
                },
                indent=2,
            )
        )

        context.metadata = {
            "seed": self._seed,
            "model_name": self._policy.model_name,
            "model_load_error": self._policy.load_error,
            "commands": commands,
            "baseline_log": str(log_path),
        }


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
    parser.add_argument(
        "--instruction",
        default="Inspect the environment and decide the next bash command.",
        help="Instruction to provide to the model-backed command policy.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Model name for Qwen command generation.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    raw_state = args.state or sys.stdin.read()
    state = parse_state(raw_state)
    policy = QwenCommandPolicy(model_name=args.model)
    command = policy.generate_command(args.instruction, state)
    print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
