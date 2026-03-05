#!/usr/bin/env python3
"""
Phase 4 scaffold: reward-shaped terminal environment helpers.

This file intentionally separates three concerns:
- command validation and guardrails
- state/reward bookkeeping
- an environment interface that can later be backed by Harbor task execution
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TerminalState:
    instruction: str
    cwd: str = ""
    stdout: str = ""
    stderr: str = ""
    last_command: str = ""
    step_index: int = 0
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardConfig:
    success_reward: float = 1.0
    invalid_command_penalty: float = -0.1
    crash_penalty: float = -0.1
    step_penalty: float = -0.01
    repeated_command_penalty: float = -0.02


@dataclass
class EpisodeConfig:
    max_steps: int = 10
    max_command_chars: int = 512
    max_output_chars: int = 12000
    wall_clock_timeout_sec: int = 180


@dataclass
class StepResult:
    state: TerminalState
    reward: float
    done: bool
    info: dict[str, Any]


class CommandValidator:
    """
    Conservative shell command validator for early RL loops.
    """

    _DANGEROUS_PATTERNS = (
        r"\brm\s+-rf\s+/\b",
        r"\bshutdown\b",
        r"\breboot\b",
        r":\(\)\s*\{",
    )

    def __init__(self, max_command_chars: int = 512):
        self.max_command_chars = max_command_chars

    def validate(self, command: str) -> tuple[bool, str | None]:
        stripped = command.strip()
        if not stripped:
            return False, "empty_command"
        if len(stripped) > self.max_command_chars:
            return False, "command_too_long"
        for pattern in self._DANGEROUS_PATTERNS:
            if re.search(pattern, stripped):
                return False, "dangerous_command"
        try:
            shlex.split(stripped)
        except ValueError:
            return False, "shell_parse_error"
        return True, None


class RewardShaper:
    def __init__(self, config: RewardConfig):
        self.config = config

    def step_reward(
        self,
        *,
        is_valid: bool,
        crashed: bool,
        success: bool,
        repeated: bool,
    ) -> float:
        reward = self.config.step_penalty
        if not is_valid:
            reward += self.config.invalid_command_penalty
        if crashed:
            reward += self.config.crash_penalty
        if repeated:
            reward += self.config.repeated_command_penalty
        if success:
            reward += self.config.success_reward
        return reward


class HarborTerminalEnv:
    """
    Minimal RL-friendly environment interface.

    This scaffold does not yet launch Harbor tasks directly; instead, it defines
    the logic and hooks that a future Harbor-backed runner will implement.
    """

    def __init__(
        self,
        reward_config: RewardConfig | None = None,
        episode_config: EpisodeConfig | None = None,
    ):
        self.reward_config = reward_config or RewardConfig()
        self.episode_config = episode_config or EpisodeConfig()
        self.validator = CommandValidator(
            max_command_chars=self.episode_config.max_command_chars
        )
        self.reward_shaper = RewardShaper(self.reward_config)
        self.state: TerminalState | None = None
        self._command_history: list[str] = []

    def reset(self, instruction: str, cwd: str = "") -> TerminalState:
        self.state = TerminalState(instruction=instruction, cwd=cwd)
        self._command_history = []
        return self.state

    def _transition(
        self,
        command: str,
        stdout: str,
        stderr: str,
        *,
        success: bool,
        crashed: bool,
    ) -> StepResult:
        if self.state is None:
            raise RuntimeError("Environment must be reset before stepping.")

        repeated = command in self._command_history
        self._command_history.append(command)

        is_valid, invalid_reason = self.validator.validate(command)
        self.state.stdout = stdout[: self.episode_config.max_output_chars]
        self.state.stderr = stderr[: self.episode_config.max_output_chars]
        self.state.last_command = command
        self.state.step_index += 1
        self.state.done = (
            success or crashed or self.state.step_index >= self.episode_config.max_steps
        )

        reward = self.reward_shaper.step_reward(
            is_valid=is_valid,
            crashed=crashed,
            success=success,
            repeated=repeated,
        )
        info = {
            "invalid_reason": invalid_reason,
            "success": success,
            "crashed": crashed,
            "repeated_command": repeated,
            "step_index": self.state.step_index,
        }
        return StepResult(state=self.state, reward=reward, done=self.state.done, info=info)

    def step(
        self,
        command: str,
        stdout: str = "",
        stderr: str = "",
        *,
        success: bool = False,
        crashed: bool = False,
    ) -> StepResult:
        """
        Local/state-only step helper used for dry runs and unit tests.

        A Harbor-backed implementation should replace stdout/stderr/success/crashed
        with real environment execution results.
        """
        return self._transition(
            command=command,
            stdout=stdout,
            stderr=stderr,
            success=success,
            crashed=crashed,
        )
