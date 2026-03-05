#!/usr/bin/env python3
"""
Phase 4 scaffold: PPO training entrypoint for gated terminal agents.

This script is intentionally conservative:
- loads a base causal LM
- optionally patches Qwen2 attention with gated_attention.py
- freezes non-gating parameters by default
- leaves the live Harbor rollout loop as an explicit next integration step
"""

from __future__ import annotations

import argparse
from pathlib import Path

from seed_utils import set_global_seed

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Missing training dependency: {exc}") from exc

from gated_attention import patch_qwen2_with_gated_attention
from rl_environment import EpisodeConfig, HarborTerminalEnv, RewardConfig


def freeze_non_gating_params(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        is_trainable = ("gate_proj" in name) or name.endswith(".alpha")
        param.requires_grad = is_trainable
        if is_trainable:
            trainable += param.numel()
    return trainable, total


def describe_trainable_params(model: torch.nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if param.requires_grad]


def build_env(args: argparse.Namespace) -> HarborTerminalEnv:
    reward_config = RewardConfig(
        success_reward=args.success_reward,
        invalid_command_penalty=args.invalid_command_penalty,
        crash_penalty=args.crash_penalty,
        step_penalty=args.step_penalty,
        repeated_command_penalty=args.repeated_command_penalty,
    )
    episode_config = EpisodeConfig(
        max_steps=args.max_steps,
        max_command_chars=args.max_command_chars,
        max_output_chars=args.max_output_chars,
        wall_clock_timeout_sec=args.wall_clock_timeout_sec,
    )
    return HarborTerminalEnv(reward_config=reward_config, episode_config=episode_config)


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO scaffold for gated terminal agent")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-gated-attention", action="store_true")
    parser.add_argument("--alpha-init", type=float, default=0.0)
    parser.add_argument("--learnable-alpha", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="artifacts/ppo_scaffold")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-command-chars", type=int, default=512)
    parser.add_argument("--max-output-chars", type=int, default=12000)
    parser.add_argument("--wall-clock-timeout-sec", type=int, default=180)
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--invalid-command-penalty", type=float, default=-0.1)
    parser.add_argument("--crash-penalty", type=float, default=-0.1)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--repeated-command-penalty", type=float, default=-0.02)
    args = parser.parse_args()

    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if args.patch_gated_attention:
        model = patch_qwen2_with_gated_attention(
            model,
            alpha_init=args.alpha_init,
            learnable_alpha=args.learnable_alpha,
        )

    trainable, total = freeze_non_gating_params(model)
    env = build_env(args)

    summary_path = output_dir / "setup_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"model_name={args.model_name}",
                f"tokenizer_class={tokenizer.__class__.__name__}",
                f"patch_gated_attention={args.patch_gated_attention}",
                f"alpha_init={args.alpha_init}",
                f"learnable_alpha={args.learnable_alpha}",
                f"trainable_params={trainable}",
                f"total_params={total}",
                f"trainable_param_names={describe_trainable_params(model)}",
                f"episode_max_steps={env.episode_config.max_steps}",
            ]
        )
    )

    print(f"setup_summary={summary_path}")
    print(
        "next_step=wire live Harbor rollouts into HarborTerminalEnv and connect collected "
        "trajectories/rewards to TRL PPOTrainer"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
