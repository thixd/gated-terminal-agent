# Gated-Terminal-Agent: Enhancing CLI Agents with Gated Attention & RL

## The Goal
The objective of this project is to build and benchmark an autonomous, LLM-based systems engineering agent capable of solving complex, long-horizon command-line tasks via the **TerminalBench 2.0** framework. By replacing the standard Softmax attention mechanism of a base LLM with **Gated Attention** (NeurIPS 2025 Best Paper), and fine-tuning its behavior using **Reinforcement Learning (RL)**, this project creates a highly focused agent that can filter out massive amounts of terminal log noise to efficiently execute system operations and debug errors.

## The Innovation: Beyond the Paper
While the architectural mechanism (the head-specific sigmoid gate) originates from the Alibaba Qwen team's research for static text generation, **this project pioneers the application of Gated Attention to the domain of AI DevOps and Reinforcement Learning.** When an agent runs commands like `make install` or `apt-get`, the terminal outputs thousands of tokens of compilation logs and system messages. Standard LLMs suffer severe "context drowning," often losing track of the original task by the time an error trace appears. This project tests a novel hypothesis: *Can the noise-filtering properties of Gated Attention force an agent to learn "Attention Sparsity" via RL, physically shutting off neural pathways that process successful terminal logs to maintain absolute focus on actionable error traces?*

### Project vs. Original Research Comparison

| Vector of Innovation | Original NeurIPS 2025 Paper | This Project (Gated-Terminal-Agent) |
| :--- | :--- | :--- |
| **Domain Application** | Static NLP (Next-token prediction on text corpuses) | Dynamic Systems Engineering (Sequential CLI execution in TerminalBench 2.0) |
| **Training Paradigm** | Self-Supervised Learning (SSL) and Supervised Fine-Tuning (SFT) | Reinforcement Learning with Proximal Policy Optimization (PPO) |
| **Target "Noise"** | Semantic/Conversational (Irrelevant words in chat) | Structural/System (Compilation logs, success messages, raw bash output) |

## How It Works (The Architecture)
This project modifies standard attention by introducing a learned sigmoid gate after the Scaled Dot-Product Attention (SDPA) output. 

Output = (Softmax(Q K^T / sqrt(d)) V) * sigmoid(X W_g)

* **The RL Angle:** During PPO training, the reward signal directly updates the gating weights (W_g). When the agent successfully compiles a program or configures a server, the RL algorithm reinforces the gate's decision to "ignore" the scrolling `[INFO]` logs and "attend" only to the terminal prompt and `[ERROR]` outputs.

## Hypotheses & Benchmarks
Evaluate the agent using **TerminalBench 2.0**, a rigorous suite of 89 containerized tasks reflecting professional engineering workflows (e.g., reverse-engineering binaries, configuring networked services, and resolving dependency conflicts).

By running an A/B test between a Baseline Model (Standard Attention) and our Experimental Model (Gated Attention), we track:
1. **Task Resolution Rate:** The percentage of the 89 TerminalBench tasks successfully completed (verified via automated post-execution container tests).
2. **RL Training Stability:** Variance and drop-offs in the PPO reward curve.
3. **Command Efficiency:** The average number of bash commands executed to reach the correct system state.
4. **Attention Sparsity:** Visualizations of gating weights during inference to prove the model dynamically "shuts off" attention when scrolling past massive `stdout` log dumps.

## The Approach
* **Environment:** Isolated Docker containers managed by the **Harbor** framework.
* **State Representation:** The current working directory path, the previous bash command, and the standard output/error (`stdout`/`stderr`) of the terminal.
* **Action Space:** Raw bash commands (e.g., `grep`, `curl`, `python`, `vim`).
* **Reward Function:** * `+1.0` if the final container state passes the TerminalBench verification tests.
  * `-0.1` for command syntax errors or fatal crashes.
  * `-0.01` step penalty to encourage rapid resolution.
* **Algorithms:** Proximal Policy Optimization (PPO) via the Hugging Face `trl` library, utilizing QLoRA to train only the gating weights (W_g) for local VRAM efficiency.

## Current Status
- [x] Project scope and hypotheses defined
- [x] Phase plan drafted
- [ ] Baseline agent implemented
- [ ] Gated attention patch integrated
- [ ] SFT warm-up completed
- [ ] PPO training completed
- [ ] Final A/B evaluation and analysis completed

## Implementation Guide (Local Execution)
1. **Harbor Setup:** Install the execution harness via pip (`pip install harbor`) and ensure Docker is running locally.
2. **Establish Baseline:** Run your base model (e.g., Llama-3.2-1B-Instruct) against TerminalBench using the simple command: `harbor run --dataset terminal-bench@2.0 --agent <your_baseline_agent> --n-concurrent 2`.
3. **Architectural Patching:** Monkey-patch the Hugging Face `Attention` class in PyTorch to inject the Gated Attention equation.
4. **Supervised Warm-up:** Perform a brief SFT pass so the model learns the syntax of interacting with a headless terminal.
5. **RL Optimization:** Execute the PPO training loop, relying on Harbor's automated test-driven feedback to shape the attention gates.

