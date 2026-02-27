# 🗺️ Implementation Phases: Gated-Terminal-Agent

This document outlines the step-by-step roadmap to build, train, and evaluate the Gated-Terminal-Agent using local hardware. 

---

## Phase 1: Environment & Baseline Setup
*Goal: Get the testing framework running and establish a "Standard Attention" baseline metric to beat.*

1. **Install Prerequisites:**
   * Ensure Docker is installed and running on your local machine.
   * Create a fresh Python virtual environment.
   * Install the Harbor execution harness and standard ML libraries:
     ```bash
     pip install harbor transformers torch accelerate trl wandb
     ```
   * Pin package versions in a `requirements.txt` file and record CUDA/PyTorch versions for reproducibility.
   * Set experiment seeds (Python, NumPy, PyTorch, environment) before every run.
2. **Download the Benchmark:**
   * Pull the TerminalBench 2.0 dataset via Harbor. 
3. **Build the Baseline Agent:**
   * Write a simple Python script (`baseline_agent.py`) that loads a 1B parameter model (e.g., `Llama-3.2-1B-Instruct`) and uses a standard ReAct prompting loop to output bash commands.
4. **Run the Baseline A/B Test:**
   * Execute the agent against the benchmark to record the initial Task Resolution Rate.
     ```bash
     harbor run --dataset terminal-bench@2.0 --agent baseline_agent.py --n-concurrent 2
     ```
   * *Outcome:* You now have the exact metric your RL Gated Agent needs to beat.

---

## Phase 2: Architectural Patching (The "Gated" Mechanism)
*Goal: Inject the NeurIPS 2025 Gated Attention mathematics into a Hugging Face model without pre-training from scratch.*

1. **Create the Patch Script:**
   * Create a file named `gated_attention.py`.
2. **Monkey-Patching Hugging Face:**
   * Extract the source code for the attention class of your chosen model (e.g., `LlamaAttention` from `transformers.models.llama.modeling_llama`).
   * Add a linear layer for the gating weights: `self.gate_proj = nn.Linear(hidden_size, num_heads, bias=False)`.
   * Modify the forward pass. After the standard Scaled Dot-Product Attention (SDPA) calculates the context layer, apply a head-wise gate with shape alignment:
     ```python
     # Concept code (shape-safe) for the forward pass modification
     # hidden_states: [B, T, C]
     # context_layer: [B, H, T, D]
     gate_logits = self.gate_proj(hidden_states)              # [B, T, H]
     gate = torch.sigmoid(gate_logits).permute(0, 2, 1)       # [B, H, T]
     gated_context = context_layer * gate.unsqueeze(-1)       # [B, H, T, D]
     ```
3. **Initialize Weights to Zero:**
   * Initialize `gate_proj` weights to `0.0`, but preserve baseline behavior using a residual-style gate such as:
     `gated_context = context_layer * (1.0 + alpha * (2 * sigmoid(gate_logits) - 1))`
   * Start with `alpha = 0.0` so the initial behavior is exactly baseline, then learn/increase `alpha` during training.

---

## Phase 3: Supervised Fine-Tuning (SFT) Warm-up
*Goal: Teach the newly patched model how to format its thoughts and actions before introducing the chaos of RL.*

1. **Generate Trajectories:**
   * Use a stronger model (like Claude 3.5 Sonnet or GPT-4o) to successfully solve 10-20 TerminalBench tasks, saving the logs (State, Action, Next State).
2. **Format the Dataset:**
   * Convert these successful logs into a standard Hugging Face instruction dataset.
3. **Run SFT:**
   * Train your Gated Model for 1-2 epochs on this data using LoRA. 
   * *Why?* If you start RL immediately, the model will output gibberish, fail every task, and get negative rewards, preventing it from learning the gating mechanism. SFT gives it a solid starting policy.

---

## Phase 4: Reinforcement Learning (PPO) Setup
*Goal: Connect the Harbor environment to the PPO algorithm so the agent learns to filter terminal noise based on success/failure.*

1. **Define the Custom Environment:**
   * Wrap the Harbor execution loop in a standard OpenAI `gym` or Hugging Face `trl` compatible environment.
   * Define episode boundaries and guards: max steps, wall-clock timeout, max output tokens, and crash termination.
   * Add strict action parsing so model output maps to a single executable bash command each step.
2. **Define the Reward Function:**
   * `+1.0`: Harbor returns a "Success" flag for the container state.
   * `-0.1`: The agent outputs an invalid bash command or the container crashes.
   * `-0.01`: Step penalty (per command executed) to encourage the agent to find the root cause quickly instead of randomly exploring logs.
3. **Configure the PPO Trainer:**
   * Use `trl.PPOTrainer`.
   * Load your SFT-warmed-up Gated Model. 
   * Keep base model weights frozen initially, and train only gate parameters (and optionally a small policy head).
   * Use quantization for the frozen backbone if needed; QLoRA on `gate_proj` alone is usually unnecessary because it is already small.

---

## Phase 5: Training & Evaluation Loop
*Goal: Train the model and verify the hypotheses.*

1. **Execute the RL Run:**
   * Start the PPO training script. Monitor the loss and reward curves using Weights & Biases (`wandb`).
   * Expect the training to take 24-48 hours locally depending on your GPU.
2. **Visualize Attention Sparsity:**
   * Write an evaluation script that runs the trained model on a task and logs the output of `torch.sigmoid(self.gate_proj(hidden_states))`. 
   * Create a heatmap to prove the gates "close" (values approach 0) when the agent reads irrelevant `[INFO]` logs, and "open" (values approach 1) when it reads `[ERROR]` stack traces.
3. **Final Benchmark:**
   * Run the fully trained Gated Agent through Harbor one last time. Compare the Task Resolution Rate against the Phase 1 baseline.
4. **Ablation & Significance Checks:**
   * Run at least 3 random seeds for baseline and gated variants.
   * Report mean +/- standard deviation for each metric.
   * Include an ablation where gates are present but frozen to isolate the effect of learning.

---

## Phase 6 (Optional): Local Kubernetes Learning Track
*Goal: Learn Kubernetes locally without disrupting the core Docker-first research pipeline.*

1. **When to Start This Phase:**
   * Begin only after Phase 1 baseline execution is stable in Docker.
   * Treat Kubernetes as an infrastructure-learning track, not a blocker for model research.
2. **Pick a Local Kubernetes Runtime:**
   * Recommended: `kind` (Kubernetes in Docker) for lightweight local clusters.
   * Alternatives: Docker Desktop Kubernetes or Minikube.
3. **Create Minimal K8s Manifests:**
   * Add a namespace manifest (`k8s/namespace.yaml`).
   * Add a one-shot baseline job manifest (`k8s/agent-job.yaml`) that runs the baseline agent container.
4. **Scope and Expectations:**
   * Keep TerminalBench execution logic unchanged initially; only move job orchestration to Kubernetes.
   * Validate one-job execution, logs, and teardown before attempting any distributed or multi-job setup.
5. **Do Not Over-Optimize Early:**
   * Avoid cluster autoscaling, service meshes, and advanced operators in the first pass.
   * The priority is understanding Kubernetes primitives (`Namespace`, `Job`, `Pod`, logs, lifecycle) and preserving reproducibility.
