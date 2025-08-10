# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMM-R1 is a two-stage rule-based RL framework for enhancing reasoning capabilities in 3B Large Multimodal Models (LMMs). It's built on top of OpenRLHF infrastructure and specifically designed for multimodal reasoning tasks.

Key stages:
1. **Foundational Reasoning Enhancement (FRE)**: Text-only training to build reasoning foundations
2. **Multimodal Generalization Training (MGT)**: Extension to multimodal domains

## Common Commands

### Installation and Setup
```bash
# Clone and install
git clone https://github.com/TideDra/lmm-r1.git
cd lmm-r1
pip install -e .[vllm]
pip install flash_attn --no-build-isolation

# Docker setup (alternative)
bash examples/scripts/nvidia_docker_install.sh
```

### Training Commands
```bash
# Stage 1: Foundational Reasoning Enhancement
bash examples/scripts/lmm_r1/train_fre_text.sh      # Text-only FRE
bash examples/scripts/lmm_r1/train_fre_multi.sh     # Multimodal FRE

# Stage 2: Multimodal Generalization Training
bash examples/scripts/lmm_r1/train_mgt_geo.sh       # Geometry domain
bash examples/scripts/lmm_r1/train_mgt_percereas.sh # Perception-reasoning balanced

# Direct RL training (baseline comparison)
bash examples/scripts/lmm_r1/train_direct_rl_geo.sh

# Other training modes
bash examples/scripts/train_sft_llama.sh            # Supervised fine-tuning
bash examples/scripts/train_ppo_llama_ray.sh        # PPO training
bash examples/scripts/train_dpo_llama.sh            # DPO training
```

### Testing and Quality Assurance
```bash
# Code formatting and linting (based on pyproject.toml configuration)
black --line-length 119 .
isort --profile black --line-length 119 .
ruff check --line-length 119 .

# Testing (pytest configuration in pyproject.toml)
pytest --verbose --pyargs --durations=0 --strict-markers
pytest tests/                    # Run all tests
pytest -m unit                   # Run unit tests only
pytest -m integration            # Run integration tests only
```

### Model Serving and Inference
```bash
# Serve reward model
python -m openrlhf.cli.serve_rm --model_name_or_path <path>

# Interactive chat
python -m openrlhf.cli.interactive_chat --pretrain <model_path>

# Batch inference
python -m openrlhf.cli.batch_inference --pretrain <model_path>
```

## Architecture Overview

### Core Components

**openrlhf/cli/**: Main command-line interfaces for training and inference
- `train_ppo_ray.py`: Distributed PPO training with Ray
- `train_sft.py`: Supervised fine-tuning
- `train_dpo.py`: Direct Preference Optimization
- `train_rm.py`: Reward model training
- `serve_rm.py`: Reward model serving

**openrlhf/models/**: Model implementations and multimodal support
- `actor.py`: Actor model for RL training
- `model.py`: Base model implementations
- `lmm_kits/`: Multimodal model support (Qwen2.5-VL, Phi3.5-V, Phi4-MM, Gemma3)
- `remote_rm/`: Remote reward model implementations (math_verifier, sokoban_verifier)

**openrlhf/trainer/**: Training logic
- `ppo_trainer.py`: PPO training implementation
- `sft_trainer.py`: Supervised fine-tuning trainer
- `dpo_trainer.py`: DPO trainer
- `ray/`: Distributed training with Ray

**openrlhf/datasets/**: Data processing and loading
- `sft_dataset.py`: SFT dataset handling
- `reward_dataset.py`: Reward model datasets
- `prompts_dataset.py`: Prompt datasets for RL

### Data Format Requirements

Multimodal datasets must follow OpenAI-compatible message format:
```json
[
  {
    "message": "[{\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": \"file:///path/to/image.jpg\"}, {\"type\": \"text\", \"text\": \"Question text\"}]}]",
    "answer": "Expected answer"
  }
]
```

### Key Configuration Points

**Training Scripts Configuration**:
- Modify environment variables in training scripts (WORKSPACE_DIR, DATASET_PATH, PRETRAIN_MODEL_PATH)
- Adjust model paths and dataset paths before running
- Configure Ray cluster settings for distributed training

**Model Support**:
- Primary models: Qwen2.5-VL-3B-Instruct, Phi3.5-V, Phi4-Multimodal
- Automatic model kit selection based on model_type in config
- Falls back to LLM kit for unsupported multimodal models

**Reward Models**:
- Built-in verifiers for math reasoning and Sokoban tasks
- Support for remote reward models via `--remote_rm_url`
- Multiple reward model support with `--reward_pretrain model1,model2`

### Ray Integration

This codebase heavily uses Ray for distributed training:
- Ray-based PPO and REINFORCE++ implementations
- Hybrid engine support for memory efficiency
- vLLM integration for accelerated generation
- Multi-node training capabilities

### Important Files to Modify

When working with this codebase:
1. Training scripts in `examples/scripts/lmm_r1/` - update paths and configurations
2. Dataset paths and formats in data loading
3. Model configurations in training arguments
4. Ray cluster configurations for distributed training

## Development Notes

- Use Python 3.10+ (required in setup.py)
- vLLM 0.7.2+ recommended for optimal performance  
- Flash Attention 2 supported via `--flash_attn` flag
- QLoRA and LoRA support available
- Wandb and TensorBoard logging supported