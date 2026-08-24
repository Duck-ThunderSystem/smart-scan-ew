# smart-scan-ew
# Smart Scan Strategy for Electronic Warfare (EW)

An intelligent, Reinforcement Learning (RL)-driven Receiver Scheduling framework designed to optimize Electronic Warfare (EW) signal scanning across multi-channel environments using Gymnasium and Stable-Baselines3.

## Key Features

- **Custom EW Environment**: Built on `gymnasium.Env` to simulate multi-channel EW receiver parameter tracking (UTCTime, RF, PulseWidth, AOA, PA).
- **Z-Score Feature Normalization**: Standardizes dynamic continuous pulse features to stabilize agent training convergence.
- **PPO Scheduling Agent**: Trained using Proximal Policy Optimization (PPO) with an `MlpPolicy`.
- **Benchmarking Suite**: Built-in evaluation comparing intelligent scanning against open-loop baseline strategies.

## Benchmark Results

| Scanning Strategy | Accuracy (%) | Total Reward | Performance Gain |
| :--- | :--- | :--- | :--- |
| **Random / Open-Loop Baseline** | 23.10% | -998.00 | Baseline |
| **PPO Smart Scan Agent** | **92.56%** | **+5776.00** | **+69.46%** |

## Quick Start

### 1. Installation & Environment Setup
```bash
# Set PYTHONPATH to project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Install dependencies
pip install -r requirements.txt