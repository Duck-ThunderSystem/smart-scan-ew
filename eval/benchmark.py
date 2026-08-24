import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.receiver_env import ReceiverEnv

def evaluate_rl_agent(env, model):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    correct_hits = 0
    total_steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if reward > 0:
            correct_hits += 1
        total_steps += 1
        done = terminated or truncated

    accuracy = (correct_hits / total_steps) * 100
    return total_steps, total_reward, accuracy, correct_hits

def evaluate_random_agent(env):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    correct_hits = 0
    total_steps = 0

    while not done:
        # Open-Loop / Random channel selection strategy
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if reward > 0:
            correct_hits += 1
        total_steps += 1
        done = terminated or truncated

    accuracy = (correct_hits / total_steps) * 100
    return total_steps, total_reward, accuracy, correct_hits

def main():
    print("--- Running Benchmarking Evaluation ---")
    env = ReceiverEnv()
    model = PPO.load("models/ppo_receiver_model")

    # Evaluate RL Agent
    rl_steps, rl_reward, rl_acc, rl_hits = evaluate_rl_agent(env, model)

    # Evaluate Random / Open-Loop Baseline
    rand_steps, rand_reward, rand_acc, rand_hits = evaluate_random_agent(env)

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Random/Open-Loop Strategy -> Accuracy: {rand_acc:.2f}% | Total Reward: {rand_reward:.2f}")
    print(f"PPO Smart Scan Agent        -> Accuracy: {rl_acc:.2f}% | Total Reward: {rl_reward:.2f}")
    print(f"Performance Gain            -> +{rl_acc - rand_acc:.2f}% Improvement")
    print("====================================================")

    # Plot visual performance comparison
    strategies = ['Random / Open-Loop', 'PPO Smart Scan']
    accuracies = [rand_acc, rl_acc]
    rewards = [rand_reward, rl_reward]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Scanning Strategy')
    ax1.set_ylabel('Accuracy (%)', color=color)
    bars = ax1.bar(strategies, accuracies, color=['#ff9999', '#66b3ff'], width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')

    plt.title('EW Receiver Strategy: Open-Loop vs PPO Smart Scan')
    plt.tight_layout()
    
    output_path = "eval/benchmark_results.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    main()