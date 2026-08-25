import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.receiver_env import ReceiverEnv

def main():
    env = ReceiverEnv()
    model = PPO.load("models/ppo_receiver_model")
    
    obs, info = env.reset()
    rewards = []
    actions = []
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        done = terminated or truncated

    cum_rewards = np.cumsum(rewards)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(cum_rewards, color='tab:green', linewidth=1.5)
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('PPO Smart Scan Agent - Cumulative Reward Across Sequence')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.scatter(range(len(actions)), actions, c=actions, cmap='tab10', s=10, alpha=0.7)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['Chan 0', 'Chan 1', 'Chan 2', 'Chan 3'])
    ax2.set_xlabel('Pulse Index / Step')
    ax2.set_ylabel('Selected Channel')
    ax2.set_title('Agent Receiver Channel Selection Schedule')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = "eval/performance_schedule.png"
    plt.savefig(output_path, dpi=300)
    print(f"Performance schedule plot saved to {output_path}")

if __name__ == "__main__":
    main()