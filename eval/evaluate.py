import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3 import PPO
from envs.receiver_env import ReceiverEnv

def evaluate():
    env = ReceiverEnv()
    model = PPO.load("models/ppo_receiver_model")

    obs, info = env.reset()
    total_reward = 0.0
    correct_actions = 0
    total_steps = 0

    print("--- Starting Agent Evaluation ---")
    
    # Run evaluation across the dataset
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        if reward > 0:
            correct_actions += 1
        total_steps += 1
        
        done = terminated or truncated

    accuracy = (correct_actions / total_steps) * 100
    print(f"Total Steps Evaluated: {total_steps}")
    print(f"Total Reward         : {total_reward:.2f}")
    print(f"Accuracy Rate        : {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate()