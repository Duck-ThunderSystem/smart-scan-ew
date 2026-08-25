import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.receiver_env import ReceiverEnv

st.set_page_config(page_title="Smart Scan EW Dashboard", layout="wide")

st.title("📡 Smart Scan Strategy for Electronic Warfare (EW)")
st.markdown("Interactive Demonstration of Reinforcement Learning Receiver Scheduling")

# Load model and env
@st.cache_resource
def load_agent():
    env = ReceiverEnv()
    model = PPO.load("models/ppo_receiver_model")
    return env, model

env, model = load_agent()

# Sidebar controls
st.sidebar.header("Simulation Settings")
steps = st.sidebar.slider("Steps to Simulate", min_value=10, max_value=200, value=50)

if st.sidebar.button("Run Simulation"):
    obs, info = env.reset()
    rewards = []
    actions = []
    
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if terminated or truncated:
            break

    # Metrics display
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Steps", len(actions))
    col2.metric("Cumulative Reward", f"{sum(rewards):.2f}")
    col3.metric("Hit Rate / Accuracy", f"{(np.array(rewards) > 0).mean() * 100:.1f}%")

    # Channel Hop Plot
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.scatter(range(len(actions)), actions, c=actions, cmap='tab10', s=20)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Chan 0', 'Chan 1', 'Chan 2', 'Chan 3'])
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Receiver Channel Selection")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    st.pyplot(fig)
