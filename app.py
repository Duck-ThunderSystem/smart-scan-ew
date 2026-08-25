import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from envs.receiver_env import ReceiverEnv

st.set_page_config(
    page_title="Smart Scan EW Strategy Dashboard",
    layout="wide"
)

st.title("Smart Scan Strategy for Electronic Warfare (EW)")
st.caption("Reinforcement Learning Receiver Scheduling & Real-time Spectrum Optimization")

# Load model and env
@st.cache_resource
def load_agent():
    env = ReceiverEnv()
    model = PPO.load("models/ppo_receiver_model")
    return env, model

env, model = load_agent()

# Sidebar Controls
st.sidebar.header("Simulation Settings")
sim_steps = st.sidebar.slider("Number of Steps / Pulses", min_value=20, max_value=500, value=100, step=10)
run_sim = st.sidebar.button("Run Live Simulation", use_container_width=True)

if run_sim or "sim_data" not in st.session_state:
    # 1. Run PPO Agent Simulation
    obs, info = env.reset()
    rl_actions, rl_rewards = [], []
    
    for _ in range(sim_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rl_actions.append(int(action))
        rl_rewards.append(float(reward))
        if terminated or truncated:
            break

    # 2. Run Random Baseline Simulation
    obs, info = env.reset()
    rand_actions, rand_rewards = [], []
    
    for _ in range(len(rl_actions)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rand_actions.append(int(action))
        rand_rewards.append(float(reward))
        if terminated or truncated:
            break

    # Save to session state
    st.session_state["sim_data"] = {
        "rl_actions": rl_actions,
        "rl_rewards": rl_rewards,
        "rand_actions": rand_actions,
        "rand_rewards": rand_rewards
    }

data = st.session_state["sim_data"]
rl_actions = data["rl_actions"]
rl_rewards = data["rl_rewards"]
rand_actions = data["rand_actions"]
rand_rewards = data["rand_rewards"]

# Key Metrics
rl_acc = (np.array(rl_rewards) > 0).mean() * 100
rand_acc = (np.array(rand_rewards) > 0).mean() * 100
gain = rl_acc - rand_acc

col1, col2, col3, col4 = st.columns(4)
col1.metric("PPO Hit Accuracy", f"{rl_acc:.1f}%", f"{gain:+.1f}% vs Random")
col2.metric("PPO Cumulative Reward", f"{sum(rl_rewards):.1f}")
col3.metric("Random Baseline Accuracy", f"{rand_acc:.1f}%")
col4.metric("Total Steps Processed", len(rl_actions))

st.divider()

# Streamlit Native Area Chart Display
st.subheader("Cumulative Reward Trajectory")

df_chart = pd.DataFrame({
    "PPO Agent": np.cumsum(rl_rewards),
    "Random Baseline": np.cumsum(rand_rewards)
})

st.area_chart(df_chart, color=["#8B5CF6", "#CBD5E1"])

# Tabbed Data Views
tab1, tab2 = st.tabs(["Channel Selection", "Data Inspection"])

with tab1:
    st.subheader("Receiver Channel Hopping History")
    df_hops = pd.DataFrame({
        "Selected Channel": rl_actions
    })
    st.line_chart(df_hops, color="#6366F1")

with tab2:
    st.subheader("Raw Simulation Data")
    df_table = pd.DataFrame({
        "Step": range(len(rl_actions)),
        "Agent Channel": rl_actions,
        "Agent Reward": rl_rewards,
        "Baseline Channel": rand_actions,
        "Baseline Reward": rand_rewards
    })
    st.dataframe(df_table, use_container_width=True, height=350)
