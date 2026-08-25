import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from envs.receiver_env import ReceiverEnv

st.set_page_config(
    page_title="Smart Scan EW Receiver Operations Center",
    layout="wide"
)

st.title("Smart Scan Strategy for Electronic Warfare (EW)")
st.caption("AI-Driven Receiver Channel Scheduling vs. Legacy Open-Loop Sweeping")

# Load agent and env
@st.cache_resource
def load_agent():
    env = ReceiverEnv()
    model = PPO.load("models/ppo_receiver_model")
    return env, model

env, model = load_agent()

# Control Panel
st.sidebar.header("Control Panel")
sim_steps = st.sidebar.slider("Observation Window (Time Steps)", min_value=20, max_value=200, value=60, step=10)
run_sim = st.sidebar.button("Execute Threat Scenario Simulation", use_container_width=True)

if run_sim or "sim_data" not in st.session_state:
    # 1. PPO Run
    obs, info = env.reset()
    rl_actions, rl_rewards = [], []
    for _ in range(sim_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rl_actions.append(int(action))
        rl_rewards.append(float(reward))
        if terminated or truncated:
            break

    # 2. Random Baseline Run
    obs, info = env.reset()
    rand_actions, rand_rewards = [], []
    for _ in range(len(rl_actions)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rand_actions.append(int(action))
        rand_rewards.append(float(reward))
        if terminated or truncated:
            break

    st.session_state["sim_data"] = {
        "rl_actions": rl_actions, "rl_rewards": rl_rewards,
        "rand_actions": rand_actions, "rand_rewards": rand_rewards
    }

data = st.session_state["sim_data"]
rl_actions, rl_rewards = data["rl_actions"], data["rl_rewards"]
rand_actions, rand_rewards = data["rand_actions"], data["rand_rewards"]

rl_acc = (np.array(rl_rewards) > 0).mean() * 100
rand_acc = (np.array(rand_rewards) > 0).mean() * 100
gain = rl_acc - rand_acc

# Key Executive Summary for Judges
st.subheader("Executive System Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AI Agent Intercept Rate", f"{rl_acc:.1f}%", f"+{gain:.1f}% Improvement")
col2.metric("Legacy Sweep Intercept Rate", f"{rand_acc:.1f}%")
col3.metric("Total Radar Pulses Observed", len(rl_actions))
col4.metric("Threat Detection Status", "Optimal" if rl_acc > 70 else "Degraded")

st.divider()

# Interactive Cumulative Performance
st.subheader("Real-Time Cumulative Interception Comparison")
df_cum = pd.DataFrame({
    "AI Agent (PPO Smart Scan)": np.cumsum(rl_rewards),
    "Legacy System (Random Sweep)": np.cumsum(rand_rewards)
})
st.area_chart(df_cum, color=["#10B981", "#EF4444"])

st.divider()

# Tabbed Technical Views
tab1, tab2 = st.tabs(["Channel Selection Dynamics", "Step-by-Step Pulse Audit"])

with tab1:
    st.subheader("AI Receiver Dwell Pattern Across Spectrum Channels")
    df_hops = pd.DataFrame({
        "Receiver Channel (0 to 3)": rl_actions
    })
    st.line_chart(df_hops, color="#3B82F6")

with tab2:
    st.subheader("Pulse-Level Decision Audit Log")
    df_table = pd.DataFrame({
        "Step": range(len(rl_actions)),
        "AI Selected Channel": rl_actions,
        "AI Result": ["Hit (+1)" if r > 0 else "Miss (-1)" for r in rl_rewards],
        "Legacy Selected Channel": rand_actions,
        "Legacy Result": ["Hit (+1)" if r > 0 else "Miss (-1)" for r in rand_rewards]
    })
    st.dataframe(df_table, use_container_width=True, height=350)
