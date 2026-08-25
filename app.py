import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from stable_baselines3 import PPO
from envs.receiver_env import ReceiverEnv

st.set_page_config(
    page_title="Smart Scan EW Strategy Dashboard",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Smart Scan Strategy for Electronic Warfare (EW)")
st.caption("Interactive Reinforcement Learning Receiver Scheduling & Real-time Spectrum Optimization")

# Load model and env
@st.cache_resource
def load_agent():
    env = ReceiverEnv()
    model = PPO.load("models/ppo_receiver_model")
    return env, model

env, model = load_agent()

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Settings")
sim_steps = st.sidebar.slider("Number of Steps / Pulses", min_value=20, max_value=500, value=100, step=10)
run_sim = st.sidebar.button("🚀 Run Live Simulation", use_container_width=True)

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

# Calculated Key Metrics
rl_acc = (np.array(rl_rewards) > 0).mean() * 100
rand_acc = (np.array(rand_rewards) > 0).mean() * 100
gain = rl_acc - rand_acc

# Top Metrics Bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("PPO Hit Accuracy", f"{rl_acc:.1f}%", f"{gain:+.1f}% vs Random")
col2.metric("PPO Cumulative Reward", f"{sum(rl_rewards):.1f}")
col3.metric("Random Baseline Accuracy", f"{rand_acc:.1f}%")
col4.metric("Total Steps Processed", len(rl_actions))

st.divider()

# Interactive Tabs
tab1, tab2, tab3 = st.tabs(["📊 Interactive Channel Hopping", "📈 Comparative Analytics", "📋 Step Inspection Data"])

with tab1:
    st.subheader("Live Channel Selection Trajectory")
    
    df_plot = pd.DataFrame({
        "Time Step": list(range(len(rl_actions))),
        "PPO Selected Channel": [f"Channel {a}" for a in rl_actions],
        "Hit Status": ["Hit (+1)" if r > 0 else "Miss (-1)" for r in rl_rewards],
        "Numeric Channel": rl_actions
    })

    fig_hop = px.scatter(
        df_plot,
        x="Time Step",
        y="PPO Selected Channel",
        color="Hit Status",
        symbol="Hit Status",
        color_discrete_map={"Hit (+1)": "#00CC96", "Miss (-1)": "#EF553B"},
        title="Dynamic Pulse-by-Pulse Channel Selection (Hover to inspect)",
        labels={"PPO Selected Channel": "Receiver Channel"}
    )
    fig_hop.update_traces(marker=dict(size=9))
    fig_hop.update_layout(height=400, hovermode="closest")
    st.plotly_chart(fig_hop, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Cumulative Reward Convergence")
        cum_rl = np.cumsum(rl_rewards)
        cum_rand = np.cumsum(rand_rewards)
        
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(y=cum_rl, mode='lines', name='PPO Smart Scan Agent', line=dict(color='#00CC96', width=3)))
        fig_cum.add_trace(go.Scatter(y=cum_rand, mode='lines', name='Random / Open-Loop Baseline', line=dict(color='#EF553B', width=2, dash='dash')))
        fig_cum.update_layout(xaxis_title="Step", yaxis_title="Cumulative Reward", height=380)
        st.plotly_chart(fig_cum, use_container_width=True)
        
    with col_b:
        st.subheader("Channel Allocation Distribution")
        channel_counts = pd.Series(rl_actions).value_counts().reset_index()
        channel_counts.columns = ["Channel", "Dwell Count"]
        channel_counts["Channel"] = channel_counts["Channel"].apply(lambda x: f"Channel {x}")
        
        fig_pie = px.pie(channel_counts, values="Dwell Count", names="Channel", hole=0.4, title="Agent Time Spent Per Receiver Channel")
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.subheader("Pulse Inspection Table")
    df_table = pd.DataFrame({
        "Step": range(len(rl_actions)),
        "Agent Channel": rl_actions,
        "Agent Reward": rl_rewards,
        "Baseline Channel": rand_actions,
        "Baseline Reward": rand_rewards
    })
    st.dataframe(df_table, use_container_width=True, height=350)