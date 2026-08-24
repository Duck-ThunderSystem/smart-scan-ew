import sys
from pathlib import Path

# Add project root directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from envs.receiver_env import ReceiverEnv

def train():
    # Instantiate custom Gymnasium environment
    env = ReceiverEnv()

    # Initialize PPO model with MlpPolicy
    model = PPO("MlpPolicy", env, verbose=1)

    print("--- Starting Agent Training ---")
    model.learn(total_timesteps=10000)

    # Save trained model to disk
    model.save("models/ppo_receiver_model")
    print("Model saved to models/ppo_receiver_model.zip")

if __name__ == "__main__":
    train()