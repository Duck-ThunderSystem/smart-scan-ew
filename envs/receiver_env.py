import sys
from pathlib import Path

# Add project root directory to Python path BEFORE importing local modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.dataset import TSRDDataLoader


class ReceiverEnv(gym.Env):
    """Custom Gymnasium Environment for EW Receiver Scheduling with Feature Normalization."""
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(ReceiverEnv, self).__init__()
        
        loader = TSRDDataLoader()
        self.features, self.labels, self.feature_names = loader.load_data()
        
        # Continuous feature normalization (z-score scaling)
        self.features_mean = np.mean(self.features, axis=0)
        self.features_std = np.std(self.features, axis=0) + 1e-8
        self.normalized_features = (self.features - self.features_mean) / self.features_std
        
        self.num_samples = len(self.features)
        self.current_idx = 0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        observation = self.normalized_features[self.current_idx].astype(np.float32)
        return observation, {}

    def step(self, action):
        target_label = int(self.labels[self.current_idx][0])
        
        # Reward mechanism: positive reward on hit, penalty on miss
        reward = 1.0 if action == target_label else -0.5

        self.current_idx += 1
        terminated = self.current_idx >= self.num_samples - 1
        truncated = False

        observation = (
            self.normalized_features[self.current_idx].astype(np.float32)
            if not terminated
            else np.zeros((5,), dtype=np.float32)
        )
        return observation, reward, terminated, truncated, {}


if __name__ == "__main__":
    env = ReceiverEnv()
    obs, info = env.reset()
    print("Normalized Environment Initialized Successfully!")
    print(f"Normalized Initial Obs: {obs}")