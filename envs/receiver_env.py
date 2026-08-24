import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.dataset import TSRDDataLoader

class ReceiverEnv(gym.Env):
    """Custom Gymnasium Environment for Electronic Warfare (EW) Receiver Scheduling."""
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(ReceiverEnv, self).__init__()
        
        # Load radar dataset
        loader = TSRDDataLoader()
        self.features, self.labels, self.feature_names = loader.load_data()
        self.num_samples = len(self.features)
        self.current_idx = 0

        # Action Space: Select which frequency band/emitter to tune to (e.g., 4 discrete channels)
        self.action_space = spaces.Discrete(4)

        # Observation Space: Radar pulse parameters [UTCTime, RF, PulseWidth, AOA, PA]
        # Using 5 continuous feature values
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        observation = self.features[self.current_idx].astype(np.float32)
        info = {}
        return observation, info

    def step(self, action):
        # Determine target label for current pulse sample
        target_label = int(self.labels[self.current_idx][0])
        
        # Reward mechanism: positive reward if action matches target emitter class
        reward = 1.0 if action == target_label else -0.1

        self.current_idx += 1
        terminated = self.current_idx >= self.num_samples - 1
        truncated = False

        observation = self.features[self.current_idx].astype(np.float32) if not terminated else np.zeros((5,), dtype=np.float32)
        info = {}

        return observation, reward, terminated, truncated, info

if __name__ == "__main__":
    env = ReceiverEnv()
    obs, info = env.reset()
    print("Environment initialized successfully!")
    print(f"Initial Observation: {obs}")
    
    # Test a dummy step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step Action: {action} | Reward: {reward} | Next Obs Shape: {obs.shape}")