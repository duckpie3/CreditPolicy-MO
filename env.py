import gymnasium as gym
from gymnasium import spaces
import numpy as np

segments = 4
default_config = {
    "segments": segments,
    "horizon": 120,
    "period_lag_for_defaults": 6,
    "poisson_lambda": [2000, 1500, 800, 1200],
    "score_mu_sigma": [(0.0, 1.0)] * segments,
    "initial_thresholds": [0.0] * segments,
    "ead_mean": [50000] * segments,
    "lgd_mean": [0.45] * segments,
    "apr": [0.28] * segments,
    "funding_cost": 0.08,
    "inclusion_weights": [1.0, 1.5, 2.0, 1.2],
    "fairness_group_index": 1,
    "risk_cap_default_rate": 0.05,
    "capital_floor": 1.0,
    "cooldown_len": 2,
    "tv_limit": 0.06,
    "regime_markov": [
        [0.85, 0.12, 0.03],
        [0.10, 0.80, 0.10],
        [0.05, 0.20, 0.75],
    ],
    "seed": 123,
}

class CreditPolicyEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, config_params=None):
        super(CreditPolicyEnv, self).__init__()
        self.config = default_config.copy()
        self.config.update(config_params or {})
        self.action_dim = self.config["segments"] + 1
        self.action_space = spaces.Box(
            low=np.array([-0.02] * self.action_dim), high=np.array([0.02] * self.action_dim), shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_dim = 6 + 8
        self.observation_space = spaces.Box(low=np.array([0] * self.observation_dim), high=np.array([1] * self.observation_dim), shape=(self.observation_dim,), dtype=np.float32)
        self.reward_dim = 6
        self.reward_space = spaces.Box(low=-1, high=1, shape=(self.reward_dim,), dtype=np.float32)
        
    def reset(self, *, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        observation = np.array([self.rng.random() for _ in range(self.observation_dim)], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        # TODO: Implement step logic
        pass
