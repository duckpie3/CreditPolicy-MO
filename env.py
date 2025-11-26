import gymnasium as gym
from gymnasium import spaces
import numpy as np

segments = 1
default_config = {
    "segments": segments,
    "groups": 2,
    "horizon": 120,
    "period_lag_for_defaults": 6,
    "base_pd": 0.03,
    "poisson_lambda": 2000,
    "score_mu_sigma": (0.0, 1.0),
    "initial_threshold": 0.0,
    "ead_mean": 50000,
    "lgd_mean": 0.45,
    "apr": 0.28,
    "funding_cost": 0.08,
    "profit_target": 200000,
    "inclusion_weights": 1.0,
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
        self.action_dim = 1
        self.action_space = spaces.Box(
            low=np.array([-0.02] * self.action_dim),
            high=np.array([0.02] * self.action_dim),
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_dim = 6 + 8
        self.observation_space = spaces.Box(
            low=np.array([0] * self.observation_dim),
            high=np.array([1] * self.observation_dim),
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.reward_dim = 6
        self.reward_space = spaces.Box(
            low=-1, high=1, shape=(self.reward_dim,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        obs = np.array(
            [self.rng.random() for _ in range(self.observation_dim)], dtype=np.float32
        )
        self.approvals_history = [[0] * self.config["period_lag_for_defaults"] for _ in range(self.config["groups"])]
        self.threshold = self.config["initial_thresholds"]
        self.t = 0
        self.capital = self.config["capital_floor"]
        info = {}
        return obs, info

    def step(self, action):
        # Simulate applicant arrivals and scores
        applicant_arrivals = self.rng.poisson(self.config["poisson_lambda"])
        num_groups = self.config["groups"]
        group_ids = self.rng.integers(0, num_groups, size=applicant_arrivals)
        score_dist = self.rng.normal(
            self.config["score_mu_sigma"][0],
            self.config["score_mu_sigma"][1],
            size=applicant_arrivals,
        )
        # Apply adjustment from action
        self.threshold += action[0]
        # Calculate approvals per group
        approvals = [(score_dist[group_ids == i] >= self.threshold).sum() for i in range(num_groups)]
        total_approvals = sum(approvals)
        for i in range(num_groups):
            self.approvals_history[i].append(approvals[i])
        lagged_approvals = [self.approvals_history[i].pop(0) for i in range(num_groups)]
        # Calculate  defaults
        defaults = [self.rng.binomial(lagged_approvals[i], self.config["base_pd"]) for i in range(num_groups)]
        total_defaults = sum(defaults)

        loss = total_defaults * self.config["lgd_mean"] * self.config["ead_mean"]
        exposure = total_approvals * self.config["ead_mean"]
        income = exposure * self.config["apr"] / 12
        funding_cost = exposure * self.config["funding_cost"] / 12
        profit = income - loss - funding_cost
        # Update portfolio metrics
        approval_rate = total_approvals / applicant_arrivals if applicant_arrivals > 0 else 0
        default_rate = total_defaults / sum(lagged_approvals) if sum(lagged_approvals) > 0 else 0
        portfolio_metrics = [ approval_rate, default_rate, exposure, loss, profit, applicant_arrivals]
        # Update incluison and fairness metrics
        overall_inclusion = total_approvals / applicant_arrivals if applicant_arrivals > 0 else 0
        segment_inclusion = overall_inclusion  # Single-segment environment
        

        r_profit = np.clip(profit / self.config["profit_target"], -1, 1)
        tail_loss = loss / (self.config["ead_mean"] * self.config["poisson_lambda"])
        r_risk = np.clip(tail_loss / self.config["risk_cap_default_rate"], -1, 1)
        r_inclusion = np.clip(overall_inclusion / self.config["inclusion_weights"], -1, 1)
        fairness_metric = 0.5  # Placeholder
        r_fairness = np.clip(fairness_metric, -1, 1)
        tv = np.abs(action[0])
        r_instability = np.clip(tv / self.config["tv_limit"], -1, 1)
        r_complexity = 1.0  # Placeholder
        reward = np.array(
            [
                r_profit,
                -r_risk,
                r_inclusion,
                -r_fairness,
                -r_instability,
                -r_complexity,
            ],
            dtype=np.float32,
        )
        self.t += 1
        truncated = self.t >= self.config["horizon"]
        self.capital += profit
        terminated = False if self.capital >= self.config["capital_floor"] else True
        # Placeholder for next observation
        obs = np.array(
            [self.rng.random() for _ in range(self.observation_dim)], dtype=np.float32
        )
        info = {}
        return obs, reward, terminated, truncated, info
