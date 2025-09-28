import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from pendulum_code import SpringInvertedPendulum  # your env


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide all GPUs

# -----------------------------
# Custom callback for TensorBoard
# -----------------------------
from stable_baselines3.common.callbacks import BaseCallback

class ThetaLoggerCallback(BaseCallback):
    """
    Logs per-episode average of theta, theta_dot, and cumulative reward.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.theta_sum = 0.0
        self.theta_dot_sum = 0.0
        self.reward_sum = 0.0  # <-- accumulator for reward
        self.steps = 0

    def _on_step(self) -> bool:
        # Get first env state
        envs = self.training_env.envs
        state = envs[0].state
        theta, theta_dot = state

        # Accumulate step values
        self.theta_sum += theta
        self.theta_dot_sum += theta_dot

        # Get rewards for this step
        rewards = self.locals.get("rewards")
        if rewards is not None:
            # If vectorized, sum over all envs
            self.reward_sum += sum(rewards)

        self.steps += 1

        # Check for end of episode
        dones = self.locals.get("dones")
        if dones is not None and any(dones):
            # Compute per-episode averages
            avg_theta = self.theta_sum / self.steps
            avg_theta_dot = self.theta_dot_sum / self.steps
            cum_reward = self.reward_sum  # total reward for episode

            # Log to TensorBoard
            self.logger.record("custom/avg_theta", avg_theta)
            self.logger.record("custom/avg_theta_dot", avg_theta_dot)
            self.logger.record("custom/episode_reward", cum_reward)

            # Reset accumulators for next episode
            self.theta_sum = 0.0
            self.theta_dot_sum = 0.0
            self.reward_sum = 0.0
            self.steps = 0

        return True

# -----------------------------
# Create environment
# -----------------------------
env = SpringInvertedPendulum(M=0, k_spring=10.0, max_steps=500,render_mode='human')
vec_env = DummyVecEnv([lambda: env])

# TensorBoard log folder
log_dir = "./ppo_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

# -----------------------------
# Initialize PPO
# -----------------------------
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    clip_range=0.2,
    device="cpu",                 # CPU-only
    tensorboard_log=log_dir
)

# -----------------------------
# Train the agent
# -----------------------------
callback = ThetaLoggerCallback()
model.learn(total_timesteps=1000000, tb_log_name="PPO_SpringIP", callback=callback)

# -----------------------------
# Save the trained model
# -----------------------------
model.save("ppo_spring_ip")

# -----------------------------
# Load and test
# -----------------------------
model = PPO.load("ppo_spring_ip.zip", env=vec_env)

obs= vec_env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated = vec_env.step(action)

    # Optionally log to console for debugging
    # print(f"Theta: {obs[0,0]:.2f}, Theta_dot: {obs[0,1]:.2f}, Reward: {reward:.3f}")

    if terminated or truncated:
        obs = vec_env.reset()
