import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from SpringInvertedPendulumEnvs.envs.pendulum_code import SpringInvertedPendulum
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# create the environment
def make_env():
    env=SpringInvertedPendulum(M=0,k_spring=5.67)
    return env

def custom_evaluate(model, env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)

from stable_baselines3.common.callbacks import BaseCallback
vec_env=DummyVecEnv([make_env])

## model testing
loaded_model = PPO.load("ppo_spring_pendulum_kspring=5.67Nm")
mean_reward, std_reward = custom_evaluate(loaded_model, vec_env, 50)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")