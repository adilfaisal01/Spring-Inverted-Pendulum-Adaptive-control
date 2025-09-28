import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pendulum_code import SpringInvertedPendulum  # your env

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide all GPUs

# Define all training contexts
spring_values = np.array([60,100,50,40]) * 0.11298
mass_values = np.array([0, 0.5, 1, 1.5])

# Load your trained agent
model = PPO.load("ppo_spring_ip.zip")

# Store results
results = []

for k in spring_values:
    for ma in mass_values:
        # Create environment for this context
        env = SpringInvertedPendulum(M=ma, k_spring=k, max_steps=10000, render_mode=None)
        vec_env = DummyVecEnv([lambda: env])
        obs = vec_env.reset()

        total_reward = 0
        final_theta = 0
        final_theta_dot = 0

        # Run one episode
        done = False
        steps = 0
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=False)

            # Ensure action is always a scalar float for your env
            if hasattr(action, "__len__"):  # if array-like
                action_to_env = float(np.ravel(action)[0])
            else:  # scalar
                action_to_env = float(action)
            
            action_to_env *= 1 # test stronger torque


            obs, reward, terminated, truncated= vec_env.step([action_to_env])

            # Extract batched values safely
            reward_val = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            terminated_val = terminated[0] if isinstance(terminated, (list, np.ndarray)) else terminated
            truncated_val = truncated[0] if isinstance(truncated, (list, np.ndarray)) else truncated
            obs_val = obs[0] if isinstance(obs, (list, np.ndarray)) else obs

            total_reward += reward_val
            final_theta = obs_val[0]
            final_theta_dot = obs_val[1]
            done = terminated_val or truncated_val
            steps += 1

        results.append({
            "mass": ma,
            "spring": k,
            "total_reward": total_reward,
            "final_theta": final_theta,
            "final_theta_dot": final_theta_dot
        })

# Convert to DataFrame for easy analysis
df_results = pd.DataFrame(results)
print(df_results)

# Optional: save to CSV
df_results.to_csv("agent_performance_across_contexts.csv", index=False)
