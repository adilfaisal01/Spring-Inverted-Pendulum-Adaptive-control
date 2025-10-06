import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from pendulum_code import SpringInvertedPendulum
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# create the environment
def make_env():
    env=SpringInvertedPendulum(M=0,k_spring=4.75)
    return env


from stable_baselines3.common.callbacks import BaseCallback

class LogStateCallback(BaseCallback):
    def _on_step(self) -> bool:
        # vectorized env obs
        state = self.locals['new_obs'][0]  
        reward = self.locals['rewards'][0]   # SB3 returns rewards in locals
        self.logger.record('env/reward', reward)
        self.logger.record('env/position', state[0])
        self.logger.record('env/velocity', state[1])
        return True


vec_env=DummyVecEnv([make_env])
model=PPO(policy='MlpPolicy',env=vec_env,verbose=1,tensorboard_log="./tensorboard_logs/",learning_rate=5e-5,normalize_advantage=True)
model.learn(total_timesteps=100000,callback=LogStateCallback()) #100k steps
model.save("ppo_spring_pendulum_kspring=4.75Nm")


