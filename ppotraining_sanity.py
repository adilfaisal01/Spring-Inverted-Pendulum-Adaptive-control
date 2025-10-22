import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from SpringInvertedPendulumEnvs.envs.pendulum_code import SpringInvertedPendulum
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
# from SpringInvertedPendulumCARL import CARLSpringInvertedPendulum

# create the environment
mass_on_top=0 #kg
spring_const=9.08 #Nm

def make_env():
    env=SpringInvertedPendulum(M=mass_on_top, k_spring=spring_const)
    return env


from stable_baselines3.common.callbacks import BaseCallback

class LogStateCallback(BaseCallback):
    def _on_step(self) -> bool:
        # vectorized env obs
        state = self.locals['new_obs'][0]  
        reward = self.locals['rewards'][0]   # SB3 returns rewards in locals
        actions=self.locals['infos'][0].get('t_motor',0)
        self.logger.record('env/reward', reward)
        self.logger.record('env/position', state[0])
        self.logger.record('env/velocity', state[1])
        self.logger.record('env/torques', actions)
        return True


# model training
vec_env=DummyVecEnv([make_env])
model=PPO(policy='MlpPolicy',env=vec_env,verbose=1,tensorboard_log=f"./tensorboard_logs_springpendulum/M={mass_on_top}, k={spring_const}",learning_rate=5e-5,normalize_advantage=True)
model.learn(total_timesteps=1000000,callback=LogStateCallback()) #1M steps
model.save("ppo_spring_pendulum_kspring=3.50Nm, M=0.0kg, Max_torque=2 Nm")