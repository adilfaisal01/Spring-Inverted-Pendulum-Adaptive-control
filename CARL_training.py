import gymnasium as gym
from gymnasium.wrappers import FlattenObservation 
from SpringInvertedPendulumEnvs.envs.pendulum_code import SpringInvertedPendulum
from SpringInvertedPendulumCARL import CARLSpringInvertedPendulum

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from carl.context.context_space import UniformFloatContextFeature,NormalFloatContextFeature
from carl.context.sampler import ContextSampler

# print(f'Context features: {CARLSpringInvertedPendulum.get_context_space().context_feature_names}')
# print(f'Context features: {CARLSpringInvertedPendulum.get_context_space().get_default_context()}')
# print(f'Verify default context: {CARLSpringInvertedPendulum.get_context_space().verify_context(CARLSpringInvertedPendulum.get_context_space().get_default_context())}')

## creating environment-- randomly sampled contexts for the spring stiffness and masses on top, inspired by sa

context_distributions=[UniformFloatContextFeature(name="k_spring", lower=4.70, upper=6.90),
                       NormalFloatContextFeature(name='M', mu=0.0, sigma=0.30, lower=0.0,upper=0.50)]

contexts_sampler=ContextSampler(
    context_distributions=context_distributions,
    context_space=CARLSpringInvertedPendulum.get_context_space(),
    seed=12,
)

contexts=contexts_sampler.sample_contexts(n_contexts=15) # building 15 random worlds from the defined set

print(f' given contexts:{contexts}')

class LogStateCallback(BaseCallback):
    def _on_step(self) -> bool:
        # vectorized env of size 1
        carl_env = self.training_env.envs[0]  # CARL wrapper
        base_env = carl_env.env.unwrapped    # base env for state only

        # true pendulum state
        theta, theta_dot = base_env.state

        # reward
        reward = self.locals['rewards'][0]

        # context from CARL
        context = carl_env.context

        # log
        self.logger.record('env/reward', reward)
        self.logger.record('env/position', theta)
        self.logger.record('env/velocity', theta_dot)
        self.logger.record('context/M', context.get('M', 0.0))
        self.logger.record('context/k_spring', context.get('k_spring', 0.0))
        return True


env_carl=CARLSpringInvertedPendulum(contexts=contexts,env=None)
vec_env=DummyVecEnv([lambda: FlattenObservation(env_carl)])
model=PPO(policy='MlpPolicy',env=vec_env,verbose=1,tensorboard_log="./logs_tensorboard_carlSpringPendulum/pporuns", learning_rate=5e-5)
model.learn(total_timesteps=int(1e6),callback=LogStateCallback())
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
model.save("CARL_model_1")




