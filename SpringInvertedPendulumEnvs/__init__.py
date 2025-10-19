from SpringInvertedPendulumEnvs.envs.pendulum_code import SpringInvertedPendulum
from gymnasium.envs.registration import register

register(
    id="SpringInvertedPendulum-v0",
    entry_point="SpringInvertedPendulumEnvs.envs.pendulum_code:SpringInvertedPendulum",
)
