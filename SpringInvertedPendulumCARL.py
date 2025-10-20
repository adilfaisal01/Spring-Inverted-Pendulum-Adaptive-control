from typing import Optional

from carl.context.context_space import ContextFeature, UniformFloatContextFeature
from carl.envs.gymnasium.carl_gymnasium_env import CARLGymnasiumEnv
import numpy as np

# import envs


class CARLSpringInvertedPendulum(CARLGymnasiumEnv):
    env_name: str= "SpringInvertedPendulum-v0"
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    @staticmethod 
    #k_spring: spring constant in Nm, M: mass attached on top
    def get_context_features() -> dict [str, ContextFeature]:
        return {
            "k_spring": UniformFloatContextFeature(
                "k_spring", lower=4.745, upper=6.892,log=False, default_value=5.70
            ),
            "M": UniformFloatContextFeature(
                "M", lower=0.0, upper= 0.50, default_value=0.0
            ),
            "initial_state_lower": UniformFloatContextFeature(
                "initial_state_lower", lower=-np.pi/2, upper=np.deg2rad(-10),default_value=-0.2
            ),
            "initial_state_upper": UniformFloatContextFeature(
                "initial_state_upper", lower=np.deg2rad(10), upper= np.pi/2, default_value=0.2
            )
        }
    
   
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed, options=options)
        self.env.unwrapped.state = self.env.np_random.uniform(
            low=self.context.get(
                "initial_state_lower",
                self.get_context_features()["initial_state_lower"].default_value,
            ),
            high=self.context.get(
                "initial_state_upper",
                self.get_context_features()["initial_state_upper"].default_value,
            ),
            size=(2,),
        )
        state = np.array(self.env.unwrapped.state, dtype=np.float32)
        info = {}
        state = self._add_context_to_state(state)
        info["context_id"] = self.context_id
        return state, info

