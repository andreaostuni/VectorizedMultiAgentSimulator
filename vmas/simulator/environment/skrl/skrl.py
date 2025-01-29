"""Wrappers to convert vmas envs to gym envs."""

from typing import ClassVar, Optional
import importlib

import numpy as np
from vmas.simulator.environment.environment import (
    Environment,
)

if (
    importlib.util.find_spec("gymnasium") is not None
    and importlib.util.find_spec("shimmy") is not None
):
    import gymnasium as gym
    from shimmy.openai_gym_compatibility import _convert_space
else:
    raise ImportError(
        "Gymnasium or shimmy is not installed. Please install it with `pip install gymnasium shimmy`."
    )


class SKRLWrapper(gym.Env):
    """A wrapper that converts an Envioronment to a Gym Env."""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: Environment, seed: int = 0):
        self._env = env
        self.seed(seed)
        self._state = None
        assert (
            env.num_envs == 1
        ), "GymnasiumEnv wrapper only supports singleton VMAS environment! For vectorized environments, use vectorized wrapper with `wrapper=gymnasium_vec`."
        assert (
            self._env.terminated_truncated
        ), "GymnasiumWrapper is only compatible with termination and truncation flags. Please set `terminated_truncated=True` in the VMAS environment."
        self.observation_space = _convert_space(self._env.observation_space)
        self.action_space = _convert_space(self._env.action_space)

    @property
    def unwrapped(self) -> Environment:
        return self._env

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._env.seed(seed)
        obs, info = self._env.reset_at(index=0, return_info=True)
        return obs, info

    def step(self, action):
        obs, rews, terminated, truncated, info = self._env.step(action)
        return (
            obs,
            rews,
            terminated,
            truncated,
            info,
        )

    def render(
        self,
        agent_index_focus: Optional[int] = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        return self._env.render(
            env_index=0,
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
            **kwargs,
        )
