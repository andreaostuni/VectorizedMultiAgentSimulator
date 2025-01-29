"""Wrappers to convert vmas envs to gym envs."""

from typing import Optional
import importlib

import warnings
import numpy as np
from vmas.simulator.environment.environment import Environment
import torch

if (
    importlib.util.find_spec("gymnasium") is not None
    and importlib.util.find_spec("shimmy") is not None
):
    import gymnasium as gym
    from gymnasium.vector.utils import batch_space
    from shimmy.openai_gym_compatibility import _convert_space
else:
    raise ImportError(
        "Gymnasium or shimmy is not installed. Please install it with `pip install gymnasium shimmy`."
    )


class SKRLVectorizedWrapper(gym.Env):
    metadata = Environment.metadata

    def __init__(
        self,
        env: Environment,
    ):
        self._env = env
        self._num_envs = self._env.num_envs
        assert (
            self._env.terminated_truncated
        ), "GymnasiumWrapper is only compatible with termination and truncation flags. Please set `terminated_truncated=True` in the VMAS environment."
        self.single_observation_space = _convert_space(self._env.observation_space)
        self.single_action_space = _convert_space(self._env.action_space)
        self.observation_space = batch_space(
            self.single_observation_space, n=self._num_envs
        )
        self.action_space = batch_space(self.single_action_space, n=self._num_envs)
        # warnings.warn(
        #     "The Gymnasium Vector wrapper currently does not have auto-resets or support partial resets."
        #     "We warn you that by using this class, individual environments will not be reset when they are done and you"
        #     "will only have access to global resets. We strongly suggest using the VMAS API unless your scenario does not implement"
        #     "the `done` function and thus all sub-environments are done at the same time."
        # )

    @property
    def unwrapped(self) -> Environment:
        return self._env

    def step(self, action):

        obs, rews, terminated, truncated, info = self._env.step(action)

        for i in torch.nonzero(terminated).squeeze(1):
            if self._env.dict_spaces:
                for key in obs.keys():
                    obs[key][i], info[key][i] = self._env.reset_at(
                        index=i, return_info=True
                    )
            else:
                for j in range(len(obs)):
                    obs[j][i], info[j][i] = self._env.reset_at(
                        index=i, return_info=True
                    )

        return (
            obs,
            rews,
            terminated,
            truncated,
            info,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._env.seed(seed)
        obs, info = self._env.reset(return_info=True)
        return obs, info

    def render(
        self,
        agent_index_focus: Optional[int] = None,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        return self._env.render(
            agent_index_focus=agent_index_focus,
            visualize_when_rgb=visualize_when_rgb,
            **kwargs,
        )
