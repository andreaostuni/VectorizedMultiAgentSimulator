"""Wrappers to convert vmas envs to gym envs."""

from typing import Optional
import importlib

import warnings
import torch
import numpy as np
from vmas.simulator.environment.environment import Environment
from vmas.simulator.environment.environment_mpc import EnvironmentMPC

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


class SKRLSingleAgentVectorizedWrapper(gym.Env):
    """Wrappers to convert vmas envs to gym envs for single agent."""

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
        # Convert the observation space to gym space and
        # access the first key of the dictionary
        if self._env.dict_spaces:
            self.first_key = list(self._env.observation_space.keys())[0]

        if self._env.dict_spaces:
            # get the first key of the dictionary
            self.single_observation_space = _convert_space(
                self._env.observation_space[self.first_key]
            )
            self.single_action_space = _convert_space(
                self._env.action_space[self.first_key]
            )
        else:
            self.single_observation_space = _convert_space(
                self._env.observation_space[0]
            )
            self.single_action_space = _convert_space(self._env.action_space[0])

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

        action = {self.first_key: action} if self._env.dict_spaces else [action]
        obs, rews, terminated, truncated, info = self._env.step(action)

        rews = rews[self.first_key if self._env.dict_spaces else 0].unsqueeze(-1)
        terminated = terminated.unsqueeze(-1)
        truncated = truncated.unsqueeze(-1)

        if torch.any(terminated | truncated):
            if torch.all(terminated | truncated):
                # reset the environment if all the environments are done
                obs, info = self._env.reset(return_info=True)
            else:
                # reset all the environment which are terminated or truncated
                for i in torch.nonzero(
                    terminated.squeeze(-1) | truncated.squeeze(-1)
                ).flatten():
                    obs, info = self._env.reset_at(index=i.item(), return_info=True)

        obs = obs[self.first_key if self._env.dict_spaces else 0]
        info = info[self.first_key if self._env.dict_spaces else 0]
        return obs, rews, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._env.seed(seed)
        obs, info = self._env.reset(return_info=True)

        if self._env.dict_spaces:
            return obs[self.first_key], info[self.first_key]
        return obs[0], info[0]

    def render(
        self,
        visualize_when_rgb: bool = False,
        **kwargs,
    ) -> Optional[np.ndarray]:
        return self._env.render(
            agent_index_focus=0,
            visualize_when_rgb=visualize_when_rgb,
            **kwargs,
        )


class SKRLSingleAgentVectorizedWrapperMPC(SKRLSingleAgentVectorizedWrapper):
    """Wrappers to convert vmas envs to gym envs for single agent with MPC."""

    metadata = Environment.metadata

    def __init__(
        self,
        env: EnvironmentMPC,
    ):
        super().__init__(env)

        if self._env.dict_spaces:
            self.first_key = list(self._env.mpc_state_space.keys())[0]

        if self._env.dict_spaces:
            # get the first key of the dictionary
            self.single_mpc_state_space = _convert_space(
                self._env.mpc_state_space[self.first_key]
            )
        else:
            self.single_mpc_state_space = _convert_space(self._env.mpc_state_space[0])

        self.mpc_state_space = batch_space(
            self.single_mpc_state_space, n=self._num_envs
        )

    def mpc_state(self):
        return self._env.mpc_state()[self.first_key if self._env.dict_spaces else 0]
