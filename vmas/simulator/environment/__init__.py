#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from enum import Enum

from vmas.simulator.environment.environment import Environment
from vmas.simulator.environment.environment_mpc import EnvironmentMPC
from typing import Union


class Wrapper(Enum):
    RLLIB = 0
    GYM = 1
    GYMNASIUM = 2
    GYMNASIUM_VEC = 3
    SKRL_VEC = 4
    SKRL = 5
    SKRL_SINGLE_AGENT = 6
    SKRL_SINGLE_AGENT_VEC = 7
    SKRL_SINGLE_AGENT_MPC = 8
    SKRL_SINGLE_AGENT_VEC_MPC = 9

    def get_env(self, env: Union[Environment, EnvironmentMPC], **kwargs):
        if self is self.RLLIB:
            from vmas.simulator.environment.rllib import VectorEnvWrapper

            return VectorEnvWrapper(env, **kwargs)
        elif self is self.GYM:
            from vmas.simulator.environment.gym import GymWrapper

            return GymWrapper(env, **kwargs)
        elif self is self.GYMNASIUM:
            from vmas.simulator.environment.gym.gymnasium import GymnasiumWrapper

            return GymnasiumWrapper(env, **kwargs)
        elif self is self.GYMNASIUM_VEC:
            from vmas.simulator.environment.gym.gymnasium_vec import (
                GymnasiumVectorizedWrapper,
            )

            return GymnasiumVectorizedWrapper(env, **kwargs)
        elif self is self.SKRL_VEC:
            from vmas.simulator.environment.skrl.skrl_vec import SKRLVectorizedWrapper

            return SKRLVectorizedWrapper(env, **kwargs)
        elif self is self.SKRL:
            from vmas.simulator.environment.skrl.skrl import SKRLWrapper

            return SKRLWrapper(env, **kwargs)
        elif self is self.SKRL_SINGLE_AGENT:
            from vmas.simulator.environment.skrl.skrl_single_agent import (
                SKRLSingleAgentWrapper,
            )

            return SKRLSingleAgentWrapper(env, **kwargs)
        elif self is self.SKRL_SINGLE_AGENT_VEC:
            from vmas.simulator.environment.skrl.skrl_single_agent_vec import (
                SKRLSingleAgentVectorizedWrapper,
            )

            return SKRLSingleAgentVectorizedWrapper(env, **kwargs)

        elif self is self.SKRL_SINGLE_AGENT_MPC:
            from vmas.simulator.environment.skrl.skrl_single_agent import (
                SKRLSingleAgentWrapperMPC,
            )

            return SKRLSingleAgentWrapperMPC(env, **kwargs)
        elif self is self.SKRL_SINGLE_AGENT_VEC_MPC:
            from vmas.simulator.environment.skrl.skrl_single_agent_vec import (
                SKRLSingleAgentVectorizedWrapperMPC,
            )

            return SKRLSingleAgentVectorizedWrapperMPC(env, **kwargs)
