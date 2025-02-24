from vmas.simulator.environment import Environment
from gym import spaces
from typing import Union, List, Dict, Optional
from vmas.simulator.core import Agent
import numpy as np
from torch import Tensor

AGENT_MPC_STATE_TYPE = Union[Tensor, Dict[str, Tensor]]


class EnvironmentMPC(Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mpc_state = self.get_mpc_state_from_scenario()
        self.mpc_state_space = self.get_mpc_state_space(mpc_state)

    def get_mpc_state_space(self, mpc_state: Union[List, Dict]):
        if not self.dict_spaces:
            return spaces.Tuple(
                [
                    self.get_agent_mpc_state_space(agent, mpc_state[agent.name])
                    for i, agent in enumerate(self.agents)
                ]
            )
        else:
            return spaces.Dict(
                {
                    agent.name: self.get_agent_mpc_state_space(
                        agent, mpc_state[agent.name]
                    )
                    for agent in self.agents
                }
            )

    def get_agent_mpc_state_space(self, agent: Agent, mpc_state: AGENT_MPC_STATE_TYPE):
        if isinstance(mpc_state, Tensor):
            return spaces.Box(
                low=-np.float32("inf"),
                high=np.float32("inf"),
                shape=mpc_state.shape[1:],
                dtype=np.float32,
            )
        elif isinstance(mpc_state, Dict):
            return spaces.Dict(
                {
                    key: self.get_agent_mpc_state_space(agent, value)
                    for key, value in mpc_state.items()
                }
            )
        else:
            raise NotImplementedError(
                f"Invalid type of observation {mpc_state} for agent {agent.name}"
            )

    def get_mpc_state_from_scenario(
        self,
        dict_agent_names: Optional[bool] = None,
    ):

        if dict_agent_names is None:
            dict_agent_names = self.dict_spaces
        mpc_states = {} if dict_agent_names else []
        for agent in self.agents:
            mpc_state = self.scenario.mpc_state(agent)
            if dict_agent_names:
                mpc_states.update({agent.name: mpc_state})
            else:
                mpc_states.append(mpc_state)
        return mpc_states

    def mpc_state(self):
        return self.get_mpc_state_from_scenario()
