#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union

import torch

import vmas.simulator.core
from vmas.simulator.utils import Color

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Sensor(ABC):
    def __init__(self, world: vmas.simulator.core.World):
        super().__init__()
        self._world = world
        self._agent: Union[vmas.simulator.core.Agent, None] = None

    @property
    def agent(self) -> Union[vmas.simulator.core.Agent, None]:
        return self._agent

    @agent.setter
    def agent(self, agent: vmas.simulator.core.Agent):
        self._agent = agent

    @abstractmethod
    def measure(self):
        raise NotImplementedError

    @abstractmethod
    def render(self, env_index: int = 0) -> "List[Geom]":
        raise NotImplementedError

    def to(self, device: torch.device):
        raise NotImplementedError


class Lidar(Sensor):
    def __init__(
        self,
        world: vmas.simulator.core.World,
        angle_start: float = 0.0,
        angle_end: float = 2 * torch.pi,
        n_rays: int = 8,
        max_range: float = 1.0,
        entity_filter: Callable[[vmas.simulator.core.Entity], bool] = lambda _: True,
        render_color: Union[Color, Tuple[float, float, float]] = Color.GRAY,
        alpha: float = 1.0,
        render: bool = True,
    ):
        super().__init__(world)
        if (angle_start - angle_end) % (torch.pi * 2) < 1e-5:
            angles = torch.linspace(
                angle_start, angle_end, n_rays + 1, device=self._world.device
            )[:n_rays]
        else:
            angles = torch.linspace(
                angle_start, angle_end, n_rays, device=self._world.device
            )

        self._angles = angles.repeat(self._world.batch_dim, 1)
        self._max_range = max_range
        self._last_measurement = None
        self._render = render
        self._entity_filter = entity_filter
        self._render_color = render_color
        self._alpha = alpha

    def to(self, device: torch.device):
        self._angles = self._angles.to(device)

    @property
    def entity_filter(self):
        return self._entity_filter

    @entity_filter.setter
    def entity_filter(
        self, entity_filter: Callable[[vmas.simulator.core.Entity], bool]
    ):
        self._entity_filter = entity_filter

    @property
    def render_color(self):
        if isinstance(self._render_color, Color):
            return self._render_color.value
        return self._render_color

    @property
    def alpha(self):
        return self._alpha

    def measure(self, vectorized: bool = True):
        if not vectorized:
            dists = []
            for angle in self._angles.unbind(1):
                dists.append(
                    self._world.cast_ray(
                        self.agent,
                        angle + self.agent.state.rot.squeeze(-1),
                        max_range=self._max_range,
                        entity_filter=self.entity_filter,
                    )
                )
            measurement = torch.stack(dists, dim=1)

        else:
            measurement = self._world.cast_rays(
                self.agent,
                self._angles + self.agent.state.rot,
                max_range=self._max_range,
                entity_filter=self.entity_filter,
            )
        self._last_measurement = measurement
        return measurement

    def set_render(self, render: bool):
        self._render = render

    def render(self, env_index: int = 0) -> "List[Geom]":
        if not self._render:
            return []
        from vmas.simulator import rendering

        geoms: List[rendering.Geom] = []
        if self._last_measurement is not None:
            for angle, dist in zip(
                self._angles.unbind(1), self._last_measurement.unbind(1)
            ):
                angle = angle[env_index] + self.agent.state.rot.squeeze(-1)[env_index]
                ray = rendering.Line(
                    (0, 0),
                    (dist[env_index], 0),
                    width=0.05,
                )
                xform = rendering.Transform()
                xform.set_translation(*self.agent.state.pos[env_index])
                xform.set_rotation(angle)
                ray.add_attr(xform)
                ray.set_color(r=0, g=0, b=0, alpha=self.alpha)

                ray_circ = rendering.make_circle(0.01)
                ray_circ.set_color(*self.render_color, alpha=self.alpha)
                xform = rendering.Transform()
                rot = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
                pos_circ = (
                    self.agent.state.pos[env_index] + rot * dist.unsqueeze(1)[env_index]
                )
                xform.set_translation(*pos_circ)
                ray_circ.add_attr(xform)

                geoms.append(ray)
                geoms.append(ray_circ)
        return geoms


class AgentsPoses(Sensor):
    """Sensor that measures the poses of other agents in the world"""

    def __init__(
        self,
        world: vmas.simulator.core.World,
        entity_filter: Callable[[vmas.simulator.core.Entity], bool] = lambda _: True,
        neighbors: int = 1,
    ):
        super().__init__(world)
        self._entity_filter = entity_filter
        self._k = neighbors

    @property
    def entity_filter(self):
        return self._entity_filter

    @entity_filter.setter
    def entity_filter(
        self, entity_filter: Callable[[vmas.simulator.core.Entity], bool]
    ):
        self._entity_filter = entity_filter

    def find_k_agents_distances(
        self,
        entity_filter: Callable[[vmas.simulator.core.Entity], bool] = lambda _: False,
    ):
        """

        Returns the indexes of the k nearest agents in the obs_range,
        excluding the agent itself,
        and filtered by the entity_filter,
        if the k nearest agents are less than k,
        the rest of the indexes are filled with -1

        """
        agents = []
        distances = []
        for i, agent in enumerate(self._world.agents):
            if agent is self.agent or entity_filter(agent):
                continue
            dist = (agent.state.pos - self.agent.state.pos).norm(dim=-1)
            # if the agent is in the observation range
            # since it is a tensor, we need to use the all method
            if (dist > self.agent._obs_range).all():
                continue
            dist = torch.where(
                dist < self.agent._obs_range,
                dist,
                torch.tensor(torch.inf, device=self._world.device),
            )

            agent_index = torch.full(
                (self._world.batch_dim,),
                fill_value=i,
                dtype=torch.int64,
                device=self._world.device,
            )

            agent_index = torch.where(
                dist < self.agent._obs_range, agent_index, torch.tensor(-1)
            )

            # for the agents that are not in the observation range
            distances.append(dist)
            agents.append(agent_index)

        # sort the agents by distance

        if len(agents) < self._k:
            n_pad = self._k - len(agents)
            agent_pad_tensor = torch.full(
                (self._world.batch_dim,),
                fill_value=-1,
                dtype=torch.int64,
                device=self._world.device,
            )
            dist_pad_tensor = torch.full(
                (self._world.batch_dim,),
                fill_value=torch.inf,
                device=self._world.device,
            )
            agents.extend([agent_pad_tensor] * n_pad)
            distances.extend([dist_pad_tensor] * n_pad)
        agents = torch.stack(agents)
        distances = torch.stack(distances)
        _, indexes = torch.topk(distances, self._k, largest=False, dim=0, sorted=True)
        return torch.gather(agents, 0, indexes).permute(1, 0), torch.gather(
            distances, 0, indexes
        ).permute(1, 0)

    def measure(self):
        """Returns the poses of the k nearest agents in the obs_range"""
        indexes, distances = self.find_k_agents_distances()
        # add axis to the indexes
        indexes_agents = indexes.unsqueeze(-1)
        agents = self._world.agents
        agents = torch.stack([agent.state.pos for agent in agents], dim=1)

        # if some indexes_agents are -1, we need to fill them with zeros
        # to avoid indexing errors
        # we need to use the where method to avoid indexing errors

        poses = torch.gather(
            agents, 1, indexes_agents.clip(min=0).expand(-1, -1, 2)
        )  # (batch_dim, k, 2)
        velocities = torch.gather(
            agents, 1, indexes_agents.clip(min=0).expand(-1, -1, 2)
        )  # (batch_dim, k, 2)
        # replace the -1 indexes with zeros

        relative_poses = poses - self.agent.state.pos.unsqueeze(1)  # (batch_dim, k, 2)
        angles = torch.atan2(relative_poses[..., 1], relative_poses[..., 0])
        velocities_norms = velocities.norm(dim=-1)

        # if torch.linalg.norm(self.agent.state.vel, dim=-1) < 1e-5:
        #     angles_velocities = torch.atan2(
        #         velocities[..., 1], velocities[..., 0]
        #     ) - torch.atan2(self.agent.state.rot.sin(), self.agent.state.rot.cos())
        # else:
        #     angles_velocities = torch.atan2(velocities[..., 1], velocities[..., 0]) - (
        #         torch.atan2(self.agent.state.vel[..., 1], self.agent.state.vel[..., 0])
        #     )
        zero_vel = torch.where(
            torch.linalg.norm(self.agent.state.vel, dim=-1) < 1e-5,
            torch.tensor(True, device=self._world.device),
            torch.tensor(False, device=self._world.device),
        )
        angles_velocities = torch.atan2(
            velocities[..., 1], velocities[..., 0]
        ) - torch.atan2(
            self.agent.state.vel[..., 1], self.agent.state.vel[..., 0]
        ).unsqueeze(
            -1
        )
        angles_velocities[zero_vel] = torch.atan2(
            velocities[zero_vel, ..., 1], velocities[zero_vel, ..., 0]
        ) - torch.atan2(
            self.agent.state.rot[zero_vel].sin(), self.agent.state.rot[zero_vel].cos()
        )

        if (indexes == -1).any():
            # where the indexes are -1, the poses and velocities are zeros
            angles[indexes == -1] = 0.0
            distances[indexes == -1] = 0.0
            velocities_norms[indexes == -1] = 0.0
            angles_velocities[indexes == -1] = 0.0
        return torch.stack(
            [distances, angles, velocities_norms, angles_velocities], dim=-1
        ).flatten(
            start_dim=1
        )  # (batch_dim, 2 * k)

    # TODO: Implement the to method
    # use the same implementation of the applr gym
    # agents distance and relative angle
    # filtered to the nearest k and capped by the observation range
    # do not include the agent itself
    # return a tensor of shape (batch_dim, 2 * k)
    # the first element of each pair is the distance
    # the second element of each pair is the relative angle

    def render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        """Renders the poses of all agents in the world as a line that connects the agents position to the agent in exam position"""
        geoms: List[rendering.Geom] = []
        agents = [
            agent
            for agent in self._world.agents
            if agent is not self.agent and not self._entity_filter(agent)
        ]
        # Render the line between the agent and the other agents in the observation range
        # only render the k nearest agents
        # sorted by distance
        agents.sort(
            key=lambda agent: (agent.state.pos - self.agent.state.pos).norm(dim=-1)
        )

        for agent in agents[: self._k]:
            line = rendering.Line(
                self.agent.state.pos[env_index],
                agent.state.pos[env_index],
                width=0.05,
            )
            line.set_color(1, 0, 0)
            geoms.append(line)
        return geoms

    def to(self, device: torch.device):
        pass
