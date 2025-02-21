import torch

from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Box, Line
from typing import Optional, List, Tuple
from vmas.simulator.utils import Color, ScenarioUtils
from torch import Tensor
from vmas.simulator.human_dynamics.policies import utils as policy_utils
from abc import ABC, abstractmethod


def create_boundaries(
    world: World, x_semidim: float, y_semidim: float
) -> List[Landmark]:
    """
    Generate boundaries for the environment
    putting line obstacles on the edges of the environment

    Args:
        world: World object
    """

    # Add landmarks

    boundaries = []

    for i in range(4):
        length = 2 * (x_semidim if i % 2 == 0 else y_semidim)

        boundary = Landmark(
            name=f"obstacle boundary {i}",
            collide=True,
            movable=False,
            shape=Line(length=length),
            color=Color.RED,
        )
        world.add_landmark(boundary)
        boundaries.append(boundary)

    return boundaries


def spawn_boundaries(
    boundaries: List[Landmark],
    x_semidim: float,
    y_semidim: float,
    device: torch.device,
    env_index: Optional[int] = None,
) -> None:

    for i, boundary in enumerate(boundaries):
        if i % 2 == 1:
            boundary.set_pos(
                torch.tensor(
                    [
                        -x_semidim if i == 1 else x_semidim,
                        0.0,
                    ],
                    dtype=torch.float32,
                    device=device,
                ),
                batch_index=env_index,
            )
            boundary.set_rot(
                torch.tensor(
                    [torch.pi / 2],
                    dtype=torch.float32,
                    device=device,
                ),
                batch_index=env_index,
            )
            boundary.shape._length = y_semidim * 2

        else:
            boundary.set_pos(
                torch.tensor(
                    [
                        0.0,
                        -y_semidim if i == 0 else y_semidim,
                    ],
                    dtype=torch.float32,
                    device=device,
                ),
                batch_index=env_index,
            )
            boundary.shape._length = x_semidim * 2


def create_random_obstacles(
    world: World, n_box_obstacles: int, n_sphere_obstacles: int, agent_radius: float
) -> Tuple[List[Landmark], List[Landmark]]:
    """
    Create random obstacles in the environment

    Args:
        world: World object
    """

    # Add landmarks
    box_obstacles = []
    sphere_obstacles = []

    for i in range(n_box_obstacles):
        obstacle = Landmark(
            name=f"obstacle {i}",
            collide=True,
            movable=False,
            shape=Box(
                length=agent_radius * 2,
                width=agent_radius * 2,
            ),
            color=Color.RED,
        )
        box_obstacles.append(obstacle)
        world.add_landmark(obstacle)

    for i in range(n_sphere_obstacles):
        obstacle = Landmark(
            name=f"obstacle {i + n_box_obstacles}",
            collide=True,
            movable=False,
            shape=Sphere(radius=agent_radius),
            color=Color.RED,
        )
        sphere_obstacles.append(obstacle)
        world.add_landmark(obstacle)

    return box_obstacles, sphere_obstacles


def spawn_random_obstacles(
    env_index: int,
    occupied_positions: Tensor,
    world: World,
    box_obstacles: List[Landmark],
    sphere_obstacles: List[Landmark],
    min_distance_between_entities: float,
    world_spawning_x: float,
    world_spawning_y: float,
) -> None:
    """
    Spawn random obstacles in the environment

    Args:
        env_index: Environment index
        occupied_positions: Occupied positions in the environment
    """

    for i, obstacle in enumerate(box_obstacles + sphere_obstacles):
        position = ScenarioUtils.find_random_pos_for_entity(
            occupied_positions=occupied_positions,
            env_index=env_index,
            world=world,
            min_dist_between_entities=min_distance_between_entities * 1.5,
            x_bounds=(-world_spawning_x, world_spawning_x),
            y_bounds=(-world_spawning_y, world_spawning_y),
        )
        obstacle.set_pos(position.squeeze(1), batch_index=env_index)
        if isinstance(obstacle.shape, Box):
            obstacle.set_rot(
                torch.tensor(
                    [torch.rand(1).item() * 2 * torch.pi],
                    dtype=torch.float32,
                    device=world.device,
                ),
                batch_index=env_index,
            )
        occupied_positions = torch.cat([occupied_positions, position], dim=1)


class AgentsStateBuffer:
    """Class to store the state of the entities in the environment

    the buffer is a tensor of shape (n_envs, buffer_size, n_agents, n_features)
    """

    def __init__(self, buffer_size: int, world: World):
        self.device = world.device
        self.buffer_size = buffer_size
        self.buffer = None
        self.index = torch.zeros(world.batch_dim, device=self.device)

    def add_state(self, state: Tensor) -> None:
        """
        Add a state to the buffer

        """
        if self.buffer is None:
            self.buffer = torch.full(
                (state.shape[0], self.buffer_size, *tuple(state.shape[1:])),
                device=self.device,
                dtype=state.dtype,
                fill_value=float("nan"),
            )
        # self.buffer[:, self.index, :] = state
        self.buffer[torch.arange(state.shape[0]), self.index.long()] = state
        self.index = (self.index + 1) % self.buffer_size

    def get_state_at_step(self, t: int) -> Tensor:
        """
        Get the state at t step before the current state
        """
        if t >= self.buffer_size:
            raise ValueError("t should be less than buffer_size")
        if torch.isnan(self.buffer[:, (self.index - t) % self.buffer_size, :]).any():
            raise ValueError("State at t is nan")
        return self.buffer[:, (self.index - t) % self.buffer_size, :, :]

    def get_last_state(self) -> Tensor:
        """
        Get the last state in the buffer
        """
        if torch.isnan(
            self.buffer[torch.arange(self.buffer.shape[0]), (self.index - 1).long(), :]
        ).any():
            raise ValueError("Last state is nan")
        return self.buffer[
            torch.arange(self.buffer.shape[0]), (self.index - 1).long(), :, :
        ]

    def reset(self, env_index: Optional[int] = None) -> None:
        """
        Reset the buffer
        """
        if env_index is None:
            self.buffer = None
            self.index = torch.zeros_like(self.index)
        else:
            self.reset_at(env_index)

    def reset_at(self, env_index: int) -> None:
        """
        Reset the buffer at a specific environment index
        fill the buffer with nan values
        """
        self.buffer[env_index] = torch.full_like(
            self.buffer[env_index], fill_value=float("nan")
        )
        self.index[env_index] = 0

    def get_agent_last_state(self, agent_index: int) -> Tensor:
        """
        Get the last state of an agent
        """
        return self.get_last_state()[:, agent_index, :]

    def get_agent_state_at_step(self, agent_index: int, t: int) -> Tensor:
        """
        Get the state of an agent at t step before the current state
        """
        return self.get_state_at_step(t)[:, agent_index, :]
