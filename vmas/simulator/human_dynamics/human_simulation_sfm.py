from vmas.simulator.core import Agent, World, Line, Sphere, Box
import torch
from vmas.simulator.human_dynamics.policies.sfm import SocialForcePolicy
from vmas.simulator.human_dynamics.policies.utils import (
    box_to_tensor,
    line_to_tensor,
    sphere_to_tensor,
    agents_to_tensor,
)


class HumanSimulation:

    def __init__(
        self,
        time_step: float,
        neighbor_dist: float,
        max_neighbors: int,
        time_horizon: float,
        time_horizon_obst: float,
        radius: float,
        max_speed: float,
        world: World,
        safety_space: float = 0.1,
    ):
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.radius = radius
        self.max_speed = max_speed

        self.robots = []  # robots are agents with policies
        self.humans = []  # humans are agents with scripts
        self.agents_tensor = None
        self.box_obstacles_tensor = None
        self.sphere_obstacles_tensor = None
        self.line_obstacles_tensor = None

        self.new_states = None

        self.safety_space = safety_space
        self.batch_dim = world.batch_dim
        self.device = world.device

        self.policy = SocialForcePolicy(
            time_step,
            neighbor_dist,
            max_neighbors,
            time_horizon,
            time_horizon_obst,
            radius,
            max_speed,
            device=self.device,
            safety_space=safety_space,
        )

    def reset(self, world: World):
        """
        Reset the simulation.

        :param world: The world to simulate.
        :return:
        """

        self.robots = world.policy_agents
        self.humans = world.scripted_agents

        self.agents_tensor = agents_to_tensor(
            self.robots + self.humans
        )  # (batch_dim, n_agents, 8)

        box_obstacles = []
        sphere_obstacles = []
        line_obstacles = []

        for landmark in world.landmarks:

            # if the landmark name contains "obstacle", add it as an obstacle
            if landmark.collide:
                if isinstance(landmark.shape, Sphere):
                    sphere_obstacles.append(
                        sphere_to_tensor(landmark.state, landmark.shape)
                    )
                elif isinstance(landmark.shape, Line):
                    line_obstacles.append(
                        line_to_tensor(landmark.state, landmark.shape)
                    )
                elif isinstance(landmark.shape, Box):
                    box_obstacles.append(box_to_tensor(landmark.state, landmark.shape))
                else:
                    raise ValueError("Obstacle shape not recognized")

        if len(box_obstacles) > 0:
            # box_obstacles_tensor: (batch_dim, n_obstacles, 4, 2)
            # box_obstacles: list of tensors of shape (batch_dim, 4, 2)
            self.box_obstacles_tensor = torch.stack(box_obstacles, dim=1).to(
                self.device
            )  # (batch_dim, n_obstacles, 4, 2)
        if len(sphere_obstacles) > 0:
            self.sphere_obstacles_tensor = torch.stack(sphere_obstacles, dim=1).to(
                self.device
            )  # (batch_dim, n_obstacles, 3)
        if len(line_obstacles) > 0:
            self.line_obstacles_tensor = torch.stack(line_obstacles, dim=1).to(
                self.device
            )  # (batch_dim, n_obstacles, 2, 2)
        print("Simulation reset")

    def simulate_policy(self, world: World):
        """
        Run the simulation for one time step.

        :return:
        """
        if self.agents_tensor is None:
            raise ValueError("Agents must be set before simulation")

        self.old_agent_tensor = agents_to_tensor(
            self.robots + self.humans
        )  # (batch_dim, n_agents, 8)

        self.agents_tensor = self.policy(
            self.old_agent_tensor,
            self.box_obstacles_tensor,
            self.sphere_obstacles_tensor,
            self.line_obstacles_tensor,
        )

    def find_agent_index(self, agent: Agent):
        """
        Find the index of the agent in the simulation.
        """
        for i, human in enumerate(self.robots + self.humans):
            if human is agent:
                return i
        raise ValueError("Agent not found in simulation")

    def run(self, agent: Agent, world: World):
        """
        Return the action to take for the agent.

        :param agent: The agent to run the simulation for.
        :param world: The world to simulate.
        """
        index = self.find_agent_index(agent)

        if self.agents_tensor is None:
            raise ValueError("Agents must be set before simulation")

        control = self.agents_tensor[:, index, 2:4]  # (batch_dim, agents, 2)

        control = torch.clamp(control, min=-agent.u_range, max=agent.u_range)
        agent.action.u = control * agent.u_multiplier
