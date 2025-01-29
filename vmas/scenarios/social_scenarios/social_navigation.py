#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Box, Line
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar, AgentsPoses
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic import Holonomic

from vmas.simulator.human_dynamics.human_simulation_sfm import HumanSimulation

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

from typing import Optional


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.fixed_passage = kwargs.pop("fixed_passage", False)
        self.n_agents = kwargs.pop("n_agents", 1)

        self.n_scripted_agents = kwargs.pop("n_scripted_agents", 4)  # scripted agents
        self.collisions = kwargs.pop("collisions", True)

        self.world_spawning_x = kwargs.pop(
            "world_spawning_x", 2.5
        )  # X-coordinate limit for entities spawning
        self.world_spawning_y = kwargs.pop(
            "world_spawning_y", 2.5
        )  # Y-coordinate limit for entities spawning
        self.enforce_bounds = kwargs.pop(
            "enforce_bounds", True
        )  # If False, the world is unlimited; else, constrained by world_spawning_x and world_spawning_y.

        self.agents_with_same_goal = kwargs.pop("agents_with_same_goal", 1)
        self.split_goals = kwargs.pop("split_goals", False)
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)

        self.lidar_range = kwargs.pop("lidar_range", 1.0)
        self.agent_radius = kwargs.pop("agent_radius", 0.25)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 36)

        self.n_box_obstacles = kwargs.pop("n_box_obstacles", 3)
        self.n_sphere_obstacles = kwargs.pop("n_sphere_obstacles", 2)
        self.middle_angle_180 = kwargs.pop("middle_angle_180", False)

        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2
        self.min_collision_distance = 0.005

        if self.enforce_bounds:
            self.x_semidim = self.world_spawning_x + self.min_distance_between_entities
            self.y_semidim = self.world_spawning_y + self.min_distance_between_entities
        else:
            self.x_semidim = None
            self.y_semidim = None

        assert 1 <= self.agents_with_same_goal <= self.n_agents
        if self.agents_with_same_goal > 1:
            assert (
                not self.collisions
            ), "If agents share goals they cannot be collidables"
        # agents_with_same_goal == n_agents: all agent same goal
        # agents_with_same_goal = x: the first x agents share the goal
        # agents_with_same_goal = 1: all independent goals
        if self.split_goals:
            assert (
                self.n_agents % 2 == 0
                and self.agents_with_same_goal == self.n_agents // 2
            ), "Splitting the goals is allowed when the agents are even and half the team has the same goal"

        # Make world
        world = World(
            batch_dim,
            device,
            substeps=2,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
        )

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        # Random colors for agents till the known colors are exhausted

        if self.n_agents + self.n_scripted_agents > len(known_colors):
            # random tuples of 3
            for i in range(self.n_agents + self.n_scripted_agents - len(known_colors)):
                known_colors.append(
                    (torch.rand(1).item(), torch.rand(1).item(), torch.rand(1).item())
                )

        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(
            e, Agent
        ) or (isinstance(e, Landmark) and e.collide)

        self.human_simulation = HumanSimulation(
            time_step=world.dt,
            neighbor_dist=self.lidar_range,
            max_neighbors=3,
            time_horizon=30 * world.dt,
            time_horizon_obst=10 * world.dt,
            radius=self.agent_radius,
            max_speed=0.5,
            world=world,
            safety_space=0.1,
        )

        self.middle_angle = torch.zeros((world.batch_dim, 1), device=world.device)
        self.passage_width = self.agent_radius
        self.passage_length = self.agent_radius * 3 + 0.1
        self.scenario_length = 2 + 2 * self.agent_radius
        self.n_boxes = int(self.scenario_length // self.passage_length)

        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                obs_range=5.0,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=self.n_lidar_rays,
                            max_range=self.lidar_range,
                            entity_filter=entity_filter_agents,
                        ),
                        AgentsPoses(
                            world,
                            entity_filter=entity_filter_agents,
                            neighbors=3,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
                dynamics=DiffDrive(world, integration="rk4"),
            )

            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # Add goals
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal

        for i in range(self.n_agents, self.n_agents + self.n_scripted_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=self.n_lidar_rays,
                            max_range=self.lidar_range,
                            entity_filter=entity_filter_agents,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
                dynamics=Holonomic(),
                action_script=self.human_simulation.run,
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # Add goals
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal

        # Add obstacles
        self.create_boundaries(world)
        self.create_random_obstacles(world)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()
        return world

    def create_boundaries(self, world: World):
        """
        Generate boundaries for the environment
        putting line obstacles on the edges of the environment

        Args:
            world: World object
        """

        # Add landmarks

        self.boundaries = []

        for i in range(4):
            length = 2 * (self.x_semidim if i % 2 == 0 else self.y_semidim)

            boundary = Landmark(
                name=f"obstacle boundary {i}",
                collide=True,
                movable=False,
                shape=Line(length=length),
                color=Color.RED,
            )
            world.add_landmark(boundary)
            self.boundaries.append(boundary)

    def spawn_boundaries(self, env_index: Optional[int] = None):

        for i, boundary in enumerate(self.boundaries):
            if i % 2 == 0:
                boundary.set_pos(
                    torch.tensor(
                        [
                            -self.x_semidim if i == 0 else self.x_semidim,
                            0.0,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                boundary.set_rot(
                    torch.tensor(
                        [torch.pi / 2],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

            else:
                boundary.set_pos(
                    torch.tensor(
                        [
                            0.0,
                            -self.y_semidim if i == 1 else self.y_semidim,
                        ],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

    def create_random_obstacles(self, world: World):
        """
        Create random obstacles in the environment

        Args:
            world: World object
        """

        # Add landmarks
        self.box_obstacles = []
        self.sphere_obstacles = []

        for i in range(self.n_box_obstacles):
            obstacle = Landmark(
                name=f"obstacle {i}",
                collide=True,
                movable=False,
                shape=Box(
                    length=self.agent_radius * 2,
                    width=self.agent_radius * 2,
                ),
                color=Color.RED,
            )
            self.box_obstacles.append(obstacle)
            world.add_landmark(obstacle)

        for i in range(self.n_sphere_obstacles):
            obstacle = Landmark(
                name=f"obstacle {i + self.n_box_obstacles}",
                collide=True,
                movable=False,
                shape=Sphere(radius=self.agent_radius),
                color=Color.RED,
            )
            self.sphere_obstacles.append(obstacle)
            world.add_landmark(obstacle)

    def spawn_random_obstacles(self, env_index: int, occupied_positions: Tensor):
        """
        Spawn random obstacles in the environment

        Args:
            env_index: Environment index
            occupied_positions: Occupied positions in the environment
        """

        for i, obstacle in enumerate(self.box_obstacles + self.sphere_obstacles):
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities * 2,
                x_bounds=(-self.world_spawning_x, self.world_spawning_x),
                y_bounds=(-self.world_spawning_y, self.world_spawning_y),
            )
            obstacle.set_pos(position.squeeze(1), batch_index=env_index)
            if isinstance(obstacle.shape, Box):
                obstacle.set_rot(
                    torch.tensor(
                        [torch.rand(1).item() * 2 * torch.pi],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
            occupied_positions = torch.cat([occupied_positions, position], dim=1)

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (
                -self.world_spawning_x,
                self.world_spawning_x,
            ),
            (
                -self.world_spawning_y,
                self.world_spawning_y,
            ),
        )

        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents]
            + [
                landmark.state.pos
                for landmark in self.world.landmarks
                if landmark.collide
            ],
            dim=1,
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.world_spawning_x, self.world_spawning_x),
                y_bounds=(-self.world_spawning_y, self.world_spawning_y),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)

        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i

            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )
        # self.spawn_passage_map(env_index)
        if self.enforce_bounds:
            self.spawn_boundaries(env_index)

        self.spawn_random_obstacles(env_index, occupied_positions)
        self.human_simulation.reset(self.world)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )

            self.final_rew[self.all_goal_reached] = self.final_reward

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def observation(self, agent: Agent):
        goal_poses = []
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
        return torch.cat(
            [
                agent.state.vel,
            ]
            + goal_poses
            + ([agent.sensors[0].measure()] if self.collisions else [])
            + ([agent.sensors[1].measure()] if len(agent.sensors) > 1 else []),
            dim=-1,
        )

    def pre_step(self):
        self.human_simulation.simulate_policy(self.world)
        super().pre_step()

    def done(self):
        """
        This environment is done when all agents reach their goals
        in a batched environment we check if all agents in their respective environments have reached their goals
        """
        # add collision check
        is_done = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.bool
        )
        is_collision_with_obstacle = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.bool
        )

        is_collision_with_agents = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.bool
        )

        is_done = (
            is_collision_with_agents
            | is_collision_with_obstacle
            | self.all_goal_reached
        )
        return is_done

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, clf_epsilon=0.2, clf_slack=100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_epsilon = clf_epsilon  # Exponential CLF convergence rate
        self.clf_slack = clf_slack  # weights on CLF-QP slack variable

    def compute_action(self, observation: Tensor, u_range: Tensor) -> Tensor:
        """
        QP inputs:
        These values need to computed apriri based on observation before passing into QP

        V: Lyapunov function value
        lfV: Lie derivative of Lyapunov function
        lgV: Lie derivative of Lyapunov function
        CLF_slack: CLF constraint slack variable

        QP outputs:
        u: action
        CLF_slack: CLF constraint slack variable, 0 if CLF constraint is satisfied
        """
        # Install it with: pip install cvxpylayers
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer

        self.n_env = observation.shape[0]
        self.device = observation.device
        agent_vel = observation[:, 0:2]

        goal_pos = (-1.0) * (observation[:, 2:4])
        # Lidar measurements
        lidar_measurements = observation[:, 4 : 4 + 12]
        lidar_measurements = lidar_measurements.view(self.n_env, -1, 1)

        # Other agents positions in range bearing format
        other_agents_pos = observation[:, 4 + 12 :]
        other_agents_pos = other_agents_pos.view(
            self.n_env, -1, 2
        )  # (n_env, n_agents, 2)

        # Pre-compute tensors for the CLF and CBF constraints,
        # Lyapunov Function from: https://arxiv.org/pdf/1903.03692.pdf

        # Laypunov function
        V_value = (
            goal_pos[:, X] ** 2
            + 0.5 * goal_pos[:, X] * agent_vel[:, X]
            + agent_vel[:, X] ** 2
            + goal_pos[:, Y] ** 2
            + 0.5 * goal_pos[:, Y] * agent_vel[:, Y]
            + agent_vel[:, Y] ** 2
        )

        LfV_val = (2 * goal_pos[:, X] + agent_vel[:, X]) * (agent_vel[:, X]) + (
            2 * goal_pos[:, Y] + agent_vel[:, Y]
        ) * (agent_vel[:, Y])
        LgV_vals = torch.stack(
            [
                0.5 * goal_pos[:, X] + 2 * agent_vel[:, X],
                0.5 * goal_pos[:, Y] + 2 * agent_vel[:, Y],
            ],
            dim=1,
        )
        # Define Quadratic Program (QP) based controller
        u = cp.Variable(2)
        V_param = cp.Parameter(1)  # Lyapunov Function: V(x): x -> R, dim: (1,1)
        lfV_param = cp.Parameter(1)
        lgV_params = cp.Parameter(
            2
        )  # Lie derivative of Lyapunov Function, dim: (1, action_dim)
        clf_slack = cp.Variable(1)  # CLF constraint slack variable, dim: (1,1)

        constraints = []

        # QP Cost F = u^T @ u + clf_slack**2
        qp_objective = cp.Minimize(cp.sum_squares(u) + self.clf_slack * clf_slack**2)

        # control bounds between u_range
        constraints += [u <= u_range]
        constraints += [u >= -u_range]
        # CLF constraint
        constraints += [
            lfV_param + lgV_params @ u + self.clf_epsilon * V_param + clf_slack <= 0
        ]

        QP_problem = cp.Problem(qp_objective, constraints)

        # Initialize CVXPY layers
        QP_controller = CvxpyLayer(
            QP_problem,
            parameters=[V_param, lfV_param, lgV_params],
            variables=[u],
        )

        # Solve QP
        CVXpylayer_parameters = [
            V_value.unsqueeze(1),
            LfV_val.unsqueeze(1),
            LgV_vals,
        ]
        action = QP_controller(*CVXpylayer_parameters, solver_args={"max_iters": 500})[
            0
        ]

        return action


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=False,
        enforce_bounds=True,
    )
