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

# from vmas.simulator.human_dynamics.human_simulation import HumanSimulation
from vmas.simulator.human_dynamics.human_simulation_sfm import HumanSimulation

# from vmas.simulator.human_dynamics.roadmap import find_vertices

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


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

        self.n_passages = kwargs.pop("n_passages", 5)
        self.middle_angle_180 = kwargs.pop("middle_angle_180", False)

        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -1)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2
        self.min_collision_distance = 0.005

        if self.enforce_bounds:
            self.x_semidim = self.world_spawning_x
            self.y_semidim = self.world_spawning_y
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
        # self.create_passage_map(world)
        self.create_boundaries(world)

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

    def spawn_boundaries(self, env_index):
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
                    ).repeat(self.world.batch_dim, 1),
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
                    ).repeat(self.world.batch_dim, 1),
                    batch_index=env_index,
                )

    def create_passage_map(self, world: World):
        """
        Create a passage map with n_passages passages

        Args:
            world: World object
        """

        # Add landmarks
        self.passages = []
        self.collide_passages = []
        self.non_collide_passages = []

        def is_passage(i):
            return i < self.n_passages

        for i in range(self.n_boxes):
            passage = Landmark(
                name=f"obstacle {i}",
                collide=not is_passage(i),
                movable=False,
                shape=Box(length=self.passage_length, width=self.passage_width),
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            if not passage.collide:
                self.non_collide_passages.append(passage)
            else:
                self.collide_passages.append(passage)
            self.passages.append(passage)
            world.add_landmark(passage)

    def set_n_passages(self, val):
        if val == 4:
            self.middle_angle_180 = True
        elif val == 3:
            self.middle_angle_180 = False
        else:
            raise AssertionError()
        self.n_passages = val
        del self.world._landmarks[-self.n_boxes :]
        self.create_passage_map(self.world)
        self.reset_world_at()

    def spawn_passage_map(self, env_index):
        if self.fixed_passage:
            big_passage_start_index = torch.full(
                (self.world.batch_dim,) if env_index is None else (1,),
                5,
                device=self.world.device,
            )
            small_left_or_right = torch.full(
                (self.world.batch_dim,) if env_index is None else (1,),
                1,
                device=self.world.device,
            )
        else:
            big_passage_start_index = torch.randint(
                0,
                self.n_boxes - 1,
                (self.world.batch_dim,) if env_index is None else (1,),
                device=self.world.device,
            )
            small_left_or_right = torch.randint(
                0,
                2,
                (self.world.batch_dim,) if env_index is None else (1,),
                device=self.world.device,
            )

        small_left_or_right[
            big_passage_start_index > self.n_boxes - 1 - (self.n_passages + 1)
        ] = 0
        small_left_or_right[big_passage_start_index < self.n_passages] = 1
        small_left_or_right[small_left_or_right == 0] -= 3
        small_left_or_right[small_left_or_right == 1] += 3

        def is_passage(i):
            is_pass = big_passage_start_index == i
            is_pass += big_passage_start_index == i - 1
            is_pass += big_passage_start_index + small_left_or_right == i
            if self.n_passages == 4:
                is_pass += (
                    big_passage_start_index + small_left_or_right
                    == i - torch.sign(small_left_or_right)
                )
            return is_pass

        def get_pos(i):
            """Get the position of the passage at index i"""
            pos = torch.tensor(
                [
                    -1 - self.agent_radius + self.passage_length / 2,
                    0.0,
                ],
                dtype=torch.float32,
                device=self.world.device,
            ).repeat(
                i.shape[0], 1
            )  # (batch_dim, 2)
            pos[:, X] += self.passage_length * i  # (batch_dim, 2)
            return pos

        for index, i in enumerate(
            [
                big_passage_start_index,
                big_passage_start_index + 1,
                big_passage_start_index + small_left_or_right,
            ]
            + (
                [
                    big_passage_start_index
                    + small_left_or_right
                    + torch.sign(small_left_or_right)
                ]
                if self.n_passages == 4
                else []
            )
        ):
            self.non_collide_passages[index].is_rendering[:] = False
            self.non_collide_passages[index].set_pos(get_pos(i), batch_index=env_index)

        big_passage_pos = (
            get_pos(big_passage_start_index) + get_pos(big_passage_start_index + 1)
        ) / 2
        small_passage_pos = get_pos(big_passage_start_index + small_left_or_right)
        pass_center = (big_passage_pos + small_passage_pos) / 2

        if env_index is None:
            self.small_left_or_right = small_left_or_right
            self.pass_center = pass_center
            self.big_passage_pos = big_passage_pos
            self.small_passage_pos = small_passage_pos
            self.middle_angle[small_left_or_right > 0] = torch.pi
            self.middle_angle[small_left_or_right < 0] = 0
        else:
            self.pass_center[env_index] = pass_center
            self.small_left_or_right[env_index] = small_left_or_right
            self.big_passage_pos[env_index] = big_passage_pos
            self.small_passage_pos[env_index] = small_passage_pos
            self.middle_angle[env_index] = (
                0 if small_left_or_right.item() < 0 else torch.pi
            )

        i = torch.zeros(
            (self.world.batch_dim,) if env_index is None else (1,),
            dtype=torch.int,
            device=self.world.device,
        )  # (batch_dim, 1)
        for passage in self.collide_passages:
            is_pass = is_passage(i)
            while is_pass.any():
                i[is_pass] += 1
                is_pass = is_passage(i)
            passage.set_pos(get_pos(i), batch_index=env_index)
            i += 1

    def reset_world_at(self, env_index: int = None):
        new_bownd_x = self.world_spawning_x - self.min_distance_between_entities
        new_bownd_y = self.world_spawning_y - self.min_distance_between_entities
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (
                -new_bownd_x,
                new_bownd_x,
            ),
            (
                -new_bownd_y,
                new_bownd_y,
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
                x_bounds=(-new_bownd_x, new_bownd_x),
                y_bounds=(-new_bownd_y, new_bownd_y),
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
        self.spawn_boundaries(env_index)
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
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
                < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)

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
