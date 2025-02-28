#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Callable, Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Box, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar, AgentsPoses
from vmas.simulator.utils import Color, ScenarioUtils, X, Y
from vmas.scenarios.social_scenarios.env_utils import env_utils, human_utils

from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic import Holonomic

from vmas.simulator.human_dynamics.human_simulation_sfm import HumanSimulation

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

from typing import Optional


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.n_agents = kwargs.pop("n_agents", 1)

        self.n_scripted_agents = kwargs.pop("n_scripted_agents", 2)  # scripted agents
        self.collisions = kwargs.pop("collisions", True)

        self.world_spawning_x = kwargs.pop(
            "world_spawning_x", 1.0
        )  # X-coordinate limit for entities spawning
        self.world_spawning_y = kwargs.pop(
            "world_spawning_y", 3.0
        )  # Y-coordinate limit for entities spawning
        self.enforce_bounds = kwargs.pop(
            "enforce_bounds", True
        )  # If False, the world is unlimited; else, constrained by world_spawning_x and world_spawning_y.
        self.observe_all_goals = kwargs.pop("observe_all_goals", False)

        self.lidar_range = kwargs.pop("lidar_range", 3.0)
        self.agent_radius = kwargs.pop("agent_radius", 0.25)
        self.comms_range = kwargs.pop("comms_range", 0)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 36)

        self.n_box_obstacles = kwargs.pop("n_box_obstacles", 2)
        self.n_sphere_obstacles = kwargs.pop("n_sphere_obstacles", 1)
        # self.middle_angle_180 = kwargs.pop("middle_angle_180", False)

        self.shared_rew = kwargs.pop("shared_rew", True)
        self.pos_shaping_factor = kwargs.pop("pos_shaping_factor", 1)
        self.final_reward = kwargs.pop("final_reward", 1000)

        self.agent_collision_penalty = kwargs.pop("agent_collision_penalty", -400)
        self.scenario_type = kwargs.pop("human_scenario_type", None)
        self.dt = kwargs.pop("dt", 0.1)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.min_distance_between_entities = self.agent_radius * 2
        self.min_collision_distance = 0.005

        if self.enforce_bounds:
            self.x_semidim_mean = (
                self.world_spawning_x + self.min_distance_between_entities
            )
            self.y_semidim_mean = (
                self.world_spawning_y + self.min_distance_between_entities
            )
        else:
            self.x_semidim_mean = None
            self.y_semidim_mean = None
        # Make world

        world = World(batch_dim, device, substeps=2, dt=self.dt)

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
                max_speed=0.6,
                u_range=(0.6, 1.5),
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
                            neighbors=min(
                                3, self.n_agents + self.n_scripted_agents - 1
                            ),
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

        # Add a state buffer for the policy agents
        self.agents_state_buffer = env_utils.AgentsStateBuffer(2, world)

        self.Rd = None
        self.Rh = None
        self.Rv = None
        self.Ro = None
        self.Rp = None
        self.Rs = None

        # Add obstacles
        self.boundaries = env_utils.create_boundaries(
            world, self.x_semidim_mean, self.y_semidim_mean
        )
        self.box_obstacles, self.sphere_obstacles = env_utils.create_random_obstacles(
            world,
            self.n_box_obstacles,
            self.n_sphere_obstacles,
            self.agent_radius,
        )
        return world

    def reset_world_at(self, env_index: int = None):
        with torch.profiler.record_function("reset_world_at"):
            spawning_x = self.world_spawning_x
            spawning_y = self.world_spawning_y
            x_semidim = self.x_semidim_mean
            y_semidim = self.y_semidim_mean

            if self.scenario_type is not None:
                scenario_type = self.scenario_type
            else:
                scenario_type = human_utils.Scenario.sample_scenario()

            # TODO - make boundaries variable
            # use the world_spawning_x as a mean

            if scenario_type == human_utils.Scenario.RANDOM:
                spawning_x = spawning_y = max(spawning_x, spawning_y)
                x_semidim = y_semidim = max(x_semidim, y_semidim)

            with torch.profiler.record_function("generate_scenario"):
                occupied_positions = human_utils.generate_scenario(
                    self.world.policy_agents,
                    self.world.scripted_agents,
                    self.world,
                    env_index,
                    self.min_distance_between_entities,
                    (
                        -spawning_x,
                        spawning_x,
                    ),
                    (
                        -spawning_y,
                        spawning_y,
                    ),
                    current_scenario=scenario_type,
                )

            with torch.profiler.record_function("spawn_boundaries"):

                if self.enforce_bounds:
                    env_utils.spawn_boundaries(
                        self.boundaries,
                        x_semidim,
                        y_semidim,
                        self.world.device,
                        env_index,
                    )

            with torch.profiler.record_function("spawn_obstacles"):
                env_utils.spawn_random_obstacles(
                    env_index,
                    occupied_positions,
                    self.world,
                    self.box_obstacles,
                    self.sphere_obstacles,
                    self.min_distance_between_entities,
                    x_semidim,
                    y_semidim,
                    # self.world_spawning_x,
                    # self.world_spawning_y,
                )
            with torch.profiler.record_function("human_simulation_reset"):
                self.human_simulation.reset(self.world, env_index)

            with torch.profiler.record_function("agents_state_buffer_reset"):
                self.agents_state_buffer.reset(env_index=env_index)

            with torch.profiler.record_function("reset_collision_flags"):
                if env_index is None:
                    self.is_collision_with_agents = torch.zeros(
                        self.world.batch_dim, device=self.world.device, dtype=torch.bool
                    )
                    self.is_collision_with_obstacles = torch.zeros(
                        self.world.batch_dim, device=self.world.device, dtype=torch.bool
                    )
                    self.goal_reached = torch.zeros(
                        self.world.batch_dim, device=self.world.device, dtype=torch.bool
                    )
                else:
                    self.is_collision_with_agents[env_index] = False
                    self.is_collision_with_obstacles[env_index] = False
                    self.goal_reached[env_index] = False

    def reward(self, agent: Agent):
        """
        Issue rewards for the given agent in all envs.
            Positive Rewards:
                Moving forward (become negative if the projection of the moving direction to its reference path is negative)
                Moving forward with high speed (become negative if the projection of the moving direction to its reference path is negative)
                Reaching goal (optional)

            Negative Rewards (penalties):
                Too close to other agents (proxemics)
                Social work
                Changing steering too quick
                Colliding with other agents

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            A tensor with shape [batch_dim].
        """
        # Initialize
        with torch.profiler.record_function("reward"):
            rew = torch.zeros(self.world.batch_dim, device=self.world.device)

            # Get the index of the current agent
            agent_index = self.human_simulation.find_agent_index(agent)
            cd = 1.0

            # Get the distance to the goal
            distance_to_goal = torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos, dim=-1
            )
            # [reward] forward movement
            latest_state = self.agents_state_buffer.get_agent_last_state(agent_index)
            if latest_state is not None:
                previous_distance_to_goal = torch.linalg.vector_norm(
                    latest_state[:, :2] - latest_state[:, 5:7], dim=-1
                )
                forward_movement = previous_distance_to_goal - distance_to_goal
                self.Rd = forward_movement * 35.0
                self.Rd.clamp(min=-3.0, max=3.0)

            # [reward] Heading towards the goal

            ch = 0.6
            # check if the agent is moving towards the goal
            # the direction of the agent is the nomalized velocity
            # the direction to the goal is the normalized vector from the agent to the goal

            desired_direction_angle = torch.atan2(
                (agent.goal.state.pos - agent.state.pos)[:, 1],
                (agent.goal.state.pos - agent.state.pos)[:, 0],
            )
            # if agent.state.vel is close to zero, the agent is not moving

            agent_yaw = torch.where(
                (torch.linalg.vector_norm(agent.state.vel, dim=-1) < 10e-5).unsqueeze(
                    -1
                ),
                torch.atan2(torch.sin(agent.state.rot), torch.cos(agent.state.rot)),
                torch.atan2(agent.state.vel[:, 1:2], agent.state.vel[:, 0:1]),
            ).squeeze(-1)

            angle = desired_direction_angle - agent_yaw
            self.Rh = 1.0 - 2 * torch.sqrt(torch.abs(angle / torch.pi))

            # [penalty] linear velocity
            cv = 0.8
            # check if the agent is moving too slow

            self.Rv = (
                torch.linalg.vector_norm(agent.state.vel, dim=-1) - agent.max_speed
            ) / agent.max_speed

            # [penalty] Obstacle avoidance

            co = 2.0
            # min distance to the obstacles
            # TODO get it from the human simulation
            # use the lidar sensor to get the distance to the obstacles
            # get the distance to the closest obstacle
            self.Ro = (agent.sensors[0].measure() - agent.sensors[0]._max_range).min(
                dim=-1
            )[0] / agent.sensors[0]._max_range

            # [penalty] social disturbance
            cp = 2.0
            self.Rp = -torch.clamp(
                1 / (agent.sensors[1].measure()[:, 0]).min(dim=-1)[0], min=0.0, max=2.5
            )
            cs = 2.5
            self.Rs = self.human_simulation.social_work[:, agent_index]
            self.Rs = -torch.clamp(self.Rs, min=0.0, max=2.5)

            # rew = Rd * cd + Rh * ch + Rv * cv + Ro * co + Rp * cp + Rs * cs

            if self.scenario_type == human_utils.Scenario.EASY:
                rew = self.Rd * cd + self.Rh * ch
            else:
                rew = (
                    self.Rd * cd
                    + self.Rh * ch
                    + self.Rv * cv
                    + self.Ro * co
                    + self.Rp * cp
                    + self.Rs * cs
                )
            # [penalty] time penalty
            rew = rew - 0.5

            # [reward] goal reached
            # check if the agent is on the goal
            self.goal_reached = (
                distance_to_goal - agent.shape.radius
            ) < agent.goal.shape.radius
            rew = rew + self.final_reward * self.goal_reached

            # [penalty] agent-agent collision

            is_collision_with_agents = (
                self.human_simulation.agents_distances[:, agent_index, :]
                < self.min_collision_distance
            )
            is_collision_with_agents[:, agent_index] = False

            self.is_collision_with_agents = is_collision_with_agents.any(dim=-1)

            is_collision_with_obstacles = (
                self.human_simulation.obstacle_distances[:, agent_index, :]
                < self.min_collision_distance
            )
            self.is_collision_with_obstacles = is_collision_with_obstacles.any(dim=-1)

            rew = rew - 1500 * (
                self.is_collision_with_agents | self.is_collision_with_obstacles
            )

            return rew

    def observation(self, agent: Agent):
        with torch.profiler.record_function("observation"):
            distance_to_goal = torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos, dim=-1
            )
            angle_to_goal = torch.atan2(
                agent.goal.state.pos[:, 1] - agent.state.pos[:, 1],
                agent.goal.state.pos[:, 0] - agent.state.pos[:, 0],
            )
            agent_vel = agent.state.vel.norm(dim=-1)

            return torch.cat(
                [
                    distance_to_goal.unsqueeze(-1),
                    angle_to_goal.unsqueeze(-1),
                    agent_vel.unsqueeze(-1),
                    agent.state.ang_vel,
                ]
                + ([agent.sensors[0].measure()] if self.collisions else [])
                + ([agent.sensors[1].measure()] if len(agent.sensors) > 1 else []),
                dim=-1,
            )

    def mpc_state(self, agent: Agent):
        return torch.cat(
            [
                agent.state.pos,
                agent.state.rot,
                # agent.state.vel.norm(dim=-1).unsqueeze(-1),
            ],
            dim=-1,
        )

    def pre_step(self):
        with torch.profiler.record_function("prestep"):
            with torch.profiler.record_function("simulate_policy"):
                self.human_simulation.simulate_policy(self.world)
            self.agents_state_buffer.add_state(self.human_simulation.old_agent_tensor)
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
        is_done = (
            self.goal_reached  # check if agent is on goal
            | self.is_collision_with_agents
            | self.is_collision_with_obstacles
        )
        return is_done

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """info to be returned to the agent"""
        # self.Rd, self.Rh, self.Rv, self.Ro, self.Rp, self.Rs
        if self.Rd is None:
            return {}
        return {
            "episode": {
                "reward_distance_mean": self.Rd.mean(),
                "reward_heading_mean": self.Rh.mean(),
                "reward_velocity_mean": self.Rv.mean(),
                "reward_obstacle_mean": self.Ro.mean(),
                "reward_proxemics_mean": self.Rp.mean(),
                "reward_social_work_mean": self.Rs.mean(),
                "reward_distance_max": self.Rd.max(),
                "reward_heading_max": self.Rh.max(),
                "reward_velocity_max": self.Rv.max(),
                "reward_obstacle_max": self.Ro.max(),
                "reward_proxemics_max": self.Rp.max(),
                "reward_social_work_max": self.Rs.max(),
                "reward_distance_min": self.Rd.min(),
                "reward_heading_min": self.Rh.min(),
                "reward_velocity_min": self.Rv.min(),
                "reward_obstacle_min": self.Ro.min(),
                "reward_proxemics_min": self.Rp.min(),
                "reward_social_work_min": self.Rs.min(),
                "reward_distance_std": self.Rd.std(),
                "reward_heading_std": self.Rh.std(),
                "reward_velocity_std": self.Rv.std(),
                "reward_obstacle_std": self.Ro.std(),
                "reward_proxemics_std": self.Rp.std(),
                "reward_social_work_std": self.Rs.std(),
            }
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


if __name__ == "__main__":
    # seed = 0

    render_interactively(
        __file__,
        control_two_agents=False,
        enforce_bounds=True,
    )
