# IMPLEMENTATION OF THE SOCIAL FORCE MODEL FOR HUMAN DYNAMICS
# This file contains the implementation of the
# Social Force Model (SFM) for human dynamics.

# The Social Force Model (SFM) is a physics-based model
# that describes the interactions between pedestrians in a crowd.
# The model is based on the idea that pedestrians are
# subject to forces that influence their movement.
# These forces include a desired force that
#  represents the pedestrian's goal,
# an obstacle force that repels the pedestrian from obstacles,
# a social force that attracts the pedestrian to other pedestrians,
#  and a group force that represents the influence of
# a group of pedestrians on the individual pedestrian.


import torch
from typing import Optional
from vmas.simulator.human_dynamics.policies.utils import (
    distance_agents_to_boxes,
    distance_agents_to_spheres,
    distance_agents_to_lines,
)


def compute_group_tensor(agents: torch.Tensor) -> torch.Tensor:
    """Compute the tensor representation of the groups in the simulation.

    Args:
        agents (torch.Tensor): agents in the simulation
        agents.shape = (envs, n_agents, n_features)
        (x, y, vx, vy, radius, goal_x, goal_y, group_id)

    Returns:
        group_tensor (torch.Tensor): Tensor representation of the groups
        group_tensor.shape = (envs, n_groups, n_agents + 3)
        tensor = [agent_1, agent_2, ..., agent_n, center_x, center_y, n_agents]
    """
    envs, n_agents, _ = agents.shape

    # the number of groups is the number of unique group ids different from -1
    n_groups = agents[:, :, -1].unique().ne(-1).sum()
    group_tensor = torch.zeros(envs, n_groups, n_agents + 3)

    # compute the number of agents in each group using the group id in a batched manner

    group_tensor[:, :, :-3] = agents[:, :, -1].unsqueeze(1) == torch.arange(
        n_groups
    ).unsqueeze(0).unsqueeze(
        -1
    )  # (envs, n_groups, n_agents)
    # compute the center of mass of each group
    group_tensor[:, :, -3] = (
        group_tensor[:, :, :-3] * agents[:, :, :2].unsqueeze(1)
    ).sum(
        dim=-2
    )  # (envs, n_groups, 2)
    group_tensor[:, :, -3] /= group_tensor[:, :, :-3].sum(dim=-1)  # (envs, n_groups, 2)
    # compute the number of agents in each group
    group_tensor[:, :, -1] = group_tensor[:, :, :-3].sum(dim=-1)  # (envs, n_groups)

    return group_tensor


class SocialForcePolicy(torch.nn.Module):
    """Simulator for the Social Force Model (SFM) for human dynamics.

    Every agent in the simulation is represented by an entry vector
    (x, y, vx, vy, radius, goal_x, goal_y).
      - x, y: position of the agent
      - vx, vy: velocity of the agent
      - radius: radius of the agent
      - goal_x, goal_y: goal of the agent
    """

    def __init__(
        self,
        time_step: float,
        neighbor_dist: float,
        max_neighbors: int,
        time_horizon: float,
        time_horizon_obst: float,
        radius: float,
        max_speed: float,
        device: str,
        safety_space: float,
        *args,
        force_factor_desired: float = 2.0,
        force_factor_obstacle: float = 10.0,
        force_sigma_obstacle: float = 0.2,
        force_factor_social: float = 2.1,
        force_factor_group_gaze: float = 3.0,
        force_factor_group_coherence: float = 2.0,
        force_factor_group_repulsion: float = 1.0,
        obstacle_max_dist: float = 2.0,
        agent_max_dist: float = 1.5,
        lambda_: float = 2.0,
        gamma: float = 0.35,
        n: float = 2.0,
        n_prime: float = 3.0,
        relaxation_time: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        """
        Initialize the Social Force Model simulator.

        Args:
            time_step (float): Time step for the simulation
            neighbor_dist (float): Distance at which agents are considered neighbors
            max_neighbors (int): Maximum number of neighbors for each agent
            time_horizon (float): Time horizon for the agent
            time_horizon_obst (float): Time horizon for obstacles
            radius (float): Radius of the agent
            max_speed (float): Maximum speed of the agent
            device (str): Device to run the simulation on
            safety_space (float): Safety space between agents

        Keyword Args:
          force_factor_desired: 2.0
          force_factor_obstacle: 10.0
          force_sigma_obstacle: 0.2
          force_factor_social: 2.1
          force_factor_group_gaze: 3.0
          force_factor_group_coherence: 2.0
          force_factor_group_repulsion: 1.0
          obstacle_max_dist: 2.0
          agent_max_dist: 2.0
          lambda_: 2.0
          gamma: 0.35
          n: 2.0
          n_prime: 3.0
          relaxation_time: 0.5
        """
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.radius = radius
        self.max_speed = max_speed
        self.device = device
        self.safety_space = safety_space

        self.force_factor_desired = force_factor_desired
        self.force_factor_obstacle = force_factor_obstacle
        self.force_sigma_obstacle = force_sigma_obstacle
        self.force_factor_social = force_factor_social
        self.force_factor_group_gaze = force_factor_group_gaze
        self.force_factor_group_coherence = force_factor_group_coherence
        self.force_factor_group_repulsion = force_factor_group_repulsion
        self.obstacle_max_dist = obstacle_max_dist
        self.agent_max_dist = agent_max_dist
        self.lambda_ = lambda_
        self.gamma = gamma
        self.n = n
        self.n_prime = n_prime
        self.relaxation_time = relaxation_time

    def forward(
        self,
        agents: torch.Tensor,
        box_obstacles: torch.Tensor,
        sphere_obstacles: torch.Tensor,
        lines: torch.Tensor,
    ):
        """Forward pass of the Social Force Model simulator.

        Args:
          agents (torch.Tensor): agents in the simulation
          agents.shape = (envs, n_agents, n_features)
          (x, y, vx, vy, radius, goal_x, goal_y, group_id)
          box_obstacles (torch.Tensor): tensor of box obstacles
            in the simulation (n_boxes, 4, 2)
          sphere_obstacles (torch.Tensor): tensor of sphere obstacles
            in the simulation (n_spheres, 3)
          lines (torch.Tensor): tensor of lines in the simulation (n_lines, 2, 2)

        Returns:
          agents (torch.Tensor): updated agents in the simulation
          agents.shape = (envs, n_agents, n_features)
          (x, y, vx, vy, radius, goal_x, goal_y, group_id)
        """
        forces = self.compute_forces(agents, box_obstacles, sphere_obstacles, lines)
        # agents = self.update_position(agents, forces, self.time_step)
        self.update_position(agents, forces, self.time_step)
        return agents

    def compute_forces(
        self,
        agents: torch.Tensor,
        box_obstacles: torch.Tensor,
        sphere_obstacles: torch.Tensor,
        lines: torch.Tensor,
    ):
        """Compute the forces for agents in the simulation.

        Args:
          agents (torch.Tensor): agents in the simulation
          agents.shape = (envs, n_agents, n_features)
          (x, y, vx, vy, radius, goal_x, goal_y, group_id)
          box_obstacles (torch.Tensor): tensor of box obstacles
            in the simulation (n_boxes, 4, 2)
          sphere_obstacles (torch.Tensor): tensor of sphere obstacles
            in the simulation (n_spheres, 3)
          lines (torch.Tensor): tensor of lines in the simulation (n_lines, 2, 2)

        Returns:
          forces (torch.Tensor): forces acting on the agents in the simulation
          forces.shape = (envs, n_agents, 2)
        """

        envs, n_agents, n_features = agents.shape
        forces = torch.zeros(envs, n_agents, 2)  # (envs, n_agents, 2)

        # TODO implement add group forces
        # groups_tensor = compute_group_tensor(agents)  # (envs, n_groups, n_agents + 3)

        # compute the group forces for each agent

        # group_forces = self.compute_group_force(
        #      agents, groups_tensor
        # )  # (envs, n_agents, 2)

        # compute the desired force for each agent
        desired_force = self.compute_desired_force(agents)

        # compute the obstacle force for each agent
        obstacle_force = self.compute_obstacle_force(
            agents, box_obstacles, sphere_obstacles, lines
        )

        # compute the social force for each agent
        social_force = self.compute_social_force(agents)  # (envs, n_agents, 2)

        # sum the desired, obstacle, and social forces to get the total force
        forces = (
            desired_force + obstacle_force + social_force
        )  # + group_forces  # (envs, n_agents, 2)
        return forces

    def compute_desired_force(self, agents: torch.Tensor):
        """Compute the desired force for agents in the simulation.
        Args:
          agents (torch.Tensor): agents in the simulation
          agents.shape = (envs, n_agents, n_features)
          (x, y, vx, vy, radius, goal_x, goal_y, group_id)

        Returns:
          desired_force (torch.Tensor): Desired force acting on the agent
          desired_force.shape = (envs, n_agents, 2)
        """
        diff = agents[:, :, 5:7] - agents[:, :, :2]

        desired_direction = diff / (
            torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        )  # direction of the relative position vector
        # use a conditional to check if the agent goals are
        # too close to the agent position
        # if the agent goals are too close to the agent position,
        # the agent should not move
        condition = torch.where(
            torch.norm(diff, dim=-1, keepdim=True) > agents[:, :, 4:5], 1.0, 0.0
        )  # (envs, n_agents, 1)

        desired_force = (
            self.force_factor_desired
            * condition
            * (desired_direction * self.max_speed - agents[:, :, 2:4])
            / self.relaxation_time
            + (1 - condition) * -agents[:, :, 2:4] / self.relaxation_time
        )  # (envs, n_agents, 2)

        return desired_force

    def compute_obstacle_force(
        self,
        agents: torch.Tensor,
        box_obstacles: Optional[torch.Tensor],
        sphere_obstacles: Optional[torch.Tensor],
        lines: Optional[torch.Tensor],
    ):
        """Compute the obstacle force for an agent in the simulation.
        Args:
          agents (torch.Tensor): agents in the simulation
          agents.shape = (envs, n_agents, n_features)
          (x, y, vx, vy, radius, goal_x, goal_y, group_id)
          box_obstacles (torch.Tensor): tensor of box obstacles
            in the simulation (envs, n_boxes, 4, 2)
          sphere_obstacles (torch.Tensor): tensor of sphere obstacles
            in the simulation (envs, n_spheres, 3)
          lines (torch.Tensor): tensor of lines in the simulation (envs, n_lines, 2, 2)

        Returns:
          obstacle_force (torch.Tensor): Obstacle force acting on the agent
          obstacle_force.shape = (envs, n_agents, 2)
        """
        envs, n_agents, n_features = agents.shape
        # extend the agent tensor to have the same shape as the obstacles
        box_obstacles_num = 0 if box_obstacles is None else box_obstacles.shape[1]
        sphere_obstacles_num = (
            0 if sphere_obstacles is None else sphere_obstacles.shape[1]
        )
        lines_num = 0 if lines is None else lines.shape[1]

        obstacles_dim = box_obstacles_num + sphere_obstacles_num + lines_num

        closest_points = torch.zeros(
            envs, n_agents, obstacles_dim, 2, device=self.device
        )  # (envs, n_agents, n_obstacles, 2)
        distances = torch.zeros(
            envs, n_agents, obstacles_dim, device=self.device
        )  # (envs, n_agents, n_obstacles)

        if box_obstacles is not None:
            # compute the distance between the agents and the box obstacles
            (
                closest_points[:, :, :box_obstacles_num],
                distances[:, :, :box_obstacles_num],
            ) = distance_agents_to_boxes(agents, box_obstacles)

        # compute the distance between the agents and the sphere obstacles
        if sphere_obstacles is not None:
            (
                closest_points[
                    :,
                    :,
                    box_obstacles_num : box_obstacles_num + sphere_obstacles_num,
                ],
                distances[
                    :,
                    :,
                    box_obstacles_num : box_obstacles_num + sphere_obstacles_num,
                ],
            ) = distance_agents_to_spheres(agents, sphere_obstacles)

        if lines is not None:
            # compute the distance between the agents and the lines
            (
                closest_points[
                    :,
                    :,
                    box_obstacles_num + sphere_obstacles_num :,
                ],
                distances[
                    :,
                    :,
                    box_obstacles_num + sphere_obstacles_num :,
                ],
            ) = distance_agents_to_lines(agents, lines)

        # closest_points.shape = (envs, n_agents, n_obstacles, 2)
        # distances.shape = (envs, n_agents, n_obstacles)

        # compute the obstacle force for each agent

        min_diff = agents[:, :, :2].unsqueeze(2) - closest_points[:, :, :, :2]
        # (envs, n_agents, n_obstacles, 2)

        diff_direction = min_diff / (
            torch.norm(min_diff, dim=-1, keepdim=True) + 1e-8
        )  # (envs, n_agents, n_obstacles, 2)

        obstacle_force = (
            self.force_factor_obstacle
            * torch.exp(-distances.unsqueeze(-1) / self.force_sigma_obstacle)
            * diff_direction
        )  # (envs, n_agents, n_obstacles, 2)

        # if the distance is greater than the obstacle max distance
        # the obstacle force is 0
        obstacle_force = torch.where(
            distances.unsqueeze(-1) > self.obstacle_max_dist,
            torch.zeros_like(obstacle_force),
            obstacle_force,
        )  # (envs, n_agents, n_obstacles, 2)

        # get the number of active obstacles
        active_obstacles = torch.where(
            distances < self.obstacle_max_dist, 1.0, 0.0
        ).sum(
            dim=-1
        )  # (envs, n_agents)

        # sum the obstacle forces from all obstacles to get
        # the total obstacle force for the agent weighted
        # by the number of active obstacles

        return obstacle_force.sum(dim=2) / (active_obstacles.unsqueeze(-1) + 1e-8)
        # (envs, n_agents, 2)

    def compute_social_force(self, agents: torch.Tensor) -> torch.Tensor:
        """Compute the social force for agents in the simulation.
        Args:
          index (int): Index of the agent in the list of agents
          agents (torch.Tensor): List of agents in the simulation
          agents.shape = (envs, n_agents, n_features)
          (x, y, vx, vy, radius, goal_x, goal_y, group_id)

        Returns:
          social_force (torch.Tensor): Social force acting on the agent
          social_force.shape = (envs, n_agents, 2)
        """
        envs, n_agents, _ = agents.shape  # (n_envs, n_agents, n_features)

        x_i = agents.unsqueeze(2)  # (n_envs, n_agents, 1, n_features)
        x_j = agents.unsqueeze(1)  # (n_envs, 1, n_agents, n_features)

        diff = (
            x_j[:, :, :, :2] - x_i[:, :, :, :2]
        )  # relative position (x, y) of agent j with respect to agent i
        diff_direction = diff / (
            torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        )  # direction of the relative position vector (n_envs, n_agents, n_agents, 2)
        vel_diff = (
            x_i[:, :, :, 2:4] - x_j[:, :, :, 2:4]
        )  # relative velocity (vx, vy) of agent j with respect to agent i
        interaction_vector = (
            self.lambda_ * vel_diff + diff_direction
        )  # interaction vector (n_envs, n_agents, n_agents, 2)
        interaction_length = torch.norm(
            interaction_vector, dim=-1, keepdim=True
        )  # interaction length (n_envs, n_agents, n_agents, 1)
        interaction_direction = interaction_vector / (
            interaction_length + 1e-8
        )  # interaction direction (n_envs, n_agents, n_agents, 2)

        theta = -torch.atan2(
            interaction_direction[:, :, :, 1], interaction_direction[:, :, :, 0]
        ) + torch.atan2(diff_direction[:, :, :, 1], diff_direction[:, :, :, 0])
        # angle between interaction direction and diff direction
        # (n_envs, n_agents, n_agents, 1)

        B = self.gamma * interaction_length  # B parameter
        force_velocity_amount = -torch.exp(
            -torch.norm(diff, dim=-1, keepdim=True) / (B + 1e-8)
            - torch.pow(self.n_prime * B * theta.unsqueeze(-1), 2)
        )  # force velocity amount f
        # (n_envs, n_agents, n_agents, 1)

        force_angle_amount = -torch.sign(theta.unsqueeze(-1)) * torch.exp(
            -torch.norm(diff, dim=-1, keepdim=True) / (B + 1e-8)
            - torch.pow(self.n * B * theta.unsqueeze(-1), 2)
        )  # force angle amount (n_envs, n_agents, n_agents, 1)
        force_velocity = (
            force_velocity_amount * interaction_direction
        )  # force velocity (n_envs, n_agents, n_agents, 2)
        normal_interaction_direction = torch.stack(
            [-interaction_direction[:, :, :, 1], interaction_direction[:, :, :, 0]],
            dim=-1,
        )  # normal interaction direction (n_envs, n_agents, n_agents, 2)
        force_angle = force_angle_amount * normal_interaction_direction

        social_force = self.force_factor_social * (
            force_velocity + force_angle
        )  # social forces (n_envs, n_agents, n_agents, 2)

        # if the agent i is the same as agent j, if i == j, then the force is 0
        idx = torch.arange(n_agents)
        social_force[:, idx, idx] = 0.0
        # obtain the social force for the single agent as
        # the sum of the social forces from all other agents

        # set to 0 the social force for agents that are not neighbors
        social_force = torch.where(
            torch.norm(diff, dim=-1, keepdim=True) > self.agent_max_dist,
            torch.zeros_like(social_force),
            social_force,
        )

        return social_force.sum(dim=2)  # (n_envs, n_agents, 2)

    def compute_group_force(
        self,
        agents: torch.Tensor,
        groups: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the group force for an agent in the simulation.

        Args:
          agents (torch.Tensor): List of agents in the simulation
          (envs, n_agents, n_features)
          groups (torch.Tensor): Groups of agents in the simulation
          (envs, n_groups, n_agents + 3)
          groups.shape = (envs, n_groups, n_agents + 3)
          (agent_1, agent_2, ..., agent_n, center_x, center_y, n_agents) for each group

        Returns:
          group_force (torch.Tensor): Group force acting on the agent
          group_force.shape = (envs, n_agents, 2)
        """

        envs, n_agents, _ = agents.shape
        group_force = torch.zeros(envs, n_agents, 2)

        # check if the agents are in a group and if the group has more than one agent
        group_indices = torch.where(groups[:, :, -1] > 1)  # (envs, n_groups, 1)

        # Gaze force
        #   utils::Vector2d com = group.center;
        #   com = (1 / (double)(group.agents.size() - 1)) *
        #         (group.agents.size() * com - agent.position);

        # compute the center of mass of the group
        # 1 / (n_agents_in_group - 1) *
        # (n_agents_in_group * group_center - agent_position)
        com = groups[group_indices][:, :, -3:-1]  # (envs, n_groups, 2)
        com = (
            1
            / (groups[group_indices][:, :, -1] - 1)
            * (groups[group_indices][:, :, -1] * com - agents[:, :, :2].unsqueeze(1))
        )  # (envs, n_groups, 2)

        # compute the relative center of mass
        relative_com = com - agents[:, :, :2].unsqueeze(
            1
        )  # (envs, n_agents, n_groups, 2)

        # compute the vision angle
        vision_angle = torch.pi / 2
        # compute the element product
        # TODO: use the desired direction instead of the velocity
        element_product = torch.sum(
            agents[:, :, 5:7].unsqueeze(1) * relative_com, dim=-1
        )  # (envs, n_agents, n_groups)
        # compute the angle between the desired direction
        # and the relative center of mass
        com_angle = torch.acos(
            element_product
            / (torch.norm(agents[:, :, 5:7], dim=-1) * torch.norm(relative_com, dim=-1))
        )  # (envs, n_agents, n_groups)

        # check if the angle is greater than the vision angle
        condition = torch.where(
            com_angle > vision_angle, 1.0, 0.0
        )  # (envs, n_agents, n_groups)

        # compute the necessary rotation
        # TODO: use the desired direction instead of the velocity
        desired_direction_distance = (
            element_product / torch.norm(agents[:, :, 5:7], dim=-1) ** 2
        )  # (envs, n_agents, n_groups)
        # TODO: use the desired direction instead of the velocity
        group_gaze_force = (
            desired_direction_distance * agents[:, :, 5:7]
        )  # (envs, n_agents, n_groups)

        # multiply the group gaze force by the group gaze force factor
        group_gaze_force *= self.force_factor_group_gaze  # (envs, n_agents, n_groups)

        # Coherence force

        com = groups[group_indices][:, :, -3:-1]  # (envs, n_groups, 2)
        relative_com = com - agents[:, :, :2].unsqueeze(
            1
        )  # (envs, n_agents, n_groups, 2)
        distance = torch.norm(relative_com, dim=-1)  # (envs, n_agents, n_groups)
        max_distance = (groups[group_indices][:, :, -1] - 1) / 2  # (envs, n_groups)
        softened_factor = (
            self.force_factor_group_coherence
            * (torch.tanh(distance - max_distance) + 1)
            / 2
        )
        group_coherence_force = relative_com * softened_factor.unsqueeze(
            -1
        )  # (envs, n_agents, n_groups, 2)

        # Repulsion Force

        # get the agent indices in the group
        agent_indices = groups[group_indices][:, :, :-3]  # (envs, n_groups, n_agents)
        agent_indices = torch.where(agent_indices == 1)  # (envs, n_groups, n_agents)

        # get the agent positions
        agent_positions = agents[:, :, :2].unsqueeze(1)  # (envs, 1, n_agents, 2)

        # get the agent positions in the group
        group_agent_positions = agent_positions[
            agent_indices
        ]  # (envs, n_groups, n_agents, 2)

        # compute the difference between
        # the agent positions and the group agent positions
        diff = group_agent_positions - agents[:, :, :2].unsqueeze(
            1
        )  # (envs, n_agents, n_groups, 2)

        # compute the norm of the difference
        norm_diff = torch.norm(diff, dim=-1)  # (envs, n_agents, n_groups)

        # check if the norm of the difference is less than the sum of the radii
        condition = torch.where(
            norm_diff
            < agents[:, :, 4:5].unsqueeze(1) + agents[agent_indices][:, :, 4:5],
            1.0,
            0.0,
        )

        # compute the repulsion force
        group_repulsion_force = (
            condition.unsqueeze(-1) * diff
        )  # (envs, n_agents, n_groups, 2)

        # multiply the group repulsion force by the group repulsion force factor
        group_repulsion_force *= (
            self.force_factor_group_repulsion
        )  # (envs, n_agents, n_groups, 2)

        # sum the group forces to get the total group force
        group_force = (
            group_gaze_force + group_coherence_force + group_repulsion_force
        )  # (envs, n_agents, n_groups, 2)
        # the group force is applied to the agent of the group
        group_force = group_force * groups[group_indices][:, :, :-3].unsqueeze(
            -1
        )  # (envs, n_agents, n_groups, 2)
        return group_force.sum(dim=-2)  # (envs, n_agents, 2)

    def move(self, agent: torch.Tensor, dt: float):
        pass

    def update_position(self, agents: torch.Tensor, forces: torch.Tensor, dt: float):
        """Update the position of an agent in the simulation.

        Args:
          agents (torch.Tensor): agents in the simulation
          agents.shape = (envs, n_agents, n_features)
          (x, y, vx, vy, radius, goal_x, goal_y, group_id)
          forces (torch.Tensor): forces acting on the agents in the simulation
          forces.shape = (envs, n_agents, 2)
          dt (float): time step for the simulation

        Returns:
          agents (torch.Tensor): updated agents in the simulation
          agents.shape = (envs, n_agents, n_features)
          (x, y, vx, vy, radius, goal_x, goal_y, group_id)
        """

        envs, n_agents, n_features = agents.shape

        # init_pos = agents[:, :, :2].clone()  # (envs, n_agents, 2)
        # movement = torch.zeros(envs, n_agents, 2)

        # update the velocity of the agent
        agents[:, :, 2:4] += forces * dt  # (envs, n_agents, 2)
        agents[:, :, 2:4] = torch.where(
            torch.norm(agents[:, :, 2:4], dim=-1, keepdim=True) > self.max_speed,
            agents[:, :, 2:4]
            / (torch.norm(agents[:, :, 2:4], dim=-1, keepdim=True) + 1e-8)
            * self.max_speed,
            agents[:, :, 2:4],
        )

        agents[:, :, :2] += agents[:, :, 2:4] * dt  # (envs, n_agents, 2)
        # movement = agents[:, :, :2] - init_pos  # (envs, n_agents, 2)
        # TODO implement the cyclic goals logic for the agents
        # return agents
