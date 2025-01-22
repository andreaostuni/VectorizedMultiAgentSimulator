import torch
from typing import Tuple, List
from vmas.simulator.core import Box, Sphere, Line, Entity, EntityState
from vmas.simulator.core import Agent


def agent_to_tensor(agent: Agent) -> torch.Tensor:
    """
    Convert an Agent object to a tensor

    Args:
        agent (Agent): The agent to convert to a tensor
        it is already vectorized
        device (torch.device): The device to use

    Returns:
        torch.Tensor: The tensor representation of the agent
        tensor.shape = (n_envs, n_features)
        (x, y, vx, vy, radius, goal_x, goal_y, group_id)
    """
    # TODO implement group_id
    return torch.cat(
        [
            agent.state.pos,  # x, y
            # if the vel is nan, set it to 0
            torch.nan_to_num(agent.state.vel, nan=0.0),  # vx, vy
            torch.tensor([agent.shape.radius], device=agent.state.pos.device).repeat(
                agent.state.pos.shape[0], 1
            ),  # radius
            agent.goal.state.pos,  # goal_x, goal_y
            torch.tensor([-1], device=agent.state.pos.device).repeat(
                agent.state.pos.shape[0], 1
            ),  # group_id
        ],
        dim=-1,
    )


def agents_to_tensor(agents: List[Agent]) -> torch.Tensor:
    """
    Convert a list of Agent objects to a tensor

    Args:
        agents (List[Agent]): The agents to convert to a tensor
        the agents are already vectorized
        device (torch.device): The device to use

    Returns:
        torch.Tensor: The tensor representation of the agents
        tensor.shape = (n_envs, n_agents, n_features)
        (x, y, vx, vy, radius, goal_x, goal_y, group_id)
    """
    # TODO implement group_id
    # stack the agents in the first dimension
    return (
        torch.stack([agent_to_tensor(agent) for agent in agents])
        .permute(1, 0, 2)
        .to(agents[0].state.pos.device)
    )  # (n_envs, n_agents, n_features)


def distance_agents_to_lines(
    agents: torch.Tensor, lines: torch.Tensor, env_index: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the distance between an agent and a line

    Args:
        agents (torch.Tensor): agents in the simulation
          agents.shape = (envs, n_agents, n_features)
          (x, y, vx, vy, radius, goal_x, goal_y, group_id)

          The agent's position (x, y, vx, vy, radius)
        lines (torch.Tensor): The lines are represented by their vertices
          lines.shape = (envs, n_lines, 2, 2)
        env_index (int, optional): The index of the environment. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The closest point on the
        line and the distance
        closest_point.shape = (envs, n_agents, n_lines, 2)
        distance.shape = (envs, n_agents, n_lines)
    """

    # Line starts and ends
    lines_start = lines[:, :, 0, :]  # Shape (n_envs, n_lines, dim)
    lines_end = lines[:, :, 1, :]  # Shape (n_envs, n_lines, dim)

    # Compute line vectors
    line_vectors = lines_end - lines_start  # Shape (n_envs, n_lines, dim)

    # Expand tensors for batched computation

    # take only the x and y coordinates of the agents
    points_exp = agents[:, :, :2].unsqueeze(2)  # Shape (n_envs, n_points, 1, 2)
    lines_start_exp = lines_start.unsqueeze(1)  # Shape (n_envs, 1, n_lines, dim)
    line_vectors_exp = line_vectors.unsqueeze(1)  # Shape (n_envs, 1, n_lines, dim)

    # Compute vectors from line starts to points
    w = points_exp - lines_start_exp  # Shape (n_envs, n_points, n_lines, dim)

    # Compute the projection scalar t for each point-line pair
    line_lengths_squared = (
        torch.linalg.norm(line_vectors, dim=-1, keepdim=True) ** 2
    )  # Shape (n_envs, n_lines, 1)
    t = torch.sum(
        w * line_vectors_exp, dim=-1, keepdim=True
    ) / line_lengths_squared.unsqueeze(
        1
    )  # Shape (n_envs, n_points, n_lines, 1)

    # Clamp t to the range [0, 1]
    t_clamped = torch.clamp(t, 0, 1)  # Shape (n_envs, n_points, n_lines, 1)

    # Compute the closest points
    closest_points = (
        lines_start_exp + t_clamped * line_vectors_exp
    )  # Shape (n_envs, n_points, n_lines, dim)

    # Compute distances to the closest points
    distances = torch.norm(
        points_exp - closest_points, dim=-1
    )  # Shape (n_envs, n_points, n_lines)

    distances = distances - agents[:, :, 4].unsqueeze(2)
    if env_index is not None:
        return (
            closest_points[env_index, :, :, :],
            distances[env_index, :, :],
        )  # Shape (n_agents, n_lines, 2), (n_agents, n_lines)
    # Return the closest points and distances
    return (
        closest_points,
        distances,
    )  # Shape (n_envs, n_agents, n_lines, 2), (n_envs, n_agents, n_lines)


def distance_agents_to_spheres(
    agents: torch.Tensor, spheres: torch.Tensor, env_index: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the distance between a tensor of agents and a tensor of spheres

    Args:
        agents (torch.Tensor): The agents' positions
            agents.shape = (envs, n_agents, n_features)
            (x, y, vx, vy, radius, goal_x, goal_y, group_id)
        spheres (torch.Tensor): The spheres' positions and radii
            spheres.shape = (envs, n_spheres, 3)
            (x, y, radius)
        env_index (int, optional): The index of the environment. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The closest point on the
        sphere and the distance
        closest_point.shape = (envs, n_agents, n_spheres, 2)
        distance.shape = (envs, n_agents, n_spheres)
    """

    # Calculate the distance between the agent and the sphere
    distance = (
        torch.norm(
            agents[:, :, :2].unsqueeze(2) - spheres[:, :2].unsqueeze(1),
            dim=-1,
        )
        - spheres[:, 2].unsqueeze(1)
        - agents[:, :, 4].unsqueeze(2)
    )  # (n_envs, n_agents, n_spheres)

    # Calculate the closest point to the sphere
    closest_point_to_sphere = spheres[:, :2].unsqueeze(1) + (
        agents[:, :, :2].unsqueeze(2) - spheres[:, :2].unsqueeze(1)
    ) / torch.norm(
        agents[:, :, :2].unsqueeze(2) - spheres[:, :2].unsqueeze(1),
        dim=-1,
        keepdim=True,
    ) * spheres[
        :, 2
    ].unsqueeze(
        1
    )

    # Return the distance
    if env_index is not None:
        return closest_point_to_sphere[env_index, :, :, :], distance[env_index, :, :]
    return (
        closest_point_to_sphere,
        distance,
    )  # (n_envs, n_agents, n_spheres, 2), (n_envs, n_agents, n_spheres)


def distance_agents_to_boxes(
    agents: torch.Tensor, boxes: torch.Tensor, env_index: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the distance between a tensor of agents and a tensor of boxes

    Args:
        agents (torch.Tensor): The agents' positions
            agents.shape = (envs, n_agents, n_features)
            (x, y, vx, vy, radius, goal_x, goal_y, group_id)
        boxes (torch.Tensor): The boxes' vertices
            boxes.shape = (envs, n_boxes, 4, 2)
        env_index (int, optional): The index of the environment. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The closest point on the
        box and the distance
        closest_point.shape = (envs, n_agents, n_boxes, 2)
        distance.shape = (envs, n_agents, n_boxes)
    """

    # Calculate the distance between the agent and the box

    # use only valid vertices

    polygon_edges = torch.cat(
        [boxes, boxes[:, :, :1]], dim=2
    )  # close the loop (envs, n_boxes, 5, 2)
    edges_start = polygon_edges[:, :, :-1]  # Shape (envs, n_boxes, 4, 2)
    edges_end = polygon_edges[:, :, 1:]  # Shape (envs, n_boxes, 4, 2)

    edge_vectors = (
        edges_end - edges_start
    )  # vectors of the edges of the box (n_envs, n_boxes, 4, 2)

    # agents[:, :, :2].unsqueeze(2).unsqueeze(2) -> (n_envs, n_agents, 1, 1, 2)
    # edges_start.unsqueeze(1) -> (n_envs, 1, n_boxes, 4, 2)
    point_vectors = agents[:, :, :2].unsqueeze(2).unsqueeze(2) - edges_start.unsqueeze(
        1
    )  # vectors from the start of the edges to
    # the point (n_envs, n_agents, n_boxes, 4, 2)

    edge_lengths = (
        torch.linalg.norm(edge_vectors, dim=-1, keepdim=True) ** 2
    )  # squared lengths of the edges (n_envs, n_boxes, 4, 1)

    t = torch.clamp(
        (
            (point_vectors * edge_vectors.unsqueeze(1)).sum(dim=-1, keepdim=True)
            / edge_lengths.unsqueeze(1)
        ),
        0,
        1,
    )  # projection of the point on the edges (n_envs, n_agents, n_boxes, 4, 1)

    # (t = 0 -> start of the edge, t = 1 -> end of the edge)
    projections = edges_start.unsqueeze(1) + t * edge_vectors.unsqueeze(1)
    # projections of the point on the edges (n_envs, n_agents, n_boxes, 4, 2)

    distances = torch.linalg.norm(
        agents[:, :, :2].unsqueeze(2).unsqueeze(2) - projections, dim=-1
    )  # (n_envs, n_agents, n_boxes, 4)

    distances = distances - agents[:, :, 4].unsqueeze(2).unsqueeze(
        -1
    )  # (n_envs, n_agents, n_boxes, 4)

    # Calculate the closest point to the box

    index = (
        torch.argmin(distances, dim=-1, keepdim=True)
        .unsqueeze(-1)
        .expand(-1, -1, -1, -1, 2)
    )
    # (n_envs, n_agents, n_boxes, 2)
    closest_point_to_box = torch.gather(projections, dim=-2, index=index).squeeze(
        3
    )  # (n_envs, n_agents, n_boxes, 2)

    # Return the distance
    if env_index is not None:
        return (
            closest_point_to_box[env_index, :, :, :],
            distances[env_index, :, :].min(dim=-1).values,
        )
    return closest_point_to_box, distances.min(dim=-1).values


def entity_to_tensor(entity: Entity) -> torch.Tensor:
    """
    Convert a Entity object to a tensor based on its shape type

    Args:
        entity (Entity): The entity to convert

    Returns:
        torch.Tensor: The tensor representation of the entity
    """

    if isinstance(entity.shape, Box):
        return box_to_tensor(entity.state, entity.shape)
    elif isinstance(entity.shape, Sphere):
        return sphere_to_tensor(entity.state, entity.shape)
    elif isinstance(entity.shape, Line):
        return line_to_tensor(entity.state, entity.shape)
    else:
        raise NotImplementedError("The entity shape is not supported")


def box_to_tensor(state: EntityState, box: Box) -> torch.Tensor:
    """
    Convert a Box object to a tensor
    in the format [x, y] for each vertex

    Args:
        state (EntityState): The state of the box
        box (Box): The box to convert

    Returns:
        torch.Tensor: The tensor representation of the box
        (n_envs, 4, 2)
    """

    # transform the width and length to vertices of the box

    # Rotate normal vector by the angle of the box
    rotated_vector = torch.cat(
        [state.rot.cos(), state.rot.sin()], dim=-1
    )  # (n_envs, 2)
    rot_2 = state.rot + torch.pi / 2  # rotate by 90 degrees
    rotated_vector2 = torch.cat([rot_2.cos(), rot_2.sin()], dim=-1)  # (n_envs, 2)

    expanded_half_box_length = box.length / 2  # scalar
    expanded_half_box_width = box.width / 2  # scalar

    p1 = (
        state.pos  # (n_envs, 2)
        + rotated_vector * expanded_half_box_length  # (n_envs, 2)
        + rotated_vector2 * expanded_half_box_width  # (n_envs, 2)
    )  # top right (n_envs, 2)
    p2 = (
        state.pos
        - rotated_vector * expanded_half_box_length
        + rotated_vector2 * expanded_half_box_width
    )  # bottom right (n_envs, 2)
    p3 = (
        state.pos
        - rotated_vector * expanded_half_box_length
        - rotated_vector2 * expanded_half_box_width
    )  # bottom left (n_envs, 2)
    p4 = (
        state.pos
        + rotated_vector * expanded_half_box_length
        - rotated_vector2 * expanded_half_box_width
    )  # top left (n_envs, 2)

    return (
        torch.stack([p1, p2, p3, p4]).permute(1, 0, 2).to(state.pos.device)
    )  # (n_envs, 4, 2)


def sphere_to_tensor(state: EntityState, sphere: Sphere) -> torch.Tensor:
    """
    Convert a Sphere object to a tensor
    representing the center and radius of the sphere
    Args:
        sphere (Sphere): The sphere to convert

    """
    return torch.tensor(
        [state.pos[0], state.pos[1], sphere.radius], device=state.pos.device
    )


def line_to_tensor(state: EntityState, line: Line) -> torch.Tensor:
    """
    Convert a Line object to a tensor
    representing the start and end points of the line

    Args:
        line (Line): The line to convert

    """
    half_length = line.length / 2  # scalar
    start = (
        state.pos  # (n_envs, 2)
        + torch.cat([state.rot.cos(), state.rot.sin()], dim=-1) * half_length
    )  # (n_envs, 2)
    end = (
        state.pos - torch.cat([state.rot.cos(), state.rot.sin()], dim=-1) * half_length
    )  # (n_envs, 2)
    return (
        torch.stack([start, end]).permute(1, 0, 2).to(state.pos.device)
    )  # (n_envs, 2, 2)
