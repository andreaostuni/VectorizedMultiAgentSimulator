import torch
from vmas.simulator.utils import ScenarioUtils
from typing import List, Tuple, Optional
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World


""" Utils funtions for the initialization of the scripted agents based on the scenario """

# create enum for the scenarios
from enum import IntEnum


class Scenario(IntEnum):
    # CIRCLE_CROSSING = "circle_crossing"
    PASSING = 0
    CROSSING = 1
    # PASSING_CROSSING = "passing_crossing"
    OVERTAKING = 2
    RANDOM = 3
    STATIC = 4
    EASY = 5

    @classmethod
    def sample_scenario(cls):
        return Scenario(torch.randint(0, len(Scenario) - 1, (1,)).item())


"""
dictionary with the scenarios hyperparameters for the human agents bounding boxes
"""
SCENARIOS = {
    Scenario.PASSING: {
        "x_offset_factor": 0.0,  # offset from the center of the world in the x axis
        "y_offset_factor": 0.5,  # offset from the center of the world in the y axis
        "x_goal_width_factor": 1.0,
        "y_goal_width_factor": 0.5,
        "x_spawn_width_factor": 0.7,
        "y_spawn_width_factor": 0.5,
    },
    Scenario.CROSSING: {
        "x_offset_factor": 1.0,  # offset from the center of the world in the x axis
        "y_offset_factor": 0.0,  # offset from the center of the world in the y axis
        "x_goal_width_factor": 0.1,
        "y_goal_width_factor": 0.3,
        "x_spawn_width_factor": 0.1,
        "y_spawn_width_factor": 0.3,
    },
    Scenario.OVERTAKING: {
        "x_offset_factor": 0.0,  # offset from the center of the world in the x axis
        "y_offset_factor": 0.7,  # offset from the center of the world in the y axis
        "x_goal_width_factor": 1.0,
        "y_goal_width_factor": 0.2,
        "x_spawn_width_factor": 1.0,
        "y_spawn_width_factor": 0.2,
    },
    Scenario.RANDOM: {
        "x_offset_factor": 0.0,  # offset from the center of the world in the x axis
        "y_offset_factor": 0.0,  # offset from the center of the world in the y axis
        "x_goal_width_factor": 0.1,
        "y_goal_width_factor": 0.1,
        "x_spawn_width_factor": 1.0,
        "y_spawn_width_factor": 1.0,
    },
    Scenario.STATIC: {
        "x_offset_factor": 0.0,  # offset from the center of the world in the x axis
        "y_offset_factor": 0.0,  # offset from the center of the world in the y axis
        "x_goal_width_factor": 0.0,
        "y_goal_width_factor": 0.0,
        "x_spawn_width_factor": 1.0,
        "y_spawn_width_factor": 1.0,
    },
    Scenario.EASY: {
        "x_offset_factor": 0.0,  # offset from the center of the world in the x axis
        "y_offset_factor": 0.0,  # offset from the center of the world in the y axis
        "x_goal_width_factor": 0.1,
        "y_goal_width_factor": 0.1,
        "x_spawn_width_factor": 1.0,
        "y_spawn_width_factor": 1.0,
    },
}


def generate_scenario(
    robot_agents: List[Agent],
    human_agents: List[Agent],
    world: World,
    env_index: int,
    min_dist_between_entities: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    occupied_positions: Optional[torch.Tensor] = None,
    current_scenario=Scenario.sample_scenario(),
):
    """
    Wrapper function to generate the scenario based on the scenario

    """
    x_direction = 2 * torch.randint(0, 2, (1,)).item() - 1  # -1: down, 1: up
    y_direction = 2 * torch.randint(0, 2, (1,)).item() - 1  # -1: left, 1: right
    robot_spawn_bounds, robot_goal_bounds = generate_robot_bounds(
        x_bounds,
        y_bounds,
        (x_direction, y_direction),
        current_scenario=(
            current_scenario if current_scenario == Scenario.EASY else None
        ),
    )
    human_spawn_bounds, human_goal_bounds = generate_human_bounds(
        x_bounds, y_bounds, (x_direction, y_direction), current_scenario
    )

    spawn_positions, goal_positions = generate_agents_state(
        robot_agents,
        world,
        env_index,
        min_dist_between_entities,
        robot_spawn_bounds[0],
        robot_spawn_bounds[1],
        robot_goal_bounds[0],
        robot_goal_bounds[1],
        occupied_positions,
        current_scenario=(
            current_scenario if current_scenario == Scenario.EASY else None
        ),
    )

    spawn_positions, goal_positions = generate_agents_state(
        human_agents,
        world,
        env_index,
        min_dist_between_entities,
        human_spawn_bounds[0],
        human_spawn_bounds[1],
        human_goal_bounds[0],
        human_goal_bounds[1],
        occupied_positions,
        spawn_positions,
        goal_positions,
        current_scenario=current_scenario,
    )

    if occupied_positions is None:
        occupied_positions = torch.empty(
            (world.batch_dim if env_index is None else 1, 0, 2), device=world.device
        )

    occupied_positions = torch.cat(
        [occupied_positions, spawn_positions, goal_positions], dim=1
    )

    return occupied_positions


def generate_bounds(
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    x_direction: int,
    y_direction: int,
    x_offset_factor: float = 0.0,
    y_offset_factor: float = 0.9,
    x_goal_width_factor: float = 0.2,
    y_goal_width_factor: float = 0.1,
    x_spawn_width_factor: float = 0.7,
    y_spawn_width_factor: float = 0.1,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:

    x_mean = (x_bounds[0] + x_bounds[1]) / 2
    y_mean = (y_bounds[0] + y_bounds[1]) / 2

    x_semi_width = (x_bounds[1] - x_bounds[0]) / 2
    y_semi_width = (y_bounds[1] - y_bounds[0]) / 2

    # the spawn area is the upper or lower half of the world based on the direction
    y_offset = y_offset_factor * y_semi_width * y_direction
    x_offset = x_offset_factor * x_semi_width * x_direction
    x_spawn_bounds = (
        x_mean + x_offset - x_semi_width * x_spawn_width_factor,
        x_mean + x_offset + x_semi_width * x_spawn_width_factor,
    )
    y_spawn_bounds = (
        y_mean + y_offset - y_semi_width * y_spawn_width_factor,
        y_mean + y_offset + y_semi_width * y_spawn_width_factor,
    )

    x_goal_bounds = (
        x_mean - x_offset - x_semi_width * x_goal_width_factor,
        x_mean - x_offset + x_semi_width * x_goal_width_factor,
    )
    y_goal_bounds = (
        y_mean - y_offset - y_semi_width * y_goal_width_factor,
        y_mean - y_offset + y_semi_width * y_goal_width_factor,
    )

    return (x_spawn_bounds, y_spawn_bounds), (x_goal_bounds, y_goal_bounds)


def generate_robot_bounds(
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    direction: Tuple[int, int],
    current_scenario: Optional[Scenario] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if current_scenario is not None and current_scenario == Scenario.EASY:
        return generate_bounds(x_bounds, y_bounds, 0, 0, **SCENARIOS[current_scenario])
    return generate_bounds(
        x_bounds, y_bounds, x_direction=direction[0], y_direction=direction[1]
    )


def generate_human_bounds(
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    direction: Tuple[int, int],
    current_scenario: str,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # current_scenario = Scenario.STATIC
    match current_scenario:
        case "circle_crossing":
            pass
        case Scenario.PASSING:
            human_direction_x = direction[0]  # opposite direction to the robot
            human_direction_y = -direction[1]  # opposite direction to the robot
            return generate_bounds(
                x_bounds,
                y_bounds,
                human_direction_x,
                human_direction_y,
                **SCENARIOS[current_scenario],
            )
        case Scenario.CROSSING:
            human_direction_x = direction[0]
            human_direction_y = -direction[1]  # opposite direction to the robot
            return generate_bounds(
                x_bounds,
                y_bounds,
                human_direction_x,
                human_direction_y,
                **SCENARIOS[current_scenario],
            )
        case "passing_crossing":
            pass
        case Scenario.OVERTAKING:
            human_direction_x = direction[0]  # opposite direction to the robot
            human_direction_y = direction[1]  # opposite direction to the robot
            return generate_bounds(
                x_bounds,
                y_bounds,
                human_direction_x,
                human_direction_y,
                **SCENARIOS[current_scenario],
            )
        case Scenario.RANDOM:
            human_direction_x = 2 * torch.randint(0, 2, (1,)).item() - 1
            human_direction_y = 2 * torch.randint(0, 2, (1,)).item() - 1
            return generate_bounds(
                x_bounds,
                y_bounds,
                human_direction_x,
                human_direction_y,
                **SCENARIOS[current_scenario],
            )
        case Scenario.STATIC:
            return generate_bounds(x_bounds, y_bounds, 0, 0)
        case Scenario.EASY:
            """Spawn the human agents outside the robot spawn area"""
            # set the human agents spawn outside the world
            x_bounds_mean = (x_bounds[0] + x_bounds[1]) / 2
            y_bounds_mean = (y_bounds[0] + y_bounds[1]) / 2
            x_semi_width = (x_bounds[1] - x_bounds[0]) / 2
            y_semi_width = (y_bounds[1] - y_bounds[0]) / 2

            new_x_bounds = (
                x_bounds_mean + 2 * x_semi_width,
                x_bounds_mean + 3 * x_semi_width,
            )
            new_y_bounds = (
                y_bounds_mean + 2 * y_semi_width,
                y_bounds_mean + 3 * y_semi_width,
            )

            return generate_bounds(
                new_x_bounds,
                new_y_bounds,
                -direction[0],
                -direction[1],
                **SCENARIOS[current_scenario],
            )

        # TODO: direction list


def generate_agents_state(
    agents: List[Agent],
    world: World,
    env_index: int,
    min_dist_between_entities: float,
    spawn_x_bounds: Tuple[float, float],
    spawn_y_bounds: Tuple[float, float],
    goal_x_bounds: Tuple[float, float],
    goal_y_bounds: Tuple[float, float],
    occupied_positions: Optional[torch.Tensor] = None,
    spawn_positions: Optional[torch.Tensor] = None,
    goal_positions: Optional[torch.Tensor] = None,
    current_scenario: Optional[Scenario] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate the agents state based on the bounds of the spawn area and the goal area

    Args:
        - robot_agents: List of robot agents
        - human_agents: List of human agents
        - world: World object
        - env_index: Index of the environment
        - min_dist: Minimum distance between the entities
        - x_bounds: Tuple with the x bounds of the world
        - y_bounds: Tuple with the y bounds of the world
        - min_dist_between_entities: Minimum distance between the entities
        - occupied_positions: Tensor with the positions of the entities in the world
        - spawn_positions: Tensor with the positions of the other spawned entities
        - goal_positions: Tensor with the positions of the other goal entities

    Returns:
        - spawn_positions: Tensor with the positions of the spawned entities
        - goal_positions: Tensor with the positions of the goal entities
    """
    batch_size = world.batch_dim if env_index is None else 1
    if goal_positions is None:
        # create an empty tensor that can be concatenated with the occupied_positions
        goal_positions = torch.empty((batch_size, 0, 2), device=world.device)

    if spawn_positions is None:
        # create an empty tensor that can be concatenated with the occupied_positions
        spawn_positions = torch.empty((batch_size, 0, 2), device=world.device)
    if occupied_positions is None:
        occupied_positions = torch.empty((batch_size, 0, 2), device=world.device)

    occ_spawns = torch.cat([occupied_positions, spawn_positions], dim=1)
    # verify if occ spaws is not empty

    ScenarioUtils.spawn_entities_randomly(
        agents,
        world,
        env_index,
        min_dist_between_entities,
        spawn_x_bounds,
        spawn_y_bounds,
        occupied_positions=occ_spawns,
    )

    # Goal
    spawn_positions = torch.cat(
        [
            spawn_positions,
            torch.stack(
                [
                    (
                        agent.state.pos
                        if env_index is None
                        else agent.state.pos[env_index : env_index + 1]
                    )
                    for agent in agents
                ],
                dim=1,
            ),
        ],
        dim=1,
    )

    if current_scenario is not None and current_scenario == Scenario.EASY:
        """Generate the goal positions in a radius of 2 meters from the spawn positions"""
        for agent in agents:
            # easy_spawn_bounds_x = (
            #     max((agent.state.pos[:, 0] - 2.0), spawn_x_bounds[0]),
            #     min((agent.state.pos[:, 0] + 2.0), spawn_x_bounds[1]),
            # )

            # easy_spawn_bounds_y = (
            #     max((agent.state.pos[:, 1] - 2.0), spawn_y_bounds[0]),
            #     min((agent.state.pos[:, 1] + 2.0), spawn_y_bounds[1]),
            # )

            # occ_goal = torch.cat([occupied_positions, goal_positions], dim=1)
            # goal_pos = ScenarioUtils.find_random_pos_for_entity(
            #     occupied_positions=occ_goal,
            #     env_index=env_index,
            #     world=world,
            #     min_dist_between_entities=min_dist_between_entities,
            #     x_bounds=easy_spawn_bounds_x,
            #     y_bounds=easy_spawn_bounds_y,
            # )
            spawn_pos = (
                agent.state.pos
                if env_index is None
                else agent.state.pos[env_index : env_index + 1]
            )
            sign = torch.randint_like(spawn_pos, 0, 2) * 2 - 1
            goal_pos = spawn_pos + torch.empty_like(spawn_pos).uniform_(1.0, 3.0) * sign
            goal_pos[:, 0] = torch.clamp(
                goal_pos[:, 0],
                min=spawn_x_bounds[0],
                max=spawn_x_bounds[1],
            )
            goal_pos[:, 1] = torch.clamp(
                goal_pos[:, 1],
                min=spawn_y_bounds[0],
                max=spawn_y_bounds[1],
            )

            goal_positions = torch.cat([goal_positions, goal_pos.unsqueeze(1)], dim=1)
            agent.goal.set_pos(goal_pos.squeeze(1), batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = torch.linalg.norm(
                    agent.goal.state.pos - agent.state.pos, dim=1
                )
            else:
                agent.pos_shaping[env_index] = torch.linalg.norm(
                    agent.goal.state.pos[env_index] - agent.state.pos[env_index]
                ).unsqueeze(0)
        return spawn_positions, goal_positions

    if current_scenario is not None and current_scenario >= Scenario.RANDOM:
        sign = 1 if current_scenario == Scenario.STATIC else -1
        for agent in agents:
            if env_index is None:
                agent.goal.set_pos(sign * agent.state.pos, batch_index=env_index)
                agent.pos_shaping = torch.linalg.norm(
                    agent.goal.state.pos - agent.state.pos, dim=1
                )
            else:
                agent.goal.set_pos(
                    sign * agent.state.pos[env_index], batch_index=env_index
                )
                agent.pos_shaping[env_index] = torch.linalg.norm(
                    agent.goal.state.pos[env_index] - agent.state.pos[env_index]
                ).unsqueeze(0)
        goal_positions = torch.cat([goal_positions, sign * spawn_positions], dim=1)
        return spawn_positions, goal_positions

    for agent in agents:
        occ_goal = torch.cat([occupied_positions, goal_positions], dim=1)
        goal_pos = ScenarioUtils.find_random_pos_for_entity(
            occupied_positions=occ_goal,
            env_index=env_index,
            world=world,
            min_dist_between_entities=min_dist_between_entities,
            x_bounds=goal_x_bounds,
            y_bounds=goal_y_bounds,
        )
        goal_positions = torch.cat([goal_positions, goal_pos], dim=1)
        agent.goal.set_pos(goal_pos.squeeze(1), batch_index=env_index)

        if env_index is None:
            agent.pos_shaping = torch.linalg.norm(
                agent.goal.state.pos - agent.state.pos, dim=1
            )
        else:
            agent.pos_shaping[env_index] = torch.linalg.norm(
                agent.goal.state.pos[env_index] - agent.state.pos[env_index]
            ).unsqueeze(0)

    return spawn_positions, goal_positions
