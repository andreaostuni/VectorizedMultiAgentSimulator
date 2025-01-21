import rvo2
from vmas.simulator.core import Agent, World, Line, Sphere, Box
import torch
from vmas.simulator.human_dynamics.roadmap import Roadmap, find_vertices


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
        self.simulations = []
        self.robots = []  # robots are agents with policies
        self.humans = []  # humans are agents with scripts
        self.obstacles = []
        self.safety_space = safety_space
        self.human_velocities = None
        self.batch_dim = world.batch_dim
        self.device = world.device
        self.roadmaps = []

    def reset(self, world: World):
        """
        Reset the simulation.
        """
        for simulation in self.simulations:
            del simulation
        self.simulations = []
        for env_index in range(self.batch_dim):
            self.simulations.append(
                rvo2.PyRVOSimulator(
                    self.time_step,
                    self.neighbor_dist,
                    self.max_neighbors,
                    self.time_horizon,
                    self.time_horizon_obst,
                    self.radius,
                    self.max_speed,
                )
            )

            for robot in world.policy_agents:
                self.simulations[env_index].addAgent(
                    tuple(robot.state.pos[env_index].tolist()),
                    self.neighbor_dist,
                    self.max_neighbors,
                    self.time_horizon,
                    self.time_horizon_obst,
                    self.radius + 0.01 + self.safety_space,
                    self.max_speed,
                    tuple(robot.state.vel[env_index].tolist()),
                )
            for human in world.scripted_agents:
                self.simulations[env_index].addAgent(
                    tuple(human.state.pos[env_index].tolist()),
                    self.neighbor_dist,
                    self.max_neighbors,
                    self.time_horizon,
                    self.time_horizon_obst,
                    self.radius + 0.01 + self.safety_space,
                    self.max_speed,
                    tuple(human.state.vel[env_index].tolist()),
                )
            self.roadmaps.append(Roadmap(self.device, self.simulations[env_index]))
        self.robots = world.policy_agents
        self.humans = world.scripted_agents

        for landmark in world.landmarks:
            # if the landmark name contains "obstacle", add it as an obstacle
            if landmark.collide:
                if isinstance(landmark.shape, Sphere):
                    self.obstacles.append(
                        (landmark.state.pos[0], landmark.shape.radius)
                    )

                elif isinstance(landmark.shape, Line) or isinstance(
                    landmark.shape, Box
                ):
                    p1, p2, p3, p4 = find_vertices(landmark, 0, self.device)
                    self.obstacles.append((p1, p2, p3, p4))
                else:
                    raise ValueError("Obstacle shape not supported")
        for env_index in range(self.batch_dim):
            for obstacle in self.obstacles:
                self.simulations[env_index].addObstacle(
                    [
                        tuple(obstacle[0][env_index].tolist()),
                        tuple(obstacle[1][env_index].tolist()),
                        tuple(obstacle[2][env_index].tolist()),
                        tuple(obstacle[3][env_index].tolist()),
                    ]
                )

            self.simulations[env_index].processObstacles()

    def simulate_policy(self, world: World):
        """
        Run the simulation for one time step.

        :return:
        """
        if self.robots is None or self.humans is None:
            raise ValueError("Robots and humans must be set before simulation")
        if self.simulations is None:
            raise ValueError("Simulations must be set before simulation")

        if not self.roadmaps[0].roadmap:
            for env_index in range(self.batch_dim):
                self.roadmaps[env_index].setup_scenario(
                    [landmark for landmark in world.landmarks if landmark.collide],
                    [
                        agent.goal.state.pos[env_index]
                        for agent in self.robots + self.humans
                    ],
                )
                self.roadmaps[env_index].build_roadmap(self.simulations[env_index])

        self.human_velocities = torch.zeros(self.batch_dim, len(self.humans), 2)

        for env_index in range(self.batch_dim):
            if self.simulations[env_index].getNumAgents() != len(self.robots) + len(
                self.humans
            ):
                raise ValueError(
                    "Number of agents in simulation must match number of robots and humans"
                )
            # Update the position of agents in the simulation
            for i, robot in enumerate(self.robots):
                self.simulations[env_index].setAgentPosition(
                    i, tuple(robot.state.pos[env_index].tolist())
                )
                self.simulations[env_index].setAgentVelocity(
                    i, tuple(robot.state.vel[env_index].tolist())
                )
            for i, human in enumerate(self.humans):
                self.simulations[env_index].setAgentPosition(
                    i + len(self.robots), tuple(human.state.pos[env_index].tolist())
                )
                self.simulations[env_index].setAgentVelocity(
                    i + len(self.robots), tuple(human.state.vel[env_index].tolist())
                )

            # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
            for i, robot in enumerate(self.robots):
                vel_pref = robot.goal.state.pos[env_index] - robot.state.pos[env_index]
                # vel_pref = self.roadmaps[env_index].get_pref_velocity(
                #     robot.state.pos[env_index],
                #     robot.goal.state.pos[env_index],
                #     i,
                #     self.simulations[env_index],
                # )
                vel_pref /= vel_pref.norm()
                self.simulations[env_index].setAgentPrefVelocity(i, tuple(vel_pref))

            for i, human in enumerate(self.humans):
                vel_pref = human.goal.state.pos[env_index] - human.state.pos[env_index]
                # vel_pref = self.roadmaps[env_index].get_pref_velocity(
                #     human.state.pos[env_index],
                #     human.goal.state.pos[env_index],
                #     i + len(self.robots),
                #     self.simulations[env_index],
                # )
                vel_pref /= vel_pref.norm()
                self.simulations[env_index].setAgentPrefVelocity(
                    i + len(self.robots), tuple(vel_pref)
                )

            self.simulations[env_index].doStep()

            human_velocities = []
            for i, human in enumerate(self.humans):
                if (
                    human.state.pos[env_index] - human.goal.state.pos[env_index]
                ).norm() < 0.1:
                    human_velocities.append((0, 0))
                    continue
                human_velocities.append(
                    self.simulations[env_index].getAgentVelocity(i + len(self.robots))
                )
            self.human_velocities[env_index] = torch.tensor(
                human_velocities, device=self.device
            )

    def find_agent_index(self, agent: Agent):
        """
        Find the index of the agent in the simulation.
        """
        for i, human in enumerate(self.humans):
            if human is agent:
                return i
        raise ValueError("Agent not found in simulation")

    def run(self, agent, world):
        """
        Return the action to take for the agent.
        """
        index = self.find_agent_index(agent)
        if self.human_velocities is None:
            control = torch.zeros(self.batch_dim, 2)
        else:
            control = self.human_velocities[:, index]  # (batch_dim, agents, 2)

        control = torch.clamp(control, min=-agent.u_range, max=agent.u_range)
        agent.action.u = control * agent.u_multiplier
