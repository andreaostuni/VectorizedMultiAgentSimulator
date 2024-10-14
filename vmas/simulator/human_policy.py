import torch
from typing import Tuple, List
import rvo2


class ORCAPolicy:
    def __init__(
        self,
        time_step,
        neighbor_dist,
        max_neighbors,
        time_horizon,
        time_horizon_obst,
        radius,
        max_speed,
    ):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        self.time_step = time_step
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_horizon_obst = time_horizon_obst
        self.radius = radius
        self.max_speed = max_speed
        self.sim = None

    def compute_action(
        self,
        curr_vel: Tuple[float],
        goal: Tuple[float],
        humans: List[Tuple[float]],
        obstacles: List[float],
        u_range,
    ):
        """
        Compute the action given the observation

        :param obs: observation
        :param u_range: action range
        :return: action

        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken
        """
        if self.sim is not None and self.sim.getNumAgents() != len(humans) + 1:
            del self.sim
            self.sim = None

        params_ = [
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
        ]  # parameters common for the agent and humans in the simulation

        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(
                self.time_step,
                self.neighbor_dist,
                *params_,
                self.radius,
                self.max_speed,
            )

            # we set the position of the agent in the local coordinate system
            self.sim.addAgent(
                (0, 0),
                self.time_horizon,  # time_horizon
                *params_,
                self.radius + 0.01 + self.safety_space,  # radius
                self.max_speed,  # v_pref
                curr_vel,
            )  # velocity
            for human in humans:
                self.sim.addAgent(
                    human,
                    self.time_horizon,  # time_horizon
                    *params_,
                    self.radius + 0.01 + self.safety_space,  # radius
                    self.max_speed,  # v_pref
                    (0, 0),
                )  # velocity
        else:
            self.sim.setAgentPosition(0, (0, 0))
            self.sim.setAgentVelocity(0, curr_vel)
            for i, human_state in enumerate(humans):
                self.sim.setAgentPosition(i + 1, human_state)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        # The agent will follow this direction if there are no other agents in the way

        pref_vel = goal / torch.norm(torch.tensor(goal))

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))

        for i, human_state in enumerate(humans):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()

        # get the new velocity of the agent
        new_vel = self.sim.getAgentVelocity(0)

        return new_vel
