from typing import List
from heapq import heappush, heappop, heapify
import torch
from vmas.simulator.core import Landmark, Line, Box
import rvo2

# use the following code to create a roadmap for the agents
# and guide them around the obstacles.


class RoadmapVertex:
    def __init__(self, position: List[float]):  # position is a list of floats
        self.position = position
        self.neighbors = []
        self.dist_to_goal = []


class Roadmap:
    def __init__(self, device, sim: rvo2.PyRVOSimulator):

        self.roadmap = []
        self.goals = []
        self.device = device
        self.safety_space = sim.getAgentRadius(0) * 1.5

    def setup_scenario(self, obstacles: List[Landmark], goals: List[torch.Tensor]):
        """Add waypoints and obstacles to the roadmap."""

        for i, goal in enumerate(goals):
            self.roadmap.append(RoadmapVertex(goal))
            self.goals.append(i)

        for obstacle in obstacles:
            """Add waypoints around the obstacles."""
            p1, p2, p3, p4 = find_vertices(obstacle, self.safety_space, self.device)

            self.roadmap.extend(
                [
                    RoadmapVertex(p1),
                    RoadmapVertex(p2),
                    RoadmapVertex(p3),
                    RoadmapVertex(p4),
                ]
            )

    def build_roadmap(self, sim: rvo2.PyRVOSimulator):
        for i in range(len(self.roadmap)):
            for j in range(len(self.roadmap)):
                if sim.queryVisibility(
                    tuple(self.roadmap[i].position.flatten().tolist()),
                    tuple(self.roadmap[j].position.flatten().tolist()),
                    self.safety_space,
                ):
                    self.roadmap[i].neighbors.append(j)
            self.roadmap[i].dist_to_goal = torch.inf * torch.ones(
                len(self.goals), device=self.device
            )

        # Compute the distance to each of the goals (the first vertices)
        # for all vertices using Dijkstra's algorithm.

        for i in range(len(self.goals)):
            # Initialize the priority queue
            self.roadmap[self.goals[i]].dist_to_goal[i] = 0.0
            pq = [(0.0, self.goals[i])]
            heapify(pq)
            # heappush(pq, (0.0, self.goals[i]))
            # heappush(pq, (0.0, i))
            visited = set()

            while pq:
                dist, u = heappop(pq)

                if u in visited:
                    continue
                visited.add(u)

                for v in self.roadmap[u].neighbors:
                    if v not in visited:
                        dist_uv = torch.norm(
                            self.roadmap[v].position - self.roadmap[u].position
                        )
                        if (
                            self.roadmap[v].dist_to_goal[i]
                            > self.roadmap[u].dist_to_goal[i] + dist_uv
                        ):
                            self.roadmap[v].dist_to_goal[i] = (
                                self.roadmap[u].dist_to_goal[i] + dist_uv
                            )
                            heappush(pq, (dist + dist_uv, v))

    def get_pref_velocity(
        self, agent_pos, agent_goal, goal_index, sim: rvo2.PyRVOSimulator
    ):
        """
        Set the preferred velocity to be a vector of unit magnitude (speed) in the
        direction of the visible roadmap vertex that is on the shortest path to the
        goal.
        """

        min_dist = torch.inf
        min_vertex = -1
        for j in range(len(self.roadmap)):
            if torch.norm(self.roadmap[j].position - agent_pos) + self.roadmap[
                j
            ].dist_to_goal[goal_index] < min_dist and sim.queryVisibility(
                tuple(agent_pos.tolist()),
                tuple(self.roadmap[j].position.tolist()),
                self.safety_space,
            ):
                min_dist = (
                    torch.norm(self.roadmap[j].position - agent_pos)
                    + self.roadmap[j].dist_to_goal[goal_index]
                )
                min_vertex = j

        print(f"agent number {goal_index} min_vertex: {min_vertex}")

        if min_vertex == -1:
            return torch.zeros(2, device=self.device)
        else:
            angle = torch.rand(1) * 2.0 * 3.14159
            dist = torch.rand(1) * 0.0001
            # Perturb a little to avoid deadlocks due to perfect symmetry.
            offset = torch.tensor([dist * torch.cos(angle), dist * torch.sin(angle)])
            if torch.norm(self.roadmap[min_vertex].position - agent_pos) < 0.1:
                if min_vertex == self.goals[goal_index]:
                    return torch.zeros(2, device=self.device)
                else:
                    return self.roadmap[min_vertex].position - agent_pos + offset

            else:
                return self.roadmap[min_vertex].position - agent_pos + offset


def find_vertices(obstacle: Landmark, offset: float, device: torch.device):
    """
    Find the vertices of the obstacles and set a safety offset.
    """
    if isinstance(obstacle.shape, Line) or isinstance(obstacle.shape, Box):
        new_half_length = obstacle.shape.length / 2 + offset
        new_half_width = obstacle.shape.width / 2 + offset
        # convert line to two points ordered in conterclockwise direction
        cos_yaw = torch.cos(obstacle.state.rot)
        sin_yaw = torch.sin(obstacle.state.rot)
        rot_matrix = torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw], dim=-1),
                torch.stack([sin_yaw, cos_yaw], dim=-1),
            ],
            dim=-1,
        ).squeeze()

        p1 = (
            obstacle.state.pos
            + (
                torch.tensor(
                    [new_half_length, new_half_width],
                    device=device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                @ rot_matrix
            ).squeeze()
        )
        p2 = (
            obstacle.state.pos
            + (
                torch.tensor(
                    [-new_half_length, new_half_width],
                    device=device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                @ rot_matrix
            ).squeeze()
        )
        p3 = (
            obstacle.state.pos
            + (
                torch.tensor(
                    [-new_half_length, -new_half_width],
                    device=device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                @ rot_matrix
            ).squeeze()
        )

        p4 = (
            obstacle.state.pos
            + (
                torch.tensor(
                    [new_half_length, -new_half_width],
                    device=device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
                @ rot_matrix
            ).squeeze()
        )

        return p1, p2, p3, p4
