import numpy as np
from utils.planner_utils import pose2posquat
from typing import Dict, List, Optional, Sequence, Union
from habitat_sim.simulator import Simulator
from habitat_sim.agent import AgentState
from utils.transform import (
    to_habitat_position,
    from_habitat_position,
    from_habitat_rotation,
)
from itertools import combinations, product
import matplotlib.pyplot as plt
from habitat_sim.nav import GreedyGeodesicFollower, ShortestPath
import copy
from scipy.spatial.transform import Rotation as R
from planner.planner_base import PlannerBase


class HabitatPlanner(PlannerBase):

    MAX_TRANS_STEP: float = 0.25  # meters per interp step
    MAX_ROT_STEP: float = np.pi / 6  # radians per interp step

    def __init__(self, sim: Simulator, params: Optional[Dict] = None) -> None:
        super().__init__()
        params = params or {}

        # Start/goal
        self.start_pos_hab: Optional[np.ndarray] = None
        self.goal_pos_hab: Optional[np.ndarray] = None

        self.pathfinder = sim.pathfinder
        self.shortest_path = ShortestPath()
        self.sim_agent = sim.get_agent(0)
        self.sim = sim
        self.path_follower = GreedyGeodesicFollower(self.pathfinder, self.sim_agent)

        self.actions: List = []

    def snap_point(self, point: Sequence[float]) -> np.ndarray:
        habitat_point = to_habitat_position(point)
        snapped = self.pathfinder.snap_point(habitat_point)
        point_out = from_habitat_position(snapped)
        return np.array(point_out, dtype=float)

    def get_bounds(self) -> Dict[str, float]:
        return {
            "low_x": self.bounds[0],
            "high_x": self.bounds[1],
            "low_y": self.bounds[2],
            "high_y": self.bounds[3],
            "low_z": self.bounds[4],
            "high_z": self.bounds[5],
        }

    def update_start_goal(
        self, start: Union[Dict, np.ndarray], goal: Union[Dict, np.ndarray]
    ) -> bool:
        """
        Accepts either dicts {"pos":..., "quat":...} or 4x4 poses.
        Attempts to nudge invalid start into nearest valid free location if needed.
        """
        # Normalize inputs
        if isinstance(start, np.ndarray) and start.shape == (4, 4):
            start = pose2posquat(start)
        if isinstance(goal, np.ndarray) and goal.shape == (4, 4):
            goal = pose2posquat(goal)

        self.start_quat = np.asarray(start["quat"], dtype=float).copy()
        self.goal_quat = np.asarray(goal["quat"], dtype=float).copy()

        self.goal_pos_prev = np.asarray(goal["pos"], dtype=float).copy()

        self.start_pos_hab = self.sim_agent.get_state().position

        self.goal_pos_hab = self.pathfinder.snap_point(
            to_habitat_position(self.goal_pos_prev.tolist())
        )

        # Check if in navigable space
        if not self.pathfinder.is_navigable(self.start_pos_hab):
            self.log(
                "error",
                self.logging_file,
                f"Start position is not in navigable space. {self.start_pos_hab}",
            )
            return False
        if not self.pathfinder.is_navigable(self.goal_pos_hab):
            self.log(
                "error",
                self.logging_file,
                f"Goal position is not in navigable space. {self.goal_pos_hab}",
            )
            return False

        self.start_pos = from_habitat_position(self.start_pos_hab)
        self.goal_pos = from_habitat_position(self.goal_pos_hab)

        return True

    def set_bounds(self, bounds: Sequence[float]) -> None:
        self.bounds = list(bounds)

    def get_motion_check_resolution(self) -> float:
        return min(self._max_dist2free, self._min_dist2occ) * 0.5

    def solve(
        self,
        time_limit: float = 5.0,
        method: str = "rrtstar",
        depth: Optional[np.ndarray] = None,
    ) -> bool:
        if np.allclose(self.start_pos_hab, self.goal_pos_hab, atol=0.25):
            self.actions = []
            return True

        self.shortest_path.requested_start = self.start_pos_hab
        self.shortest_path.requested_end = self.goal_pos_hab

        if not self.pathfinder.find_path(self.shortest_path):
            return False

        points = self.shortest_path.points

        actions = []

        current_state_copy = copy.deepcopy(self.sim_agent.get_state())

        try:
            for i in range(1, len(points)):
                end_hab = points[i]

                section = self.path_follower.find_path(end_hab)

                for action in section:
                    if action is None:
                        break
                    self.sim_agent.act(action)
                    actions.append(action)
        finally:
            self.sim_agent.set_state(current_state_copy)

        self.actions = actions

        if self.actions is None or len(self.actions) == 0 or self.actions[0] is None:
            self.log("error", self.logging_file, "No path found by planner.")
            return False

        return True

    def state_to_posquat(self, state: AgentState) -> Dict[str, np.ndarray]:
        pos = from_habitat_position(state.position)
        rotmat = from_habitat_rotation(state.rotation)
        pose = np.eye(4)
        pose[:3, :3] = rotmat
        pose[:3, 3] = pos
        posquat = pose2posquat(pose)
        return posquat

    def interpolate_path(
        self,
        num_interp_points: int = 1,
        external_waypoints: Optional[List[Sequence[float]]] = None,
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        current_state_copy = copy.deepcopy(self.sim_agent.get_state())
        path = [self.state_to_posquat(current_state_copy)]
        try:
            for action in self.actions:
                if action is None:
                    break
                self.sim_agent.act(action)
                state = self.sim_agent.get_state()
                posquat = self.state_to_posquat(state)
                path.append(posquat)

            # If the last position is within the goal_radius, calculate orientation error
            last_pos_hab = to_habitat_position(path[-1]["pos"])
            goal_pos_hab = self.goal_pos_hab
            dist_to_goal = np.linalg.norm(
                np.array(last_pos_hab) - np.array(goal_pos_hab)
            )
            goal_radius = self.path_follower.goal_radius
            if dist_to_goal <= goal_radius:
                rot_goal = R.from_quat(self.goal_quat)
                rot_last = R.from_quat(path[-1]["quat"])
                rot_diff = rot_goal * rot_last.inv()
                euler_diff = rot_diff.as_euler("xyz", degrees=False)

                # Use the z difference and get how many 30 degree rotations are needed
                yaw_diff = euler_diff[2]
                rotations = int(np.round(yaw_diff / (np.pi / 6)))
                direction = np.sign(rotations) if rotations != 0 else 0
                num_rotations = abs(rotations)

                for _ in range(num_rotations):
                    action = 2 if direction > 0 else 3  # 3: turn_right, 2: turn_left

                    # Insert before None action
                    if len(self.actions) > 0 and self.actions[-1] is None:
                        self.actions = self.actions[:-1] + [action, None]
                    else:
                        self.actions.append(action)

                    self.sim_agent.act(action)
                    state = self.sim_agent.get_state()
                    posquat = self.state_to_posquat(state)
                    path.append(posquat)
        finally:
            self.sim_agent.set_state(current_state_copy)
        self.solution = path[1:]
        return self.solution

    def plot_3d(self) -> None:

        axes = plt.figure().add_subplot(111, projection="3d")

        # Set limits around bbox
        min_min = min(self.bounds)
        max_max = max(self.bounds)

        axes.set_xlim(min_min - 1, max_max + 1)
        axes.set_ylim(min_min - 1, max_max + 1)
        axes.set_zlim(min_min - 1, max_max + 1)

        # label axes
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        axes.set_zlabel("Z")

        # Draw x, y, z axes as red, green, blue arrows
        axes.quiver(
            0,
            0,
            0,
            1,
            0,
            0,
            length=1.0,
            color="r",
            arrow_length_ratio=0.1,
        )  # X axis

        axes.quiver(
            0,
            0,
            0,
            0,
            1,
            0,
            length=1.0,
            color="g",
            arrow_length_ratio=0.1,
        )  # Y axis

        axes.quiver(
            0,
            0,
            0,
            0,
            0,
            1,
            length=1.0,
            color="b",
            arrow_length_ratio=0.1,
        )  # Z axis

        # Plot occupied voxels
        if self._occ_kdt is not None:
            occ_pts = self._occ_kdt.data
            occ_pts = occ_pts[occ_pts[:, 2] < (self.bounds[4] + self.bounds[5]) / 2]

            axes.scatter(
                occ_pts[:, 0],
                occ_pts[:, 1],
                occ_pts[:, 2],
                c="gray",
                marker=".",
                s=1,
                alpha=0.5,
                label="occupied voxels",
            )
        # Plot start and goal
        if self.start_pos is not None:
            axes.scatter(
                self.start_pos[0],
                self.start_pos[1],
                self.start_pos[2],
                c="green",
                marker="o",
                s=100,
                label="start",
            )
        if self.goal_pos is not None:
            axes.scatter(
                self.goal_pos[0],
                self.goal_pos[1],
                self.goal_pos[2],
                c="red",
                marker="o",
                s=100,
                label="goal",
            )
        # Plot path if available
        poses = self.get_solution_path(return_type="mat")
        if poses is not None and len(poses) > 0:
            path_pts = np.array([pose[:3, 3] for pose in poses])
            axes.plot3D(
                path_pts[:, 0],
                path_pts[:, 1],
                path_pts[:, 2],
                c="blue",
                label="path",
            )

        # Draw bounding box
        if self.bounds is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
            r = [
                [x_min, x_max],
                [y_min, y_max],
                [z_min, z_max],
            ]
            for s, e in combinations(
                np.array(list(product(*r))), 2
            ):  # all combinations of corners
                if (
                    np.sum(np.abs(s - e)) == r[0][1] - r[0][0]
                    or np.sum(np.abs(s - e)) == r[1][1] - r[1][0]
                    or np.sum(np.abs(s - e)) == r[2][1] - r[2][0]
                ):
                    axes.plot3D(*zip(s, e), color="black")

        # Plot requested start/goal
        if self.start_pos_hab is not None:
            rs = from_habitat_position(self.start_pos_hab)
            axes.scatter(
                rs[0],
                rs[1],
                rs[2],
                c="cyan",
                marker="^",
                s=100,
                label="req start",
            )
        if self.goal_pos_hab is not None:
            re = from_habitat_position(self.goal_pos_hab)
            axes.scatter(
                re[0],
                re[1],
                re[2],
                c="orange",
                marker="^",
                s=100,
                label="req goal",
            )

        if self.goal_pos_prev is not None:
            gp = self.goal_pos_prev
            axes.scatter(
                gp[0],
                gp[1],
                gp[2],
                c="magenta",
                marker="x",
                s=100,
                label="orig goal",
            )

        plt.show(block=True)
