import numpy as np
from typing import Dict, List, Optional, Sequence, Union, Tuple

from itertools import combinations, product
import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation as R
from planner.planner_base import PlannerBase

from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.obs_transformers.utils import image_resize
from utils.planner_utils import pose2posquat

import torch


class PointnavPlanner(PlannerBase):

    MAX_TRANS_STEP: float = 0.25  # meters per interp step
    MAX_ROT_STEP: float = np.pi / 6  # radians per interp step
    MAX_FORWARD_HEAT: int = 15  # max steps without approaching goal before giving up
    MAX_ROTATION_HEAT: int = 15  # max steps of rotating around itself before giving up
    is_pointnav_planner: bool = True

    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__(params=params)
        params = params or {}

        # PointNav policy
        pointnav_path: str = params.get(
            "pointnav_weights_path", "model_weights/pointnav_weights.pth"
        )

        self.pointnav_policy = WrappedPointNavResNetPolicy(pointnav_path)
        self.last_goal = np.eye(4)
        self.rho: float = 0.0
        self.theta: float = 0.0

        self.start_pose: np.ndarray = None
        self.goal_pose: np.ndarray = None

        self.action: List[int] = []

        self.is_first_step: bool = True

        self.pointnav_policy.reset()

        self.last_goal: Optional[np.ndarray] = None

        # Consecutive failures to approach goal
        self.forward_failure_heat: int = 0
        self.rotation_heat: int = 0
        self.close_enough: bool = False
        self.minimum_rho: float = float("inf")

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

        self.start_pose = start
        self.goal_pose = goal

        # Normalize inputs
        if isinstance(start, np.ndarray) and start.shape == (4, 4):
            start = pose2posquat(start)
        if isinstance(goal, np.ndarray) and goal.shape == (4, 4):
            goal = pose2posquat(goal)

        self.start_pos = np.asarray(start["pos"], dtype=float).copy()
        self.goal_pos = np.asarray(goal["pos"], dtype=float).copy()
        self.start_quat = np.asarray(start["quat"], dtype=float).copy()
        self.goal_quat = np.asarray(goal["quat"], dtype=float).copy()

        self.start_pos = self.snap_to_free(self.start_pos)
        self.goal_pos = self.snap_to_free(self.goal_pos)

        dist = np.linalg.norm(self.start_pos - self.goal_pos)

        # If start is very close to goal, nudge goal to nearest navigable point
        if dist < 1.5:
            self.goal_pos = self.get_closest_navigable_point(
                self.start_pos, self.goal_pos
            )

        if self.last_goal is None:
            self.last_goal = self.goal_pose
        else:
            if not np.array_equal(self.goal_pose, self.last_goal):
                if np.linalg.norm(self.goal_pose[:3, 3] - self.last_goal[:3, 3]) > 0.1:
                    # Reset pointnav
                    self.is_first_step = True

                # Reset heuristics
                self.minimum_rho = float("inf")
                self.close_enough = False
                self.forward_failure_heat = 0
                self.rotation_heat = 0
                self.last_goal = self.goal_pose

        return True

    def get_closest_navigable_point(self, start_position, target_position):
        """
        Get the point that is closest to the object along the agent-object line that is not occupied at navigation level.
        """
        start_position[2] = self.nav_level + 0.1
        target_position[2] = self.nav_level + 0.1

        start_position = np.array(start_position)
        target_position = np.array(target_position)

        # Compute the direction from the camera to the object
        direction = target_position - start_position

        if np.linalg.norm(direction) < 1e-6:
            return target_position

        direction /= np.linalg.norm(direction)

        # Find the point closest to the object along the line that is not occupied
        step_size = 0.1  # step size along the line
        max_distance = np.linalg.norm(target_position - start_position)
        num_steps = int(max_distance / step_size)
        for step in range(num_steps + 1):
            point = target_position - direction * step * step_size
            if not self.isoccupied(point) and self.isfree(point):
                return point
        return target_position  # fallback to object position if all points are occupied

    def set_bounds(self, bounds: Sequence[float]) -> None:
        self.bounds = list(bounds)

    def get_motion_check_resolution(self) -> float:
        return min(self._max_dist2free, self._min_dist2occ) * 0.5

    def solve(
        self,
        time_limit: float = 5.0,
        method: str = "default",
        depth: Optional[np.ndarray] = None,
    ) -> bool:
        if depth is None:
            raise ValueError("Depth input is required for PointNav planning.")

        used_pointnav = False

        too_hot = (
            self.forward_failure_heat >= self.MAX_FORWARD_HEAT
            or self.rotation_heat >= self.MAX_ROTATION_HEAT
        )

        rho, theta = self.rho_theta(self.start_pose, self.goal_pos)

        self.rho, self.theta = rho, theta

        action = None

        # If it was previously categorized as close enough, keep it that way
        close_enough = self.close_enough or rho < 0.2

        # If the last action was moving forward, check if we got closer to the goal
        if self.action == 1:  # move_forward
            if self.minimum_rho - rho > 0.2:
                self.forward_failure_heat = 0
                self.minimum_rho = rho
            else:
                self.forward_failure_heat += 1

        if self.action in [2, 3]:  # turn_left or turn_right
            self.rotation_heat += 1

        # If it's already close enough and the last few actions didn't help, just align
        if rho < 0.5 and self.forward_failure_heat >= 3:
            close_enough = True

        self.close_enough = close_enough

        # If already at goal or failed to approach rotate to align
        if close_enough or too_hot:
            current_rot = R.from_quat(self.start_quat)
            goal_rot = R.from_quat(self.goal_quat)
            relative_rot = goal_rot * current_rot.inv()
            yaw_angle = relative_rot.as_euler("zyx", degrees=False)[0]

            rotations = int(np.round(yaw_angle / (np.pi / 6)))
            direction = np.sign(rotations) if rotations != 0 else 0
            num_rotations = abs(rotations)

            if num_rotations == 0:
                action = None  # at goal
            else:
                action = 2 if direction > 0 else 3  # 3: turn_right, 2: turn_left
        else:
            action = self.pointnav(rho=rho, theta=theta, depth=depth)
            used_pointnav = True

            if action == 0:
                action = None

        self.action = action

        self.log(
            "info",
            self.logging_file,
            f"Planned actions: {action}, rho: {rho:.2f}, theta: {theta:.2f}, forward heat: {self.forward_failure_heat} rotation heat: {self.rotation_heat}",
        )

        # At goal, reset heat and close_enough
        if action is None:
            self.close_enough = False
            self.forward_failure_heat = 0
            self.rotation_heat = 0

        if action == 1:
            self.rotation_heat = 0

        if not used_pointnav:
            self.pointnav_policy.reset()
            self.is_first_step = True

        # Only fail if the policy failed to reach the goal after many attempts
        return not too_hot

    def interpolate_path(
        self,
        num_interp_points: int = 1,
        external_waypoints: Optional[List[Sequence[float]]] = None,
    ) -> Optional[List[Dict[str, np.ndarray]]]:

        if self.action is None:
            self.solution = []
        else:
            self.solution = [pose2posquat(self.goal_pose)]
        return self.solution

    def rho_theta(self, pose_SE3, goal_pos):
        """
        pose_SE3 : 4x4 robot pose matrix (world_T_robot)
        goal_pos : (gx, gy, gz) in world frame
        Convention:
            X forward, Y left, Z up
        """
        # Robot position
        x, y, _ = pose_SE3[:3, 3]
        gx, gy, _ = goal_pos

        # Extract yaw around Z axis
        R = pose_SE3[:3, :3]
        # yaw = atan2(sin(theta), cos(theta))
        yaw = np.arctan2(R[1, 0], R[0, 0])

        # World frame difference (2D)
        dx = gx - x  # forward/back
        dy = gy - y  # left/right

        # Rotate Δ into robot local frame
        cy = np.cos(-yaw)
        sy = np.sin(-yaw)

        dx_body = dx * cy - dy * sy  # forward direction
        dy_body = dx * sy + dy * cy  # left direction

        # Polar coordinates (Habitat/PointNav style)
        rho = np.sqrt(dx_body**2 + dy_body**2)
        theta = np.arctan2(dy_body, dx_body) - np.pi / 2

        return rho, theta

    # Adapted from: https://github.com/bdaiinstitute/vlfm/blob/main/vlfm/policy/base_objectnav_policy.py
    # Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
    def pointnav(
        self,
        rho: float,
        theta: float,
        depth: np.ndarray,
    ):

        if self.is_first_step:
            self.pointnav_policy.reset()
            masks = torch.zeros(
                1, 1, device="cuda", dtype=torch.bool
            )  # (num_envs, 1) all zeros to reset rnn hidden state
        else:
            masks = torch.ones(
                1, 1, device="cuda", dtype=torch.bool
            )  # (num_envs, 1) all ones to keep rnn hidden state

        # reshape depth to (1, H, W, 1)
        depth = torch.as_tensor(depth.copy()).unsqueeze(0).unsqueeze(-1)
        depth = image_resize(
            depth, size=(224, 224), channels_last=True, interpolation_mode="area"
        )

        rho_theta_tensor = torch.tensor(
            [[rho, theta]], dtype=torch.float32, device="cuda"
        )

        obs_pointnav = {
            "depth": depth.to("cuda"),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }

        action = self.pointnav_policy.act(obs_pointnav, masks, deterministic=True)

        action = action.cpu().numpy()[0][0]

        self.is_first_step = False

        return action

    def next_action(self) -> Optional[int]:
        return self.action

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

        plt.show(block=True)
