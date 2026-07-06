import numpy as np
from typing import Dict, List, Optional, Sequence, Union
from scipy.spatial import KDTree

from utils.planner_utils import pose2posquat, posquat2pose, interpolate_waypoints
from utils.geometry import compute_alignment_transforms, pose_difference
from planner.occupancy_grid import OccupancyGrid

from planner.planner_client import PlannerClient


class Server3DPathPlanner:

    def __init__(self, params: Optional[Dict] = None) -> None:
        params = params or {}

        self.client = PlannerClient(port=12184)

        self.client.send_request(function="reset", data={})

        self.grid = OccupancyGrid(params)

    @property
    def use_freegrid(self) -> bool:
        return self.grid.use_freegrid

    @use_freegrid.setter
    def use_freegrid(self, val: bool) -> None:
        self.client.send_request(
            function="set_use_freegrid",
            data={"val": bool(val)},
        )
        self.grid.use_freegrid = bool(val)

    @property
    def use_occgrid(self) -> bool:
        return self.grid.use_occgrid

    @use_occgrid.setter
    def use_occgrid(self, val: bool) -> None:
        self.client.send_request(
            function="set_use_occgrid",
            data={"val": bool(val)},
        )
        self.grid.use_occgrid = bool(val)

    @property
    def min_dist2occ(self) -> float:
        return self.grid.min_dist2occ

    @min_dist2occ.setter
    def min_dist2occ(self, val: float) -> None:
        self.client.send_request(
            function="set_min_dist2occ",
            data={"val": float(val)},
        )
        self.grid.min_dist2occ = float(val)

    @property
    def max_dist2free(self) -> float:
        return self.grid.max_dist2free

    @max_dist2free.setter
    def max_dist2free(self, val: float) -> None:
        self.client.send_request(
            function="set_max_dist2free",
            data={"val": float(val)},
        )
        self.grid.max_dist2free = float(val)

    def isfree(self, point: Sequence[float]) -> bool:
        """
        True if sample is within max_dist2free of free voxelgrid (if KDTree set).
        If no free KDTree is set, returns True.
        """
        return self.grid.isfree(point)

    def isoccupied(self, point: Sequence[float]) -> bool:
        """
        True if sample is within min_dist2occ of occupied voxelgrid (if KDTree set).
        If no occ KDTree is set, returns False.
        """
        return self.grid.isoccupied(point)

    def is_state_valid(self, state: np.ndarray) -> bool:
        return self.grid.is_state_valid(state)

    def get_bounds(self) -> Dict[str, float]:
        result = self.client.send_request(function="get_bounds", data={})
        return result["value"]

    def set_bounds(self, bounds_input: Sequence[float]) -> None:
        """
        bounds_input: [low_x, high_x, low_y, high_y, low_z, high_z]
        """
        if isinstance(bounds_input, np.ndarray):
            bounds_input = bounds_input.tolist()

        self.client.send_request(
            function="set_bounds",
            data={"bounds": bounds_input},
        )

    def update_space(
        self, free_vx: Optional[np.ndarray] = None, occ_vx: Optional[np.ndarray] = None
    ) -> None:
        """
        free_vx: Nx3 array of free voxel coordinates
        occ_vx: Mx3 array of occupied voxel coordinates
        """
        self.grid.update_space(free_vx=free_vx, occ_vx=occ_vx)

        data = {}
        if free_vx is not None:
            data["free_vx"] = free_vx.tolist()
        if occ_vx is not None:
            data["occ_vx"] = occ_vx.tolist()

        self.client.send_request(
            function="update_space",
            data=data,
        )

    def update_start_goal(
        self, start: Union[Dict, np.ndarray], goal: Union[Dict, np.ndarray]
    ) -> bool:
        """
        Accepts either dicts {"pos":..., "quat":...} or 4x4 poses.
        Attempts to nudge invalid start into nearest valid free location if needed.
        """
        result = self.client.send_request(
            function="update_start_goal",
            data={
                "start": (
                    start
                    if isinstance(start, dict)
                    else {
                        "pos": pose2posquat(start)["pos"].tolist(),
                        "quat": pose2posquat(start)["quat"].tolist(),
                    }
                ),
                "goal": (
                    goal
                    if isinstance(goal, dict)
                    else {
                        "pos": pose2posquat(goal)["pos"].tolist(),
                        "quat": pose2posquat(goal)["quat"].tolist(),
                    }
                ),
            },
        )
        return bool(result["value"])

    def get_start_goal(self) -> Dict[str, Dict[str, np.ndarray]]:
        result = self.client.send_request(function="get_start_goal", data={})
        return result["value"]

    def get_motion_check_resolution(self) -> float:
        result = self.client.send_request(
            function="get_motion_check_resolution", data={}
        )
        return float(result["value"])

    def solve(self, time_limit: float = 5.0, method: str = "rrtstar") -> bool:
        result = self.client.send_request(
            function="solve",
            data={"time_limit": float(time_limit), "method": method},
        )
        return bool(result["value"])

    def interpolate_path(
        self,
        num_interp_points: int = 1,
        external_waypoints: Optional[List[Sequence[float]]] = None,
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        """
        Build a dense, orientation-aware path. If `external_waypoints` is given, uses those.
        Otherwise uses the OMPL solution path. Orientation is:
          - start quat at first,
          - goal quat at last,
          - for intermediates: face the next waypoint.
        """
        data = {"num_interp_points": int(num_interp_points)}
        if external_waypoints is not None:
            data["external_waypoints"] = [list(wp) for wp in external_waypoints]

        result = self.client.send_request(
            function="interpolate_path",
            data=data,
        )

        if result["value"] is None:
            return None

        return result["value"]

    def get_solution_path(self, return_type: str = "dict"):
        """
        Return the dense path.
        return_type: "dict" -> [{'pos':..., 'quat':...}, ...]
                     "mat"  -> [4x4, 4x4, ...]
        """
        result = self.client.send_request(
            function="get_solution_path",
            data={"return_type": return_type},
        )

        path = result["value"]
        if path is None:
            return None
        elif return_type == "mat":
            return [np.array(pose) for pose in path]

        return result["value"]
