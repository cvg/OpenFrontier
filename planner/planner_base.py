import numpy as np
from typing import Dict, List, Optional, Sequence, Union

from utils.planner_utils import posquat2pose

from planner.occupancy_grid import OccupancyGrid


class PlannerBase(OccupancyGrid):
    MAX_TRANS_STEP: float = 0.30  # meters per interp step
    MAX_ROT_STEP: float = 0.30  # radians per interp step
    is_pointnav_planner: bool = False

    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__(params=params)

        params = params or {}

        self.set_bounds(params.get("bounds", [-5, 5, -5, 5, -5, 5]))

        # Start/goal
        self.start_pos: Optional[np.ndarray] = None
        self.goal_pos: Optional[np.ndarray] = None
        self.start_quat: Optional[np.ndarray] = None
        self.goal_quat: Optional[np.ndarray] = None
        self.solution: List[Dict[str, np.ndarray]] = []
        self.nav_level = 0.5  # Default navigation level height

    def get_start_goal(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            "start": {"pos": self.start_pos, "quat": self.start_quat},
            "goal": {"pos": self.goal_pos, "quat": self.goal_quat},
        }

    def get_solution_path(self, return_type: str = "dict"):
        """
        Return the dense path.
        return_type: "dict" -> [{'pos':..., 'quat':...}, ...]
                     "mat"  -> [4x4, 4x4, ...]
        """
        assert return_type in {"dict", "mat"}, "return_type must be 'dict' or 'mat'"
        if return_type == "dict":
            return self.solution
        mats = [posquat2pose(sol) for sol in self.solution]
        return mats

    def get_bounds(self) -> Dict[str, float]:
        raise NotImplementedError

    def set_bounds(self, bounds_input: Sequence[float]) -> None:
        raise NotImplementedError

    def update_start_goal(
        self, start: Union[Dict, np.ndarray], goal: Union[Dict, np.ndarray]
    ) -> bool:
        """
        Accepts either dicts {"pos":..., "quat":...} or 4x4 poses.
        Attempts to nudge invalid start into nearest valid free location if needed.
        """
        raise NotImplementedError

    def get_motion_check_resolution(self) -> float:
        raise NotImplementedError

    def solve(
        self,
        time_limit: float = 5.0,
        method: str = "rrtstar",
        depth: Optional[np.ndarray] = None,  # Only for depth-aware planners
    ) -> bool:
        raise NotImplementedError

    def interpolate_path(
        self,
        num_interp_points: int = 1,
        external_waypoints: Optional[List[Sequence[float]]] = None,
    ) -> Optional[List[Dict[str, np.ndarray]]]:
        """
        Build a dense, orientation-aware path. If `external_waypoints` is given, uses those.
        Otherwise uses the solution path. Orientation is:
          - start quat at first,
          - goal quat at last,
          - for intermediates: face the next waypoint.
        """
        raise NotImplementedError
