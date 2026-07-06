import numpy as np
from typing import Dict, List, Optional, Sequence, Union
from scipy.spatial import KDTree

from utils.planner_utils import pose2posquat, posquat2pose, interpolate_waypoints
from utils.geometry import compute_alignment_transforms, pose_difference

# ompl pybind
from ompl import base as ob
from ompl import geometric as og
from planner.planner_base import PlannerBase


class OccupancyGrid3DPathPlanner(PlannerBase):
    """
    3D occupancy-grid-aware path planner using OMPL.
    """

    MAX_TRANS_STEP: float = 0.30  # meters per interp step
    MAX_ROT_STEP: float = 0.30  # radians per interp step

    def __init__(self, params: Optional[Dict] = None) -> None:
        self.sp = ob.RealVectorStateSpace(3)
        super().__init__(params=params)

        # OMPL setup
        self.ss = og.SimpleSetup(self.sp)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.sp.setup()
        self.ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)

    def get_bounds(self) -> Dict[str, float]:
        bounds = self.sp.getBounds()
        return {
            "low_x": bounds.low[0],
            "high_x": bounds.high[0],
            "low_y": bounds.low[1],
            "high_y": bounds.high[1],
            "low_z": bounds.low[2],
            "high_z": bounds.high[2],
        }

    def set_bounds(self, bounds_input: Sequence[float]) -> None:
        """
        bounds_input: [low_x, high_x, low_y, high_y, low_z, high_z]
        """
        assert (
            len(bounds_input) == 6
        ), "Bounds must be [low_x, high_x, low_y, high_y, low_z, high_z]"
        lx, hx, ly, hy, lz, hz = map(float, bounds_input)
        assert (
            lx < hx and ly < hy and lz < hz
        ), "Bounds must have low < high for each axis"

        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, lx)
        bounds.setHigh(0, hx)
        bounds.setLow(1, ly)
        bounds.setHigh(1, hy)
        bounds.setLow(2, lz)
        bounds.setHigh(2, hz)
        self.sp.setBounds(bounds)

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

        self.start_pos = np.asarray(start["pos"], dtype=float).copy()
        self.goal_pos = np.asarray(goal["pos"], dtype=float).copy()
        self.start_quat = np.asarray(start["quat"], dtype=float).copy()
        self.goal_quat = np.asarray(goal["quat"], dtype=float).copy()

        self.start_pos = self.snap_to_free(self.start_pos)
        self.goal_pos = self.snap_to_free(self.goal_pos)

        start_state = ob.State(self.sp)
        start_state()[0], start_state()[1], start_state()[2] = map(
            float, self.start_pos
        )

        goal_state = ob.State(self.sp)
        goal_state()[0], goal_state()[1], goal_state()[2] = map(float, self.goal_pos)

        # The last parameter is the goal region threshold
        self.ss.setStartAndGoalStates(start_state, goal_state, 0.1)
        return True

    def get_motion_check_resolution(self) -> float:
        return float(self.ss.getSpaceInformation().getStateValidityCheckingResolution())

    def solve(
        self,
        time_limit: float = 5.0,
        method: str = "rrtstar",
        depth: Optional[np.ndarray] = None,
    ) -> bool:
        assert time_limit > 0.0, "time_limit must be positive"
        si = self.ss.getSpaceInformation()

        if method == "rrtstar":
            planner = og.RRTstar(si)
        elif method == "rrtconnect":
            planner = og.RRTConnect(si)
        elif method == "rrt":
            planner = og.RRT(si)
        else:
            raise ValueError(f"Unknown planner '{method}'")

        self.ss.setPlanner(planner)
        solved = self.ss.solve(time_limit)
        if solved:
            self.ss.simplifySolution()
            return True
        print("Planner failed to find a solution.")
        return False

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
        try:
            if external_waypoints is None:
                path = self.ss.getSolutionPath()
                if path is None or path.getStateCount() == 0:
                    print("No OMPL path available.")
                    return None
                waypoints = [[s[0], s[1], s[2]] for s in path.getStates()]
            else:
                waypoints = [
                    np.asarray(w, dtype=float).tolist() for w in external_waypoints
                ]
                if len(waypoints) < 2:
                    print("Need at least two waypoints for interpolation.")
                    return None

            # Ensure endpoints match start/goal positions
            waypoints[0] = np.asarray(self.start_pos, dtype=float).tolist()
            waypoints[-1] = np.asarray(self.goal_pos, dtype=float).tolist()

            # Build coarse (pos, quat) sequence with
            coarse: List[Dict[str, np.ndarray]] = []
            coarse.append(
                {
                    "pos": np.asarray(self.start_pos, dtype=float),
                    "quat": np.asarray(self.start_quat, dtype=float),
                }
            )

            for i in range(1, len(waypoints) - 1):
                pos = np.asarray(waypoints[i], dtype=float)
                prev_quat = coarse[-1]["quat"]

                if i == len(waypoints) - 2:
                    # second last: reuse previous orientation
                    coarse.append({"pos": pos, "quat": prev_quat})
                    continue

                next_pos = np.asarray(waypoints[i + 1], dtype=float)
                v = next_pos - pos
                n = np.linalg.norm(v)
                if n > 1e-8:
                    v /= n
                    # Compute alignment transform for a single origin
                    t_mat = compute_alignment_transforms(
                        origins=[pos],
                        align_vec=v,
                        align_axis=[0, 0, 1],
                        appr_vec=[0, 0, -1],  # CV camera convention
                        appr_axis=[0, 1, 0],
                    )[0]
                    quat = pose2posquat(t_mat)["quat"]
                    coarse.append({"pos": pos, "quat": quat})
                else:
                    coarse.append({"pos": pos, "quat": prev_quat})

            # Append goal
            coarse.append(
                {
                    "pos": np.asarray(self.goal_pos, dtype=float),
                    "quat": np.asarray(self.goal_quat, dtype=float),
                }
            )

        except Exception as e:
            print(f"No solution found: {e}")
            return None

        # Densify via segment-wise interpolation
        dense: List[Dict[str, np.ndarray]] = []
        for i in range(len(coarse) - 1):
            a, b = coarse[i], coarse[i + 1]

            # Spacing policy based on translation and rotation magnitudes
            tdiff, rdiff = pose_difference(
                posquat2pose(a).reshape(1, 4, 4),
                posquat2pose(b).reshape(1, 4, 4),
            )
            # Extract scalar (pose_difference returns (N,M))
            td = float(np.asarray(tdiff)[0, 0])
            rd = float(np.asarray(rdiff)[0, 0])

            n_trans = int(np.ceil(td / self.MAX_TRANS_STEP))
            n_rot = int(np.ceil(rd / self.MAX_ROT_STEP))
            n = max(1, n_trans, n_rot, int(num_interp_points))

            seg = interpolate_waypoints(a, b, num_interp_points=n)
            if i == 0:
                dense.extend(seg)
            else:
                # Avoid duplicating previous endpoint
                dense.extend(seg[1:])

        self.solution = dense
        return self.solution
