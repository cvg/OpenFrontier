import json
import os
from typing import Optional, Iterable, List, Dict
import logging
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from frontier.base import Base
from frontier.frontier import Frontier
from frontier.graph import FrontierGraph
from nav.detected_object import DetectedObject
from utils.frontier_utils import ft_pos_direct_distance
from utils.geometry import compute_alignment_transforms, pose_difference
from utils.vis_utils import create_cylinder_between_points
from utils.mapping_utils import (
    render_voxel_depth,
    select_visible_points,
    compute_visible_voxels,
)
import time
from scipy.spatial.transform import Rotation as R
from planner.planner_base import PlannerBase

np.set_printoptions(precision=3, suppress=True)


class FrontierManager(Base):

    def __init__(
        self, params: Optional[dict] = None, log_level: int = logging.INFO, planner=None
    ) -> None:
        """
        params keys (optional):
        - filter_bbox: list[6] or None
        - filter_max_vd_z: float or None
        - filter_min_gain: float
        - visited_trans_threshold: float
        - visited_angle_threshold: float
        - visited_gain_reduction_factor: float
        - render_K: (3,3) intrinsics
        - render_H: int
        - render_W: int
        - render_depth_range: float
        - voxel_size: float
        - render_decrease_factor: float
        - force_all_frontier_to_xy: bool
        - utility_gain_factor: float
        - max_planning_time: float
        - planning_algo: str
        - (some planner-related keys are forwarded to PathPlanner)
        """
        super().__init__(params, log_level=log_level)
        self.logger.info("FrontierManager initialized")

        p = params or {}

        # frontiers and robot poses
        self.frontiers: Dict[int, Frontier] = {}
        self.robot_poses: Dict[int, np.ndarray] = {}

        # Graph/Maps
        self.graph = FrontierGraph(p, log_level=log_level)
        self.occ_map: Optional[np.ndarray] = None
        self.free_map: Optional[np.ndarray] = None

        # Planner
        self.planner: PlannerBase = planner
        self.current_goal_pose: Optional[np.ndarray] = None
        self.current_goal_ft_id: Optional[int] = None
        self._unreachable_positions = []

        # IDs
        self._current_frontier_id: int = self.graph.type_range["F"][0]
        self._current_robot_id: int = self.graph.type_range["R"][0]

        # Parameters
        self.filter_bbox = p.get("filter_bbox", None)
        self.filter_max_vd_z = p.get("filter_max_vd_z", None)
        self.filter_min_gain = float(p.get("filter_min_gain", 0.1))
        self._initial_min_gain = self.filter_min_gain
        self.max_planning_time = float(p.get("max_planning_time", 1.5))
        self.planning_algo = p.get("planning_algo", "rrtstar")

        self.v_tras_thre = float(p.get("visited_trans_threshold", 0.4))
        self.v_angl_thre = float(p.get("visited_angle_threshold", 0.4))
        self.v_gain_reduction_factor = float(
            p.get("visited_gain_reduction_factor", 1000)
        )

        # Rendering
        if p.get("render_K") is not None:
            self.render_K = np.array(p["render_K"], dtype=np.float32).reshape(3, 3)
        else:
            self.render_K = np.array(
                [[60, 0, 64], [0, 60, 64], [0, 0, 1]], dtype=np.float32
            )

        self.render_H = int(p.get("render_H", 128))
        self.render_W = int(p.get("render_W", 128))
        self.render_d_range = float(p.get("render_depth_range", 3.5))
        self.voxel_size = float(p.get("voxel_size", 0.1))
        self.render_decrease_factor = float(p.get("render_decrease_factor", 1.0))

        self.force_all_frontier_to_xy = bool(p.get("force_all_frontier_to_xy", False))
        self.utility_g_factor = float(p.get("utility_gain_factor", 1.0))
        self.seeking_weight = float(p.get("seeking_weight", 1.0))
        self.prob_sharpness = float(p.get("prob_sharpness", 1.0))

        self.object_lockin: DetectedObject = None

        self.goal_frontier = None

        # Validation
        if self.filter_bbox is not None:
            if not (
                isinstance(self.filter_bbox, (list, tuple))
                and len(self.filter_bbox) == 6
            ):
                raise ValueError(
                    "filter_bbox must be a list/tuple of 6 numbers [xmin,xmax,ymin,ymax,zmin,zmax]."
                )
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive.")
        if self.render_H <= 0 or self.render_W <= 0:
            raise ValueError("render_H and render_W must be positive.")
        if self.render_d_range <= 0:
            raise ValueError("render_depth_range must be positive.")

        self._current_emergency_rotation = 0
        self._max_emergency_rotations = 3

        self.logging_file = "navigation_log.txt"

    def _alloc_frontier_id(self) -> int:
        """
        Return a fresh frontier ID. Skips over any IDs that might
        already be in use (e.g., after deletions).
        """
        nid = self._current_frontier_id
        # Advance counter for next time
        self._current_frontier_id += 1
        # If this nid is somehow in use, keep advancing
        while nid in self.frontiers:
            nid = self._current_frontier_id
            self._current_frontier_id += 1
        return nid

    def _alloc_robot_id(self) -> int:
        """
        Return a fresh robot ID. Skips over any IDs that might
        already be in use (e.g., after deletions).
        """
        rid = self._current_robot_id
        # Advance counter for next time
        self._current_robot_id += 1
        # If this rid is somehow in use, keep advancing
        while rid in self.robot_poses:
            rid = self._current_robot_id
            self._current_robot_id += 1
        return rid

    @property
    def all_frontiers(self) -> List[Frontier]:
        return list(self.frontiers.values())

    @property
    def all_frontiers_ids(self) -> List[int]:
        return list(self.frontiers.keys())

    @property
    def valid_frontiers(self) -> List[Frontier]:
        return [ft for ft in self.frontiers.values() if ft.is_valid]

    @property
    def utility(self) -> List[float]:
        return [ft.utility for ft in self.valid_frontiers]

    @property
    def current_robot_id(self) -> Optional[int]:
        """
        Latest existing robot ID (None if no robot poses yet).
        Previously this returned the *next* ID; this is safer.
        """
        if self.robot_poses:
            return max(self.robot_poses.keys())
        return None

    @property
    def goal_pose(self) -> np.ndarray:
        return self.current_goal_pose

    def add_robot_poses(self, robot_poses: list[np.ndarray]):
        """
        Add a list of robot poses to the manager.
        Args:
            robot_poses: List of robot poses, each pose is a 4x4 matrix.

        Returns:
            List of added robot IDs.
        """
        if len(robot_poses) == 0:
            self.logger.debug("No robot poses to add.")
            return []

        added_robot_ids = []
        for pose in robot_poses:
            assert pose.shape == (4, 4), "Each robot pose must be a 4x4 matrix."
            robot_id = self._alloc_robot_id()  # Assign a unique ID
            self.robot_poses[robot_id] = pose
            # add the robot node to the graph
            self.graph.add_node_R(robot_id)
            added_robot_ids.append(robot_id)
            # link the node with the latest robot pose (as the robot ID increments, the latest pose is always the last one added)
            if robot_id - 1 in self.robot_poses:  # avoid the first robot pose
                prev_pose = self.robot_poses[robot_id - 1]
                distance = np.linalg.norm(
                    pose[:3, 3] - prev_pose[:3, 3]
                )  # Euclidean distance in 3D space
                self.graph.add_edge_RR(robot_id, robot_id - 1, weight=distance)

        return added_robot_ids  # Return the list of added robot IDs

    def get_frontier(
        self, frontier_id: int, *, required: bool = False
    ) -> Optional[Frontier]:
        """
        Fetch a frontier by ID.

        Args:
            frontier_id: Frontier ID to fetch.
            required: If True, raise KeyError when missing; otherwise return None.

        Returns:
            Frontier or None.
        """
        ft = self.frontiers.get(frontier_id)
        if ft is None:
            msg = f"Frontier ID {frontier_id} not found."
            if required:
                raise KeyError(msg)
            self.logger.debug(msg)
        return ft

    def get_robot_pose(
        self, robot_id: int, *, required: bool = False
    ) -> Optional[np.ndarray]:
        """
        Fetch a robot pose (4x4) by ID.

        Args:
            robot_id: Robot ID to fetch.
            required: If True, raise KeyError when missing; otherwise return None.

        Returns:
            np.ndarray shape (4,4) or None.
        """
        pose = self.robot_poses.get(robot_id)
        if pose is None:
            msg = f"Robot pose ID {robot_id} not found."
            if required:
                raise KeyError(msg)
            self.logger.debug(msg)
        return pose

    @staticmethod
    def get_frontier_pose(frontier) -> np.ndarray:
        """
        Build a 4x4 pose for a frontier by aligning camera +Z to the frontier's
        view_direction and placing it at pos3d. Uses CV camera convention:
        +Z forward, +Y down (approx), +X right.

        Returns:
            (4,4) float ndarray
        """
        pos = np.asarray(frontier.pos3d, dtype=float).reshape(3)
        vd = np.asarray(frontier.view_direction, dtype=float).reshape(3)

        T = compute_alignment_transforms(
            origins=[pos],
            align_vec=vd,
            align_axis=[0, 0, 1],
            appr_vec=[0, 0, -1],  # CV convention: Y down approx
            appr_axis=[0, 1, 0],
        )[0]

        # Ensure exact translation (in case the util modifies it)
        T = np.asarray(T, dtype=float)
        T[:3, 3] = pos
        return T

    def add_frontiers(
        self, frontiers: List[Frontier], parent_ids: Optional[Iterable[int]] = None
    ) -> List[int]:
        """
        Add a list of frontiers. Optionally connect each to given robot parent IDs.

        Args:
            frontiers: Frontier objects to add.
            parent_ids: Robot pose IDs to connect (edges F-R). Duplicates ignored.
        """
        if len(frontiers) == 0:
            self.logger.debug("No frontiers to add.")
            return
        else:
            self.logger.debug(f"Adding {len(frontiers)} frontiers.")

        parent_ids = list(parent_ids) if parent_ids is not None else []

        for frontier in frontiers:
            # if force_all_frontier_to_xy is True, set the 3d vector to be in the XY plane
            if self.force_all_frontier_to_xy:
                vd = np.array(
                    [frontier.view_direction[0], frontier.view_direction[1], 0.0],
                    dtype=float,
                )
                n = np.linalg.norm(vd)
                frontier.view_direction = vd / n if n > 1e-8 else vd
            frontier.id = self._alloc_frontier_id()  # Assign a unique ID
            self.frontiers[frontier.id] = frontier  # Use ID as key for fast access
            # add the frontier node to the graph
            self.graph.add_node_F(frontier.id)
            # set the frontier's parent IDs
            frontier.parent_ids = []
            # assign 6D pose

            if not frontier.is_object:  # Objects already have pose6d assigned
                frontier.pose6d = self.get_frontier_pose(frontier)

            for parent_id in parent_ids:
                assert (
                    parent_id in self.robot_poses
                ), f"Parent robot ID {parent_id} does not exist in robot poses."
                # calculate the distance between frontier position and the robot position
                distance = np.linalg.norm(
                    frontier.pos3d - self.robot_poses[parent_id][:3, 3]
                )
                self.graph.add_edge_FR(frontier.id, parent_id, weight=distance)
                frontier.parent_ids.append(parent_id)  # Add parent ID to the frontier

    def add_objects(
        self, objects: List[DetectedObject], parent_ids: Optional[Iterable[int]] = None
    ) -> List[int]:
        """
        Add a list of object frontiers. Optionally connect each to given robot parent IDs.

        Args:
            frontiers: Frontier objects to add.
            parent_ids: Robot pose IDs to connect (edges F-R). Duplicates ignored.
        """
        if len(objects) == 0:
            self.logger.debug("No object frontiers to add.")
            return
        else:
            self.logger.debug(f"Adding {len(objects)} object frontiers.")

        parent_ids = list(parent_ids) if parent_ids is not None else []

        frontiers = []

        for obj in objects:
            if obj.is_valid is False:
                continue

            frontier = Frontier()
            frontier.gain = 1e20
            frontier.u_gain = 1e20
            frontier.probability = 1.0
            frontier.justification = "Detected Object"
            frontier.label = obj.label
            viewpoint = self.find_object_viewpoint(
                obj.viewpoint, obj.centroid, separation=0.7
            )
            frontier.pos3d = viewpoint[:3, 3]
            frontier.pose6d = viewpoint
            frontier.is_object = True
            frontier.linked_object = obj

            frontier.view_direction = viewpoint[:3, 2]  # +Z axis
            frontier.set_valid()

            obj.frontier = frontier

            frontiers.append(frontier)

        return self.add_frontiers(frontiers, parent_ids=parent_ids)

    def find_object_viewpoint(
        self,
        camera_pose: np.ndarray,
        object_centroid: np.ndarray,
        separation: float = 0.5,
    ) -> np.ndarray:
        """
        Draws a line between the pose of the camera and the centroid of the object, selects a point at a certain distance and creates a viewpoint that looks at the object centroid from that point, with the viewing rotation aligned with the camera's up direction.
        """
        cam_pos = camera_pose[:3, 3]
        obj_pos = object_centroid

        direction = obj_pos - cam_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            direction = camera_pose[:3, 2]  # Camera's +Z axis
            direction_norm = np.linalg.norm(direction)

        direction = direction / direction_norm

        viewpoint_pos = obj_pos - direction * separation

        viewpoint_pos = self.get_object_free_point(camera_pose, viewpoint_pos[:3])

        direction_to_obj = obj_pos - viewpoint_pos
        direction_to_obj = direction_to_obj / np.linalg.norm(direction_to_obj)
        up_vector = camera_pose[:3, 1]  # Camera's +Y axis
        right_vector = np.cross(up_vector, direction_to_obj)
        right_vector = right_vector / np.linalg.norm(right_vector)
        up_vector = np.cross(direction_to_obj, right_vector)
        viewpoint_pose = np.eye(4)
        viewpoint_pose[:3, 0] = right_vector
        viewpoint_pose[:3, 1] = up_vector
        viewpoint_pose[:3, 2] = direction_to_obj
        viewpoint_pose[:3, 3] = viewpoint_pos

        return viewpoint_pose

    def move_outside_occupied_space(self, position: np.ndarray) -> np.ndarray:
        if self.occ_map is None or self.free_map is None or len(self.free_map) == 0:
            return position

        buffer = self.planner.min_dist2occ * 1.5
        free_pts = self.free_map
        dists = np.linalg.norm(free_pts - position.reshape(1, 3), axis=1)
        closest_free_pt = free_pts[np.argmin(dists)]

        direction = closest_free_pt - position
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return position
        direction /= norm

        step = 0.1  # adjust for map resolution
        new_position = position.copy()

        iters = 0

        while self.planner.isoccupied(new_position) and iters < 100:
            new_position += direction * step
            norm -= step

        new_position += direction * buffer  # ensure outside
        return new_position

    def get_object_free_point(self, camera_pose, object_position):
        """
        Get the point that is closest to the object along the camera-object line that is not occupied.
        """
        camera_pos = camera_pose[:3, 3]
        object_pos = object_position

        # Compute the direction from the camera to the object
        direction = object_pos - camera_pos
        direction /= np.linalg.norm(direction)

        # Find the point closest to the object along the line that is not occupied
        step_size = 0.1  # step size along the line
        max_distance = np.linalg.norm(object_pos - camera_pos)
        num_steps = int(max_distance / step_size)
        for step in range(num_steps + 1):
            point = object_pos - direction * step * step_size
            if not self.planner.isoccupied(point):
                return point
        return camera_pos  # fallback to camera position if all points are occupied

    def remove_frontiers(self, frontier_ids, planning_failure=False):
        """
        Remove frontiers by their IDs.
        Args:
            frontier_ids: List of frontier IDs to remove.
        """
        if not isinstance(frontier_ids, list):
            raise ValueError("frontier_ids must be a list.")

        # Deduplicate while preserving order
        seen = set()
        ids = [fid for fid in frontier_ids if not (fid in seen or seen.add(fid))]

        if not ids:
            self.logger.debug("No frontier IDs provided to remove.")
            return

        removed_any = False
        for fid in ids:
            if fid in self.frontiers:
                frontier = self.frontiers[fid]
                if frontier.is_object and frontier.linked_object is not None:
                    if planning_failure:
                        frontier.linked_object.is_valid = False
                        if frontier.linked_object.verification_status == "unverified":
                            frontier.linked_object.verification_status = "no_frontier"
                del self.frontiers[fid]
                try:
                    self.graph.remove_node_F(fid)
                except Exception as e:
                    self.log(
                        "error",
                        self.logging_file,
                        f"Graph removal failed for frontier {fid}: {e}",
                    )
                removed_any = True

        # Clear goal if it was removed
        if (
            removed_any
            and self.current_goal_ft_id is not None
            and self.current_goal_ft_id not in self.frontiers
        ):
            self.current_goal_ft_id = None
            self.current_goal_pose = None

        if removed_any:
            self.logger.debug(f"Removed frontiers: {ids}")
        else:
            self.logger.debug("No matching frontiers to remove.")

    def remove_invalid_frontiers(self) -> None:
        """
        Remove all frontiers currently marked invalid.
        """
        # Build a stable list before mutating self.frontiers
        invalid_ids = [
            fid for fid, ft in list(self.frontiers.items()) if not ft.is_valid
        ]

        if not invalid_ids:
            self.logger.debug("No invalid frontiers to remove.")
            return

        self.remove_frontiers(invalid_ids)
        self.logger.debug(f"Removed {len(invalid_ids)} invalid frontiers.")

    def merge_frontiers(self):
        dbscan_params = {
            "eps": 1.8,
            "min_samples": 1,
            "weights": [1, 2],
        }

        valid_frontiers = [
            ft
            for ft in list(self.valid_frontiers)
            if not ft.is_object and ft.justification != "Rotation Frontier"
        ]  # exclude object frontiers and rotation frontiers

        _n_valid_before = len(valid_frontiers)
        if _n_valid_before < 2:
            self.logger.debug("Not enough valid frontiers to merge.")
            return

        positions = np.array([ft.pos3d for ft in valid_frontiers])
        directions = np.array([ft.view_direction for ft in valid_frontiers])
        ft_features = np.concatenate((positions, directions), axis=1)

        metric_func = lambda x, y: ft_pos_direct_distance(
            x, y, weights=dbscan_params["weights"]
        )
        dbscan = DBSCAN(
            eps=dbscan_params["eps"],
            min_samples=dbscan_params["min_samples"],
            metric=metric_func,
            n_jobs=-1,
        )
        labels = dbscan.fit_predict(ft_features)
        _n_valid_before = len(valid_frontiers)

        # Prepare merges
        to_remove = []
        to_add = []

        for cls_id in np.unique(labels):
            if cls_id == -1:
                continue
            _fts = [ft for i, ft in enumerate(valid_frontiers) if labels[i] == cls_id]
            if len(_fts) <= 1:
                continue
            else:
                # If one of the merged frontiers is the current goal, keep it
                merged_into_goal = False
                for ft in _fts:
                    if ft.id == self.current_goal_ft_id:
                        to_remove.extend([f.id for f in _fts if f.id != ft.id])
                        merged_into_goal = True
                        self.log(
                            "info",
                            self.logging_file,
                            f"Merged frontiers into current goal {ft.id}.",
                        )
                        break

                if not merged_into_goal:
                    _ft = Frontier()
                    _gains = np.array([max(ft.u_gain, 1e-4) for ft in _fts])
                    _probabilities = np.array([ft.probability for ft in _fts])
                    _ft.pos3d = np.average(
                        [ft.pos3d for ft in _fts], axis=0, weights=_gains
                    )
                    cls_vd = np.sum(
                        np.array([ft.view_direction for ft in _fts]), axis=0
                    )
                    _ft.view_direction = cls_vd / np.linalg.norm(cls_vd) + 1e-6
                    _ft.direct_angle = np.average(
                        [ft.direct_angle for ft in _fts], axis=0
                    )
                    _ft.pixel_pos = np.average([ft.pixel_pos for ft in _fts], axis=0)
                    _ft.set_valid()
                    _ft.u_gain = _gains.mean()
                    _ft.label = _fts[0].label  # keep the first label
                    _ft.justification = _fts[0].justification or "Merged Frontier"
                    _ft.probability = _probabilities.mean()
                    _ft.gain = _gains.mean()
                    _ft.id = self._alloc_frontier_id()
                    _ft.parent_ids = list(
                        set([pid for ft in _fts for pid in ft.parent_ids])
                    )
                    to_add.append((_ft, _ft.parent_ids))
                    to_remove.extend([ft.id for ft in _fts])

        # Remove and add after loop
        self.remove_frontiers(to_remove)
        for _ft, _parent_ids in to_add:
            self.add_frontiers([_ft], parent_ids=_parent_ids)

        _n_valid_after = len(self.valid_frontiers)
        self.logger.debug(
            f"Merged frontiers from {_n_valid_before} to {_n_valid_after} valid frontiers."
        )

        self.merge_object_frontiers()

    def merge_object_frontiers(self) -> None:
        dbscan_params = {
            "eps": 1.8,
            "min_samples": 1,
            "weights": [1, 2],
        }

        valid_frontiers = [
            ft for ft in list(self.valid_frontiers) if ft.is_object
        ]  # include only object frontiers
        _n_valid_before = len(valid_frontiers)
        if _n_valid_before < 2:
            self.logger.debug("Not enough valid frontiers to merge.")
            return

        positions = np.array([ft.pos3d for ft in valid_frontiers])
        directions = np.array([ft.view_direction for ft in valid_frontiers])
        ft_features = np.concatenate((positions, directions), axis=1)

        metric_func = lambda x, y: ft_pos_direct_distance(
            x, y, weights=dbscan_params["weights"]
        )
        dbscan = DBSCAN(
            eps=dbscan_params["eps"],
            min_samples=dbscan_params["min_samples"],
            metric=metric_func,
            n_jobs=-1,
        )
        labels = dbscan.fit_predict(ft_features)
        _n_valid_before = len(valid_frontiers)

        # Prepare merges
        to_remove = []
        to_add = []

        for cls_id in np.unique(labels):
            if cls_id == -1:
                continue
            _fts = [ft for i, ft in enumerate(valid_frontiers) if labels[i] == cls_id]
            if len(_fts) <= 1:
                continue
            else:

                # If one of the merged frontiers is the current goal, keep it
                merged_into_goal = False
                for ft in _fts:
                    if ft.id == self.current_goal_ft_id:
                        to_remove.extend([f.id for f in _fts if f.id != ft.id])
                        merged_into_goal = True
                        self.log(
                            "info",
                            self.logging_file,
                            f"Merged frontiers into current goal {ft.id}.",
                        )
                        _ft = ft  # keep the current goal frontier
                        break

                if not merged_into_goal:
                    _ft = Frontier()
                    _gains = np.array([max(ft.u_gain, 1e-4) for ft in _fts])
                    _ft.pos3d = np.average(
                        [ft.pos3d for ft in _fts], axis=0, weights=_gains
                    )
                    cls_vd = np.sum(
                        np.array([ft.view_direction for ft in _fts]), axis=0
                    )
                    _ft.view_direction = cls_vd / np.linalg.norm(cls_vd) + 1e-6
                    _ft.set_valid()
                    _ft.is_object = True
                    _ft.linked_object = _fts[
                        0
                    ].linked_object  # keep the first object's info
                    _ft.u_gain = _gains.mean()
                    _ft.gain = _gains.mean()
                    _ft.id = self._alloc_frontier_id()
                    _ft.parent_ids = list(
                        set([pid for ft in _fts for pid in ft.parent_ids])
                    )
                    to_remove.extend([ft.id for ft in _fts])
                    to_add.append((_ft, _ft.parent_ids))

                for ft in _fts:
                    if ft.linked_object is not _ft.linked_object:
                        ft.linked_object.is_valid = (
                            False  # mark merged objects as invalid
                        )
                        ft.linked_object.frontier = (
                            _ft  # link to the new merged frontier
                        )

        # Remove and add after loop
        self.remove_frontiers(to_remove)
        for _ft, _parent_ids in to_add:
            _ft.pos3d = self.move_outside_occupied_space(_ft.pos3d)
            _ft.pose6d = self.get_frontier_pose(_ft)
            self.add_frontiers([_ft], parent_ids=_parent_ids)

        _n_valid_after = len(self.valid_frontiers)
        self.logger.debug(
            f"Merged frontiers from {_n_valid_before} to {_n_valid_after} valid frontiers."
        )

    def update_map(self, free_map=None, occ_map=None) -> None:
        """
        Update the planner's space with the occupancy (occ_map) and free space (free_map).
        At least one must be provided. No shape/dtype checks here by design.
        """
        if free_map is None and occ_map is None:
            raise ValueError("At least one of free_map or occ_map must be provided.")

        if free_map is not None:
            self.free_map = free_map
        if occ_map is not None:
            self.occ_map = occ_map

        # Only build KDTree if provided and non-empty
        free_for_kdt = (
            free_map if (free_map is not None and len(free_map) > 0) else None
        )
        occ_for_kdt = occ_map if (occ_map is not None and len(occ_map) > 0) else None
        self.planner.update_space(free_vx=free_for_kdt, occ_vx=occ_for_kdt)

        self.logger.debug(
            "Maps updated: free=%s, occ=%s",
            None if free_map is None else len(free_map),
            None if occ_map is None else len(occ_map),
        )

    def select_goal_frontier_id(self) -> int:
        """
        Select the frontier with the highest utility among valid frontiers.
        Sets and returns self.current_goal_ft_id. Returns None if none available.
        """
        fts = self.valid_frontiers
        if not fts:
            self.logger.debug("No valid frontiers available to select as goal.")
            self.current_goal_ft_id = None
            return None

        # Keep only finite utilities
        candidates = [
            (ft.id, float(ft.utility))
            for ft in fts
            if np.isfinite(getattr(ft, "utility", np.nan))
        ]
        if not candidates:
            self.logger.debug("No finite utilities available for goal selection.")
            self.current_goal_ft_id = None
            return None

        # Tie-break: utility desc, then u_gain desc (if present), then smaller ID
        def key_fn(fid_util):
            fid, util = fid_util
            ug = getattr(self.frontiers.get(fid), "u_gain", -np.inf)
            return (util, ug, -fid)

        fid, util = max(candidates, key=key_fn)
        self.current_goal_ft_id = fid
        self.logger.debug(f"Selected goal frontier ID: {fid} (utility={util:.6f}).")
        return fid

    def gain_adjustment(self, w_history_path: bool = True, w_map: bool = True) -> None:
        """
        Adjust each valid frontier's u_gain based on:
        1) proximity to past robot poses (history), and
        2) optional: visible free voxels from the frontier's camera pose (map).
        Keeps ft.gain as the base and writes ft.u_gain.
        """
        fts = self.valid_frontiers
        n = len(fts)
        if n == 0:
            self.logger.debug("No valid frontiers to adjust gains.")
            return

        # -------------------- ADJUST 1: history proximity --------------------
        if w_history_path:
            robot_ids = self.graph.get_node_R()
            if robot_ids:
                ft_W_T_C = np.stack(
                    [ft.pose6d for ft in fts], axis=0
                )  # (N,4,4)  W_T_C (C→W)
                robot_W_T_R = np.stack(
                    [self.robot_poses[rid] for rid in robot_ids], axis=0
                )  # (M,4,4)

                trans_diff, rot_diff = pose_difference(
                    ft_W_T_C, robot_W_T_R
                )  # (N,M), (N,M)
                close_mask = (trans_diff < self.v_tras_thre) & (
                    rot_diff < self.v_angl_thre
                )
                n_close = close_mask.sum(axis=1).astype(np.float32)  # (N,)
                reduction_1 = self.v_gain_reduction_factor * n_close
            else:
                reduction_1 = np.zeros(n, dtype=np.float32)
        else:
            reduction_1 = np.zeros(n, dtype=np.float32)

        # -------------------- ADJUST 2: map-based visibility --------------------
        voxel_vol = float(self.voxel_size**3)
        reduction_2 = np.zeros(n, dtype=np.float32)

        if w_map and (self.occ_map is not None) and (self.free_map is not None):
            # Frontier pose is W_T_C; for rendering we need C_T_W (extrinsics)
            C_T_W_batch = np.linalg.inv(
                np.stack([ft.pose6d for ft in fts], axis=0)
            )  # (N,4,4)  C_T_W (W→C)

            depths = render_voxel_depth(
                occ_pts=self.occ_map,
                camera_extrinsics=C_T_W_batch,
                render_params={
                    "K": self.render_K,
                    "H": self.render_H,
                    "W": self.render_W,
                    "radius": self.voxel_size / 2,
                },
            )

            for i, (ft, C_T_W, depth) in enumerate(zip(fts, C_T_W_batch, depths)):
                if ft.is_object:
                    continue  # skip objects

                # Count free voxels visible from this frontier view (respecting depth)
                vox_in_fov, _ = select_visible_points(
                    self.free_map,
                    K=self.render_K,
                    T=C_T_W,  # world→camera
                    img_size=(self.render_H, self.render_W),
                    max_depth=self.render_d_range,
                    min_depth_map=depth,
                )

                reduction_2[i] = (
                    float(vox_in_fov.shape[0])
                    * voxel_vol
                    * float(self.render_decrease_factor)
                )

                # Optional refinement: clamp base gain by max possible visible volume from this view
                if (ft.gain is not None) and not np.all(depth < 1e-6):
                    _, max_n_vis = compute_visible_voxels(
                        K=self.render_K,
                        T=C_T_W,  # world→camera
                        img_size=(self.render_H, self.render_W),
                        voxel_size=self.voxel_size,
                        max_depth=self.render_d_range,
                        min_depth_map=depth,
                    )
                    ft.gain = min(float(max_n_vis) * voxel_vol, float(ft.gain))

        self.logger.debug(
            "Before gain adjustment: " + ", ".join(f"{ft.id}: {ft.gain}" for ft in fts)
        )

        for ft, r1, r2 in zip(fts, reduction_1, reduction_2):
            if ft.is_object:
                continue

            base = float(ft.gain) if ft.gain is not None else 1e-4
            ft.u_gain = max(base - r1 - r2, 1e-4)

        self.logger.debug(
            "After gain adjustment: " + ", ".join(f"{ft.id}: {ft.u_gain}" for ft in fts)
        )

    def update_utility(self, current_pos) -> None:
        """
        Update each valid frontier's utility (Eq.(8) in the paper):
            utility = (u_gain ** utility_g_factor) / max(distance, 1e-6)
        """
        p = np.asarray(current_pos, dtype=float).reshape(3)

        valid = self.valid_frontiers
        for ft in valid:

            # Distance to current position (avoid divide-by-zero)
            d = float(np.linalg.norm(np.asarray(ft.pos3d, dtype=float) - p))
            denom = max(d, 1e-6)

            # Prioritize objects
            if ft.is_object:
                ft.utility = 1e20 / denom
                continue

            # Base gain: prefer u_gain, fall back to gain; clamp tiny positive
            base_gain = getattr(ft, "u_gain", None)
            if base_gain is None:
                base_gain = getattr(ft, "gain", 0.0)
            base_gain = float(max(base_gain, 1e-4))

            if ft.probability is None:
                ft.probability = 0.5  # default if missing

            scaled_prob = ft.probability**self.prob_sharpness

            exploration_gain = (1 - self.seeking_weight) * base_gain

            seeking_gain = self.seeking_weight * scaled_prob * base_gain

            blended_gain = exploration_gain + seeking_gain

            # TODO Ablation for this
            ft.utility = (blended_gain ** float(self.utility_g_factor)) / denom

        self.logger.debug(f"Utility updated for {len(valid)} valid frontiers.")

    def filter_frontiers(self) -> None:
        """
        Mark frontiers invalid based on:
        1) bounding box (if set)
        2) min gain threshold
        3) view-direction z limit (if set)
        4) proximity to occupied space (if occ_map present)
        Then remove all invalid frontiers.
        """
        if not self.frontiers:
            self.logger.debug("No frontiers to filter.")
            return

        bbox = self.filter_bbox
        min_gain = float(self.filter_min_gain)
        max_vd_z = self.filter_max_vd_z
        check_occ = self.occ_map is not None

        for fid, ft in list(self.frontiers.items()):
            if not ft.is_valid:
                continue  # already invalid elsewhere

            # Skip object frontiers for all filters except occupancy
            if ft.is_object:
                """if check_occ and self.planner.isoccupied(ft.pos3d):
                ft.set_invalid()
                self.logger.debug(f"Frontier {fid} invalid (near occupied).")"""
                continue

            # 1) Bounding box filter
            if bbox is not None:
                x, y, z = map(float, ft.pos3d)
                if not (
                    bbox[0] <= x <= bbox[1]
                    and bbox[2] <= y <= bbox[3]
                    and bbox[4] <= z <= bbox[5]
                    and z < self.planner.nav_level + 2.0
                ):
                    ft.set_invalid()
                    self.logger.debug(f"Frontier {fid} invalid (bbox).")
                    continue

            # 2) Gain filter
            g = ft.u_gain if ft.u_gain is not None else ft.gain
            g = float(g if g is not None else 0.0)
            if g < min_gain:
                ft.set_invalid()
                self.logger.debug(f"Frontier {fid} invalid (gain<{min_gain}).")
                continue

            # 3) View-direction z filter
            if max_vd_z is not None:
                vd_z = float(ft.view_direction[2])
                if abs(vd_z) > max_vd_z:
                    ft.set_invalid()
                    self.logger.debug(f"Frontier {fid} invalid (|vd_z|>{max_vd_z}).")
                    continue

            # 4) Too close to occupied space
            if check_occ and self.planner.isoccupied(ft.pos3d):
                ft.set_invalid()
                self.logger.debug(f"Frontier {fid} invalid (near occupied).")
                continue

            # 5) Filter if too close to unreachable positions
            for pos in self._unreachable_positions:
                dist = np.linalg.norm(np.asarray(ft.pos3d) - np.asarray(pos))
                if dist < 0.1:
                    ft.set_invalid()
                    self.logger.debug(
                        f"Frontier {fid} invalid (too close to unreachable position)."
                    )
                    break

        # Purge all marked
        self.remove_invalid_frontiers()

    def next_emergency_rotation(self) -> bool:
        self._current_emergency_rotation += 1
        self.filter_min_gain /= 2.0  # relax gain threshold
        if self._current_emergency_rotation > self._max_emergency_rotations:
            return False
        return True

    def reset_emergency_rotation(self) -> None:
        self._current_emergency_rotation = 0
        self.filter_min_gain = self._initial_min_gain

    def add_rotation_frontiers(self, W_T_C: np.ndarray) -> None:
        """
        Adds frontiers to the sides and behind to encourage rotation
        """
        current_pos = W_T_C[:3, 3]

        angles = [-90, 90, 180]  # left, right, behind
        frontiers = []
        for angle in angles:
            R_curr = W_T_C[:3, :3]
            R_delta = R.from_euler("Z", angle, degrees=True).as_matrix()
            R_new = R_delta @ R_curr

            ft_pose = np.eye(4)
            ft_pose[:3, :3] = R_new
            ft_pose[:3, 3] = current_pos

            ft = Frontier()
            ft.pos3d = ft_pose[:3, 3]
            ft.view_direction = ft_pose[:3, 2]  # +Z axis
            ft.pose6d = ft_pose
            ft.gain = self.filter_min_gain * 2.0  # ensure passes gain filter
            ft.justification = "Rotation Frontier"
            ft.set_valid()
            frontiers.append(ft)

        new_ids = self.add_robot_poses([W_T_C])
        self.add_frontiers(frontiers, parent_ids=new_ids)
        self.gain_adjustment()
        self.filter_frontiers()
        self.update_utility(current_pos)

    def get_waypoints_from_graph(self, ft_id: int, robot_pose_id: int) -> np.ndarray:
        """
        Compute a final approach pose W_T_C for the target frontier, following the graph path
        from the given robot pose. Densifies with ~0.1 m spacing, finds the last reachable
        free-space point, and orients toward the next point (or the frontier if none).
        """
        # --- Validate inputs and fetch frontier ---
        assert ft_id in self.frontiers, f"Frontier ID {ft_id} not found."
        frontier = self.frontiers[ft_id]  # has .pos3d, .view_direction, .pose6d (W_T_C)

        # --- Shortest path (node ids R…→F) ---
        path, _ = self.graph.get_shortest_R_to_F(
            robot_id=robot_pose_id, frontier_id=ft_id
        )
        if not path:
            # No route; fall back to frontier’s own pose
            return frontier.pose6d

        # --- Coarse waypoints: robot node positions, then frontier position ---
        coarse_pts = []
        for node_id in path[:-1]:
            W_T_R = self.get_robot_pose(node_id)
            if W_T_R is None:
                raise KeyError(f"Robot pose for node {node_id} not found.")
            coarse_pts.append(W_T_R[:3, 3])
        coarse_pts.append(frontier.pos3d)

        # --- Densify path at ~0.1 m (no duplicate endpoints between segments) ---
        STEP = 0.1
        MAX_PER_SEG = 500
        fine_pts = []
        prev = None
        for cur in coarse_pts:
            cur = np.asarray(cur, dtype=float)
            if prev is None:
                prev = cur
                continue
            seg = cur - prev
            dist = float(np.linalg.norm(seg))
            if dist <= 1e-6:
                fine_pts.append(cur)
                prev = cur
                continue
            n = int(np.ceil(dist / STEP))
            n = min(max(n, 1), MAX_PER_SEG)
            # sample t in (0,1], i.e., exclude prev to avoid duplicates, include cur
            ts = np.linspace(1.0 / n, 1.0, n)
            pts = prev + ts[:, None] * seg
            fine_pts.extend(pts)
            prev = cur

        if not fine_pts:
            # No densification possible; use the frontier pose
            return frontier.pose6d

        # --- Reverse search: last reachable point (free and not occupied) ---
        break_idx = None
        for idx in range(len(fine_pts) - 1, -1, -1):
            pt = fine_pts[idx]
            if self.planner.isfree(pt) and not self.planner.isoccupied(pt):
                break_idx = idx
                break

        if break_idx is None:
            # Nothing along the path is free → go with the frontier pose
            return frontier.pose6d

        break_pt = np.asarray(fine_pts[break_idx], dtype=float)

        # Close enough to the frontier? keep its orientation, place at break_pt
        if np.linalg.norm(break_pt - frontier.pos3d) < self.v_tras_thre:
            W_T_C = frontier.pose6d.copy()
            W_T_C[:3, 3] = break_pt
            return W_T_C

        # --- Orientation at the break point ---
        # Prefer direction toward the next waypoint if there is one; otherwise toward the frontier.
        if break_idx < len(fine_pts) - 1:
            target = np.asarray(fine_pts[break_idx + 1], dtype=float)
        else:
            target = np.asarray(frontier.pos3d, dtype=float)

        direction = target - break_pt
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            # Degenerate → fall back to the frontier's view direction, or +Z
            vd = np.asarray(frontier.view_direction, dtype=float)
            vd_norm = float(np.linalg.norm(vd))
            direction = (
                (vd / vd_norm)
                if vd_norm > 1e-8
                else np.array([0.0, 0.0, 1.0], dtype=float)
            )
        else:
            direction /= norm

        W_T_C = compute_alignment_transforms(
            origins=[break_pt],
            align_vec=direction,  # +Z aligns with travel direction
            align_axis=[0, 0, 1],
            appr_vec=[0, 0, -1],  # CV convention
            appr_axis=[0, 1, 0],
        )[0]
        W_T_C[:3, 3] = break_pt
        return W_T_C

    def get_goal_pose(
        self, current_pose: np.ndarray, use_graph: bool = True
    ) -> np.ndarray:
        if self.object_lockin is not None:
            goal_pose = np.eye(4)
            goal_pose[:3, 3] = self.get_object_free_point(
                current_pose, self.object_lockin.centroid
            )
            goal_pose[:3, :3] = current_pose[:3, :3]  # keep current orientation
            goal_ft = None
            self.current_goal_ft_id = None
            self.current_goal_pose = None
        else:
            # Select goal frontier
            goal_ft_id = self.select_goal_frontier_id()
            self.logger.debug(f"Planning path to goal frontier ID: {goal_ft_id}")
            if goal_ft_id is None:
                self.log(
                    "error",
                    self.logging_file,
                    "No goal frontier available for path planning.",
                )
                return None, None
            goal_ft = self.frontiers[goal_ft_id]
            self.current_goal_ft_id = goal_ft_id

            # Decide goal pose
            if use_graph:
                current_pose_id = self.add_robot_poses([current_pose])[0]
                goal_pose = self.get_waypoints_from_graph(goal_ft_id, current_pose_id)

            else:
                goal_pose = self.get_frontier_pose(goal_ft)

            self.current_goal_pose = goal_pose

        return goal_pose, goal_ft

    def plan_path_to_goal(
        self,
        current_pose: np.ndarray,
        interpolate_solution: bool = True,
        use_graph: bool = True,
        depth: np.ndarray = None,
    ) -> list[np.ndarray]:
        """
        Plan a path from the current robot pose W_T_R to the selected goal frontier.

        Args:
            current_pose: (4,4) W_T_R (R→W) robot pose in world frame.
            interpolate_solution: If True, densify the planner's solution.
            use_graph: If True, compute a goal pose via graph waypoints; else use the frontier pose.

        Returns:
            List of (4,4) poses along the path (world-frame), or None if planning fails.
        """
        goal_pose, goal_ft = self.get_goal_pose(current_pose, use_graph=use_graph)

        if goal_pose is None:
            self.log(
                "error",
                self.logging_file,
                "Failed to obtain goal pose for path planning.",
            )
            return None

        self.planner.update_start_goal(start=current_pose, goal=goal_pose)

        try:
            path_found = self.planner.solve(
                time_limit=self.max_planning_time,
                method=self.planning_algo,
                depth=depth,
            )

            if not path_found:
                # If unreachable, drop this frontier as a goal candidate
                if goal_ft is not None:
                    self.remove_frontiers([goal_ft.id], planning_failure=True)
                    self.current_goal_ft_id = None
                    self._unreachable_positions.append(goal_ft.pos3d)

                self.current_goal_pose = None
                self.log(
                    "error",
                    self.logging_file,
                    "Planning failed, no path found, deleting frontier.",
                )
                return None

            if interpolate_solution:
                self.planner.interpolate_path()

            if len(self.planner.solution) == 0:
                if goal_ft is not None:
                    self.remove_frontiers([goal_ft.id], planning_failure=False)
                    self.current_goal_ft_id = None
                self.current_goal_pose = None
                self.log("error", self.logging_file, "Reached goal, deleting frontier.")
                return None

            return self.planner.get_solution_path(return_type="mat")

        except Exception as e:
            self.log("error", self.logging_file, f"Path planning failed: {e}")
            # If planning fails, drop this frontier as a goal candidate
            if goal_ft is not None:
                self.remove_frontiers([goal_ft.id], planning_failure=True)
                self.current_goal_ft_id = None
            self.current_goal_pose = None
            self.logger.debug("Dropped goal frontier due to planning failure.")
            return None

    def get_optimal_path_length(
        self, start_pose: np.ndarray, goal_pose: np.ndarray
    ) -> list[np.ndarray]:
        """
        Get the optimal path from start_pose to goal_pose using the planner.
        Args:
            start_pose: (4,4) W_T_R (R→W) robot pose in world frame.
            goal_pose: (4,4) W_T_C (C→W) goal pose in world frame.
        Returns:
            float: Length of the optimal path, or None if planning fails.
        """
        self.planner.update_start_goal(start=start_pose, goal=goal_pose)

        try:
            path_found = self.planner.solve(
                time_limit=self.max_planning_time, method=self.planning_algo
            )

            if not path_found:
                return float("inf")
            self.planner.interpolate_path()
            path = self.planner.get_solution_path(return_type="mat")
            # Compute path length
            length = 0.0
            for i in range(1, len(path)):
                p1 = path[i - 1][:3, 3]
                p2 = path[i][:3, 3]
                length += np.linalg.norm(p2 - p1)
            return length

        except Exception as e:
            self.log("error", self.logging_file, f"Path planning failed: {e}")
            return float("inf")

    def lock_into_object(self, obj: dict) -> None:
        """
        Lock the frontier manager into a specific object, creating a goal at its centroid.
        Args:
            obj: Detected object dictionary with 'centroid' key.
        """
        self.object_lockin = obj
        self.logger.debug(f"Locked into object: {obj.label} at {obj.centroid}")

    def get_graph_vis(self):
        """
        Get the visualization of the frontier graph.
        Returns:
            o3d geometry to visualize the graph.
            Frontiers as red spheres, robots as blue spheres, edges as thin cylinders.
        """
        ft_ids = self.graph.get_node_F()
        robot_ids = self.graph.get_node_R()
        FR_edges = self.graph.get_edge_FR()
        RR_edges = self.graph.get_edge_RR()
        geometries = []
        # Visualize frontiers
        for ft_id in ft_ids:
            ft = self.frontiers[ft_id]
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.075)
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate(ft.pos3d)
            geometries.append(sphere)
        # Visualize robots/cameras
        for robot_id in robot_ids:
            pose = self.robot_poses[robot_id]
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.3, origin=[0, 0, 0]
            )
            coordinate_frame.transform(pose)
            geometries.append(coordinate_frame)
        # Visualize edges
        for edge in FR_edges:
            start_id, end_id, _ = edge
            start_pos = (
                self.frontiers[start_id].pos3d
                if start_id in self.frontiers
                else self.robot_poses[start_id][:3, 3]
            )
            end_pos = (
                self.frontiers[end_id].pos3d
                if end_id in self.frontiers
                else self.robot_poses[end_id][:3, 3]
            )
            cylinder = create_cylinder_between_points(
                start_pos, end_pos, radius=0.012, color=[0, 0, 0]
            )
            if cylinder is not None:
                geometries.append(cylinder)
        for edge in RR_edges:
            start_id, end_id, _ = edge
            start_pos = self.robot_poses[start_id][:3, 3]
            end_pos = self.robot_poses[end_id][:3, 3]
            cylinder = create_cylinder_between_points(
                start_pos, end_pos, radius=0.02, color=[0.2, 0.8, 0.2]
            )
            if cylinder is not None:
                geometries.append(cylinder)
        return geometries

    ## -- File I/O --
    @staticmethod
    def _json_default(o):
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    def _detected_objects_to_dict(self) -> List[dict]:
        objects = {}

        for ft in self.all_frontiers:
            obj = ft.linked_object if ft.is_object else None
            if obj is None:
                continue

            if hasattr(obj, "to_dict"):
                obj_dict = obj.to_dict()
            else:
                obj_dict = {"value": obj}

            object_id = obj_dict.get("id", id(obj))
            objects[object_id] = obj_dict

        return list(objects.values())

    def write_to_file(
        self, file_path: str, detected_objects: Optional[List[DetectedObject]] = None
    ) -> None:
        """
        Append a JSON line snapshot of the manager state.
        """
        entry = {
            "all_frontiers": [ft.to_dict() for ft in self.all_frontiers],
            "valid_frontiers": [ft.to_dict() for ft in self.valid_frontiers],
            "detected_objects": self._detected_objects_to_dict(),
            "robot_poses": {
                rid: pose.tolist() for rid, pose in self.robot_poses.items()
            },
            "current_robot_id": self._current_robot_id,
            "current_ft_goal_id": (
                self.current_goal_ft_id if self.current_goal_ft_id is not None else None
            ),
            "current_goal_pose": (
                None
                if self.current_goal_pose is None
                else self.current_goal_pose.tolist()
            ),
            "graph": self.graph.return_current_graph(),
        }
        if detected_objects is not None:
            entry["detected_objects"] = [obj.to_dict() for obj in detected_objects]

        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(entry, default=self._json_default) + "\n")
            self.logger.debug(
                f"FrontierManager state appended to {file_path} (JSON lines)."
            )
        except Exception as e:
            self.log(
                "error",
                self.logging_file,
                f"Failed to write FrontierManager state to {file_path}: {e}",
            )

    @staticmethod
    def read_from_file(file_path):
        """
        Read the state of the FrontierManager from a file.
        Args:
            file_path: Path to the file to read from.

        Returns:
            A List of entrys, each entry is a dictionary containing the state of the FrontierManager.
        """
        entries = []
        try:
            with open(file_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    entries.append(entry)
            return entries
        except Exception as e:
            print(f"Failed to read FrontierManager state from {file_path}: {e}")
            return []

    def log(self, level: str, file: str, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        folder = os.path.dirname(file).split("/")
        folder = f"{folder[-2]}/{folder[-1]}" if len(folder) >= 2 else folder[0]
        message = f"[{folder}]\t[{timestamp}]\t[MANAGER]\t[{level.upper()}]\t{message}"
        with open(file, "a") as f:
            print(message)
            f.write(message + "\n")
