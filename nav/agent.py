import os
import time
import logging
import argparse
import gc
from typing import Optional, List, Callable
import numpy as np
import torch
import open3d as o3d
import cv2
from vlm.utils import (
    detect_frontier_probabilities,
    segment_target_object,
    detect_target_object,
    COMPOSITIONS,
    COMPRESSION,
)

from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R

# OpenFrontier
from frontier.detector import FrontierDetector
from frontier.model.predict import load_model
from utils.frontier_utils import read_config_yaml

# Mapping
from mapping.wavemap import WaveMapper

# Frontier Manager
from frontier.manager import FrontierManager
from nav.detected_object import DetectedObject

np.set_printoptions(precision=3, suppress=True)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations, product
from planner.planner_base import PlannerBase
from utils.api_key import get_google_api_key, mask_api_key
from utils.vis_utils import is_same_pose, pose_difference


class NavigationAgent:
    """
    A lightweight wrapper for the navigation loop and state.
    """

    # ---------- constants / defaults ----------
    REFRESH_RATE = 50  # Hz
    VOX_SIZE = 0.1

    def __init__(
        self,
        args: argparse.Namespace,
        target: str,
        planner: PlannerBase,
        cam_intrinsic: np.ndarray,
        save_dir: str,
        bbox: List[float],
        get_rgbd: Callable,
        get_cam_extrinsic: Callable,
        move_fn: Callable,
        rotate_fn: Callable,
        cam_to_agent: np.ndarray = np.eye(4),
        fix_view_level = False
    ) -> None:

        self.args = args
        self.goal = target
        self.bbox = bbox
        self.cam_intrinsic = cam_intrinsic
        self.save_dir = save_dir
        self.save_images = bool(getattr(args, "save_images", False))
        self.C_T_R = cam_to_agent

        self.get_rgbd = get_rgbd
        self.get_cam_extrinsic = get_cam_extrinsic
        self.move_simulation = move_fn
        self.rotate_simulation = rotate_fn

        self.planner: PlannerBase = planner
        # Config
        self.config = read_config_yaml(args.config)
        self.predict_interval: int = int(self.config.get("predict_interval", 5))
        self.plan_interval: int = int(self.config.get("plan_interval", 10))
        self.termination_threshold: float = float(
            self.config.get("termination_threshold", 0.7)
        )
        self.success_threshold: float = float(self.config.get("success_threshold", 1.0))
        self.segmentation_source: str = self.config.get(
            "segmentation_source", "gemini-2.5-flash"
        )
        self.probabilities_source: str = self.config.get(
            "probabilities_source", "gemini-2.5-flash"
        )
        self.detection_source: str = self.config.get(
            "detection_source", "gemini-2.5-flash"
        )

        # Frontier, mapping, detector
        self.mapper: Optional[WaveMapper] = None
        self.ft_manager: Optional[FrontierManager] = None
        self.ft_detector: Optional[FrontierDetector] = None
        self.VOX_SIZE = (
            self.config["voxel_size"]
            if self.config["voxel_size"] is not None
            else self.VOX_SIZE
        )

        use_free = self.config.get("use_free_grid", True)
        use_occ = self.config.get("use_occ_grid", True)
        self.use_map = True

        # Path & motion tracking
        self.path_to_go: List[np.ndarray] = []
        self.move_enough: bool = True
        self.last_W_T_C2: np.ndarray = np.eye(4)

        self.rotated_degrees = 0

        self.vlm_log_file = os.path.join(save_dir, f"vlm_log.txt")
        self.path_taken_file = os.path.join(save_dir, f"navigation_path.npy")
        self.logging_file = os.path.join(save_dir, f"navigation_log.txt")
        self.composition_dir = os.path.join(save_dir, f"compositions")
        self.som_dir = os.path.join(save_dir, f"som_visualizations")
        self.termination_dir = os.path.join(save_dir, f"termination_visualizations")
        self.segmentation_dir = os.path.join(save_dir, f"segmentation_visualizations")
        self.depth_seg_dir = os.path.join(save_dir, f"depth_segmentations")
        args_dict = vars(args)
        self.json_path = args_dict.get(
            "write_path", os.path.join(save_dir, f"navigation_state.json")
        )

        self.planner.logging_file = os.path.join(save_dir, f"navigation_log.txt")

        # Clear directories
        for dir_path in [
            self.composition_dir,
            self.som_dir,
            self.segmentation_dir,
            self.termination_dir,
            self.depth_seg_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)
            for f in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, f))

        self.last_rgb = None
        self.last_som_img = None
        self.path_taken = []

        self.stuck_counter = 0
        self.frozen_counter = 0

        # Reset the files
        open(self.vlm_log_file, "w").close()
        open(self.logging_file, "w").close()

        self.n_images = self.config.get("n_images", 4)

        try:
            self.composition_dims = COMPOSITIONS[self.n_images]
        except KeyError:
            raise ValueError(
                f"Unsupported n_images: {self.n_images}. Supported values are: {list(COMPOSITIONS.keys())}"
            )

        self.segmentation_image = None
        self.composition_images = []
        self.termination_images = []
        self.composition_depths = []
        self.composition_viewpoints = []
        self.detected_objects = []
        self.goal_object = None
        self.appoaching_object = False

        self.optimal_path_length = float("inf")

        self.pure_exploration = (
            True if self.config.get("seeking_weight", 0.0) == 0.0 else False
        )

        self.google_api_key = get_google_api_key(required=False)
        self.navigation_steps = -1
        self.emergency_rotation = False
        self.fix_view_level = fix_view_level
        self.map_loop = 0
        self.timings = {
            "frontier_detection_step": {"total_time": 0.0, "calls": 0},
            "frontier_detection_global": {"total_time": 0.0, "calls": 0},
            "vlm_probabilities": {"total_time": 0.0, "calls": 0},
            "vlm_call": {"total_time": 0.0, "calls": 0},
            "total_segmentation": {"total_time": 0.0, "calls": 0},
            "mapping_update": {"total_time": 0.0, "calls": 0},
            "frontier_update_no_som": {"total_time": 0.0, "calls": 0},
            "frontier_manager_update": {"total_time": 0.0, "calls": 0},
            "pointnav_planning": {"total_time": 0.0, "calls": 0},
            "navigation_time": {"total_time": 0.0, "calls": 0},
            "navigation_time_no_ai": {"total_time": 0.0, "calls": 0},
            "navigation_time_no_map": {"total_time": 0.0, "calls": 0},
            "navigation_time_no_ai_no_map": {"total_time": 0.0, "calls": 0},
            "move_time": {"total_time": 0.0, "calls": 0},
            "rotate_time": {"total_time": 0.0, "calls": 0},
            "full_frontier_update": {"total_time": 0.0, "calls": 0},
            "sam": {"total_time": 0.0, "calls": 0},
            "move_habitat": {"total_time": 0.0, "calls": 0},
            "read_habitat": {"total_time": 0.0, "calls": 0}
        }
        
    def get_google_api_key(self) -> str:
        if self.google_api_key is None:
            self.google_api_key = get_google_api_key()

        return self.google_api_key

    def get_api_key_for_model(self, model) -> Optional[str]:
        if isinstance(model, str):
            model_name = model
        else:
            model_name = model.value
        model_name = model_name.lower()

        if model_name.startswith("gemini") or model_name.endswith("-api"):
            return self.get_google_api_key()

        return None

    def handle_vlm_exception(self, error: Exception, context: str) -> None:
        message = str(error)
        self.log(
            "error",
            self.logging_file,
            f"{context}: {message}",
        )

        message_lower = message.lower()
        if (
            "set gemini_api_key" in message_lower
            or "google_api_key" in message_lower
            or "api key must be provided" in message_lower
        ):
            raise RuntimeError(message) from error

        if "exhausted" in message_lower or "429" in message:
            self.log(
                "info",
                self.logging_file,
                f"Google API key {mask_api_key(self.google_api_key)} is rate limited or quota exhausted.",
            )
            raise RuntimeError("Google API quota exhausted or rate limited.") from error

        if "permission denied" in message_lower or "403" in message:
            self.log(
                "info",
                self.logging_file,
                f"Google API key {mask_api_key(self.google_api_key)} was rejected.",
            )
            raise RuntimeError("Google API permission denied.") from error

    # ---------- main navigation logic ----------

    def initialize(self) -> None:
        assert self.ft_manager is not None and (self.mapper is not None or not self.use_map)

        self.max_steps = self.args.max_steps
        self.max_time_s = self.args.max_time
        self.start_time = time.time()

        # Initial mapping (one frame) to bootstrap map
        rgb, depth0 = self.get_rgbd()

        self.segmentation_image = None
        self.last_som_img = rgb.copy()

        C2_T_W = self.get_cam_extrinsic()
        W_T_C2 = np.linalg.inv(C2_T_W)

        if self.use_map:
            self.mapper.insert_depth_to_buffer(
                depth=depth0, transform=W_T_C2
            )  # camera IN world frame
            self.log("info", self.logging_file, "Initial mapping round started.")

            self.mapper.integrate_from_buffer()
            self.mapper.interpolate_occupancy_grid()
            og = self.mapper.get_occupancy_grid()
            self.ft_manager.update_map(free_map=og["free"], occ_map=og["occupied"])

        self.poses = [W_T_C2.copy()]

        self.is_pointnav = self.planner.is_pointnav_planner

    def navigation(self, save_images: Optional[bool] = None) -> None:
        """
        Main navigation loop.
        """
        if save_images is None:
            save_images = self.save_images

        navigation_start = time.time()
        sam_time = 0.0
        mapping_time = 0.0
        full_frontier_time = 0.0
        som_time = 0.0
        
        self.timings["frontier_detection_global"]["calls"] += 1
        
        self.navigation_steps += 1
        C2_T_W = self.get_cam_extrinsic()
        W_T_C2 = np.linalg.inv(C2_T_W)
        self.poses.append(W_T_C2.copy())

        self.planner.nav_level = W_T_C2[2, 3] - self.C_T_R[2, 3]
        self.ft_manager.planner.nav_level = W_T_C2[2, 3] - self.C_T_R[2, 3]

        if self.fix_view_level:
            self.ft_detector.fix_view_level(W_T_C2[2, 3])

        if self.navigation_steps >= self.max_steps:
            self.log(
                "info", self.logging_file, "Maximum steps reached, navigation finished."
            )
            return False, "max_steps_reached"

        read_start = time.time()
        rgb, depth = self.get_rgbd()
        self.timings["read_habitat"]["total_time"] += (time.time() - read_start)
        self.timings["read_habitat"]["calls"] += 1
        
        # Always keep the latest n_images for termination
        if len(self.termination_images) == self.n_images:
            self.termination_images.pop(0)

        self.composition_images.append(rgb)
        self.termination_images.append(rgb)
        self.composition_depths.append(depth)
        self.composition_viewpoints.append(W_T_C2.copy())

        if (
            len(self.composition_images) == self.n_images
            and self.ft_manager.object_lockin is None
        ):
            start_segm = time.time()

            rgb_composition = self.compose_images(self.composition_images)
            if save_images:
                composition_path = os.path.join(
                    self.composition_dir,
                    f"{self.navigation_steps:06d}_composition.png",
                )
                self.save_rgb_image(rgb_composition, composition_path)

            try:
                start_sam = time.time()
                response = segment_target_object(
                    rgb_composition=rgb_composition,
                    n_images=self.n_images,
                    target_object=self.goal,
                    segmentation_model=self.segmentation_source,
                    api_key=self.get_api_key_for_model(self.segmentation_source),
                )
                end_sam = time.time()
                self.timings["sam"]["total_time"] += (end_sam - start_sam)
                self.timings["sam"]["calls"] += 1
                sam_time = (end_sam - start_sam)

            except Exception as e:
                print(e)
                response = None
                self.handle_vlm_exception(e, "Error during target object segmentation")

            if response is None or len(response) != 2 or len(response[0]) == 0:
                self.log(
                    "info",
                    self.logging_file,
                    f"No masks returned.",
                )
                self.segmentation_image = None
            else:
                masks, image = response
                self.segmentation_image = np.array(image)
                if save_images:
                    segmentation_path = os.path.join(
                        self.segmentation_dir,
                        f"{self.navigation_steps:06d}_segmentation.png",
                    )
                    self.save_rgb_image(self.segmentation_image, segmentation_path)

                depth_compositions = {}

                for mask in masks:
                    image_index = mask["image_index"]
                    mask_depth = self.composition_depths[image_index]
                    viewpoint = self.composition_viewpoints[image_index]

                    obj = DetectedObject.from_mask(
                        mask=mask,
                        depth=mask_depth,
                        viewpoint=viewpoint,
                        intrinsic_mat=self.cam_intrinsic.intrinsic_matrix,
                    )

                    self.detected_objects.append(obj)

                    # For debugging
                    depth_image = (mask_depth * 1000).astype(np.uint16)
                    if depth_compositions.get(image_index) is None:
                        depth_compositions[image_index] = depth_image
                    else:
                        depth_compositions[image_index] += depth_image

                if save_images:
                    for image_index, depth_image in depth_compositions.items():
                        depth_path = os.path.join(
                            self.depth_seg_dir,
                            f"{self.navigation_steps:06d}_{image_index}_depth.png",
                        )
                        cv2.imwrite(depth_path, depth_image)

                self.merge_objects()

                if self.detected_objects:
                    new_ids = self.ft_manager.add_robot_poses([W_T_C2])
                    self.ft_manager.add_objects(
                        objects=self.detected_objects, parent_ids=new_ids
                    )
                    self.ft_manager.filter_frontiers()
                    
                self.log(
                    "info",
                    self.logging_file,
                    f"Target object detection - masks obtained: {len(masks)}. Detected objects so far: {len(self.detected_objects)}",
                )

            self.composition_images = []
            self.composition_depths = []
            self.composition_viewpoints = []
            end_segm = time.time()
            self.timings["total_segmentation"]["total_time"] += (end_segm - start_segm)
            self.timings["total_segmentation"]["calls"] += 1
        
        # if time.time() - self.start_time > self.max_time_s:
        # self.log("info", self.logging_file, "Time limit reached, navigation finished.")
        # return False, "time_limit_reached"

        no_more_frontier = (
            len(self.ft_manager.valid_frontiers) == 0 and self.navigation_steps > 10
        )

        reach_next_update = len(self.path_to_go) == 0 or (
            (self.navigation_steps - 1) % self.predict_interval == 0
        )
        

        if (
            self.emergency_rotation  # When doing lifesaving rotation, check on every step
            or self.ft_manager.object_lockin is None
            and (no_more_frontier or reach_next_update)
        ):
            full_frontier_start = time.time()
            self.log("info", self.logging_file, "Updating frontiers.")

            ft_start = time.time()
            # Frontier detection + anchoring
            self.ft_detector.detect(
                rgb=rgb,
                depth=depth,
                df_normalizer=self.config["df_normalizer"],
                df_thr=self.config["df_thr"],

            )
            ft_list = self.ft_detector.anchor_fts(depth=depth, extrinsic=C2_T_W)
            ft_end = time.time()

            self.timings["frontier_detection_step"]["total_time"] += (ft_end - ft_start)
            self.timings["frontier_detection_step"]["calls"] += 1
            self.timings["frontier_detection_global"]["total_time"] += (ft_end - full_frontier_start)
            
            if ft_list is not None and len(ft_list) > 0:
                for i, ft in enumerate(ft_list):
                    ft.label = chr(65 + i) if i < 26 else str(i)  # A-Z, then 0, 1, 2...

                    if "snap_point" in dir(self.planner):
                        ft.pos3d = self.planner.snap_point(ft.pos3d)
                        ft.pos3d[2] += self.C_T_R[1, 3]  # adjust for camera height

            # Add into manager
            if ft_list is not None:
                start_ft_manager = time.time()
                new_ids = self.ft_manager.add_robot_poses([W_T_C2])
                self.ft_manager.add_frontiers(frontiers=ft_list, parent_ids=new_ids)
                self.ft_manager.filter_frontiers()
                self.ft_manager.gain_adjustment()
                self.ft_manager.filter_frontiers()
                end_ft_manager = time.time()
                self.timings["frontier_manager_update"]["total_time"] += (end_ft_manager - start_ft_manager)
                self.timings["frontier_manager_update"]["calls"] += 1   

            # Remove the frontiers that are not valid
            if ft_list is not None and len(ft_list) > 0:
                ft_list = [ft for ft in ft_list if ft.is_valid]

            if not self.pure_exploration and ft_list is not None and len(ft_list) > 0:
                som_start = time.time()

                # visualize and save marks
                image_prompt, labels, unmarked_image = self.ft_detector.get_SoM_img(
                    ft_list=ft_list, radius=20, alpha=0.5
                )

                self.last_som_img = image_prompt.copy()
                if save_images:
                    som_path = os.path.join(
                        self.som_dir,
                        f"{self.navigation_steps:06d}_som.png",
                    )
                    self.save_rgb_image(image_prompt, som_path)
                    unmarked_path = os.path.join(
                        self.som_dir,
                        f"{self.navigation_steps:06d}_unmarked.png",
                    )
                    self.save_rgb_image(unmarked_image, unmarked_path)

                attempts = 0
                while attempts < 10:
                    try:
                        start = time.time()
                        success, result, raw_response = detect_frontier_probabilities(
                            rgb_image=image_prompt,
                            labels=labels,
                            target_object=self.goal,
                            vlm_model=self.probabilities_source,
                            api_key=self.get_api_key_for_model(self.probabilities_source),
                        )
                        end = time.time()
                        self.timings["vlm_call"]["total_time"] += (end - start)
                        self.timings["vlm_call"]["calls"] += 1
                        self.log(
                            "info",
                            self.vlm_log_file,
                            f"Frontier probabilities: {raw_response}",
                        )

                        if (
                            not success
                            or not isinstance(result, dict)
                            or len(result) == 0
                        ):
                            self.log(
                                "error",
                                self.logging_file,
                                f"Invalid or unexpected probabilities returned from VLM. {raw_response}",
                            )
                            probabilities = {}
                            attempts += 1
                            continue
                        else:
                            probabilities = result

                        if probabilities is not None and len(probabilities) > 0:
                            self.log(
                                "info",
                                self.logging_file,
                                f"{len(ft_list)} frontiers detected. {len(probabilities.keys())} with probabilities.",
                            )
                           
                            for i, label in enumerate(labels):
                                if i < len(ft_list):
                                    ft = ft_list[i]
                                    if label in probabilities:
                                        ft.probability = probabilities[label][0]
                                        ft.justification = probabilities[label][1]
                                    else:
                                        ft.probability = 0.5
                                        ft.justification = (
                                            "Failed to get probability from VLM."
                                        )
                            break
                        else:
                            self.log(
                                "error",
                                self.logging_file,
                                f"Invalid or unexpected probabilities returned from VLM. {raw_response}",
                            )
                            attempts += 1
                            continue

                    except Exception as e:
                        self.handle_vlm_exception(
                            e, "Failed to get frontier probabilities from VLM"
                        )
                        attempts += 1
                        time.sleep(15.0)
                        
                som_end = time.time()
                som_time = (som_end - som_start)
                self.timings["vlm_probabilities"]["total_time"] += som_time
                self.timings["vlm_probabilities"]["calls"] += 1
                
            full_frontier_end = time.time()
            full_frontier_time = (full_frontier_end - full_frontier_start)
            self.timings["full_frontier_update"]["total_time"] += full_frontier_time
            self.timings["full_frontier_update"]["calls"] += 1
            
            frontiers_no_som_time = full_frontier_time - som_time
            self.timings["frontier_update_no_som"]["total_time"] += frontiers_no_som_time
            self.timings["frontier_update_no_som"]["calls"] += 1

        # Update mapper continuously
        if self.use_map:
            if self.map_loop == 5:
                mapping_start = time.time()
                self.mapper.integrate_from_buffer()
                self.mapper.interpolate_occupancy_grid()
                og = self.mapper.get_occupancy_grid()
                self.ft_manager.update_map(free_map=og["free"], occ_map=og["occupied"])
                mapping_end = time.time()
                mapping_time = (mapping_end - mapping_start)
                self.timings["mapping_update"]["total_time"] += mapping_time
                self.timings["mapping_update"]["calls"] += 1
                
                self.map_loop = 0
            else:
                self.map_loop += 1
            
        start_ft_manager = time.time()
        self.ft_manager.gain_adjustment()
        self.ft_manager.filter_frontiers()
        self.ft_manager.merge_frontiers()
        self.ft_manager.filter_frontiers()
        self.ft_manager.update_utility(current_pos=W_T_C2[:3, 3])
        end_ft_manager = time.time()
        self.timings["frontier_manager_update"]["total_time"] += (end_ft_manager - start_ft_manager)
        self.timings["frontier_manager_update"]["calls"] += 1

        if (
            self.ft_manager.valid_frontiers is None
            or len(self.ft_manager.valid_frontiers) == 0
        ):
            self.log(
                "info", self.logging_file, "No frontiers, adding rotation frontiers..."
            )
            self.ft_manager.add_rotation_frontiers(W_T_C=W_T_C2)
            self.emergency_rotation = True
            if (
                self.ft_manager.valid_frontiers is None
                or len(self.ft_manager.valid_frontiers) == 0
            ):
                self.log("info", self.logging_file, "Still no frontiers, rotating...")
                reach_next_update = False
                self.move_enough = False

                if self.rotate() >= 360:
                    keep_rotating = self.ft_manager.next_emergency_rotation()
                    if not keep_rotating:
                        self.log(
                            "info",
                            self.logging_file,
                            "Maximum emergency rotations reached, ending navigation.",
                        )
                        return False, "no_frontiers"
                    else:
                        self.log(
                            "info",
                            self.logging_file,
                            "Relaxed gain threshold, continuing emergency rotations.",
                        )
                        self.rotated_degrees = 0
        else:
            if self.emergency_rotation:
                self.log(
                    "info",
                    self.logging_file,
                    "Frontiers found, ending emergency rotation.",
                )
            self.emergency_rotation = False
            self.rotated_degrees = 0
            self.ft_manager.reset_emergency_rotation()

        # Replan if needed
        if self.is_pointnav or (reach_next_update and self.move_enough):
            self.log("info", self.logging_file, "Replanning...")
            logging.debug(f"Replanning (interval={self.plan_interval}).")

            pointnav_start = time.time()
            self.path_to_go = (
                self.ft_manager.plan_path_to_goal(W_T_C2, depth=depth, use_graph=False)
                or []
            )
            pointnav_end = time.time()
            self.timings["pointnav_planning"]["total_time"] += (pointnav_end - pointnav_start)
            self.timings["pointnav_planning"]["calls"] += 1

            if self.ft_manager.current_goal_ft_id is not None:
                goal_frontier = self.ft_manager.get_frontier(
                    self.ft_manager.current_goal_ft_id
                )

                if goal_frontier.is_object:
                    self.goal_object = goal_frontier.linked_object

            if self.path_to_go:
                self.log(
                    "info",
                    self.logging_file,
                    f"Path to goal found with {len(self.path_to_go)} steps.",
                )
                self.move_enough = False
            else:
                self.path_to_go = []
                self.move_enough = True  # try again next cycle

        # Check if reached target viewpoint
        if self.ft_manager.object_lockin is None and self.goal_object is not None:

            # If planning failed and path is empty, consider reached to trigger detection
            if self.ft_manager.current_goal_pose is None:
                reached = len(self.path_to_go) == 0

            else:
                trans_diff, rot_diff = pose_difference(
                    self.ft_manager.current_goal_pose.reshape(1, 4, 4),
                    W_T_C2.reshape(1, 4, 4),
                )

                self.log(
                    "info",
                    self.logging_file,
                    f"Approaching object viewpoint. Trans diff: {trans_diff[0, 0]:.2f} m, Rot diff: {rot_diff[0, 0]:.2f}",
                )

                same_pose = is_same_pose(
                    self.ft_manager.current_goal_pose,
                    W_T_C2,
                    trans_thre=0.2,
                    rot_thre=0.2,
                )

                reached = same_pose or len(self.path_to_go) == 0

            if reached:
                self.log(
                    "info",
                    self.logging_file,
                    f"Reached object viewpoint, performing target object detection.",
                )

                termination = self.compose_images(self.termination_images)
                if save_images:
                    termination_path = os.path.join(
                        self.termination_dir,
                        f"{self.navigation_steps:06d}_termination.png",
                    )
                    self.save_rgb_image(termination, termination_path)

                attempts = 0
                while attempts < 10:
                    try:
                        detect_start = time.time()
                        success, response, raw_response = detect_target_object(
                            rgb=termination,
                            target_object=self.goal,
                            vlm_model=self.detection_source,
                            api_key=self.get_api_key_for_model(self.detection_source),
                        )
                        detect_end = time.time()
                        self.timings["vlm_call"]["total_time"] += (
                            detect_end - detect_start
                        )
                        self.timings["vlm_call"]["calls"] += 1
                        self.log(
                            "info",
                            self.vlm_log_file,
                            f"Target detection: {raw_response}",
                        )
                        if success:
                            break

                        else:
                            response = None
                            self.log(
                                "info",
                                self.logging_file,
                                f"JSON error during target object detection. {raw_response}",
                            )

                    except Exception as e:
                        response = None
                        self.handle_vlm_exception(
                            e, "Error during target object detection"
                        )
                        attempts += 1
                        time.sleep(15.0)

                if (
                    response is None
                    or not isinstance(response, dict)
                    or len(response) == 0
                ):
                    self.log(
                        "info",
                        self.logging_file,
                        f"No valid response from VLM for target object detection: {response}",
                    )
                else:
                    probability = response.get("probability", 0.0)
                    reason = response.get("reason", "")
                    self.log(
                        "info",
                        self.logging_file,
                        f"Target object detection - Probability: {probability}, Reason: {reason}",
                    )
                    # Check termination condition
                    if probability >= self.termination_threshold:
                        self.goal_object.verification_status = "true_positive"
                        self.log(
                            "info",
                            self.logging_file,
                            f"Object found with probability {probability}. Approaching object.",
                        )
                        self.ft_manager.lock_into_object(self.goal_object)
                        self.path_to_go = (
                            self.ft_manager.plan_path_to_goal(
                                W_T_C2, depth=depth, use_graph=False
                            )
                            or []
                        )
                    else:
                        self.goal_object.is_valid = False
                        self.goal_object.verification_status = "false_positive"
                        self.log(
                            "info",
                            self.logging_file,
                            f"Object not found (probability {probability}), continuing exploration.",
                        )

                # remove the current goal frontier to avoid repeated checks
                self.goal_object.frontier.set_invalid()
                goal_ft = self.ft_manager.get_frontier(self.ft_manager.current_goal_ft_id)
                if goal_ft is not None:
                    goal_ft.set_invalid()
                self.goal_object = None
                self.ft_manager.filter_frontiers()
                reach_next_update = True
                self.move_enough = True

        if self.ft_manager.object_lockin is not None:
            object_pos = self.ft_manager.object_lockin.centroid
            dist = np.linalg.norm(object_pos - W_T_C2[:3, 3])

            self.log(
                "info",
                self.logging_file,
                f"Approaching locked-in object. Distance to object center: {dist:.2f} m",
            )

            if len(self.path_to_go) == 0 or dist < self.success_threshold:
                self.log(
                    "info",
                    self.logging_file,
                    f"Reached object, navigation finished successfully.",
                )
                return False, "object_found"
            
        # Execute one movement step if path exists
        if self.path_to_go:
            logging.debug("Moving along the path.")
            self.move(steps=1)

        if np.allclose(W_T_C2[:3, 3], self.last_W_T_C2[:3, 3]):
            self.stuck_counter += 1

            if self.stuck_counter == 200:
                self.log(
                    "info",
                    self.logging_file,
                    "Robot stuck detected after 200 steps, ending navigation.",
                )
                return False, "robot_stuck"
        else:
            self.stuck_counter = 0

        if is_same_pose(W_T_C2, self.last_W_T_C2):
            self.frozen_counter += 1

            if self.stuck_counter % 5 == 0:
                self.log(
                    "info",
                    self.logging_file,
                    f"Robot frozen for {self.frozen_counter} steps.",
                )
                reach_next_update = True
                self.move_enough = True
                self.rotate()

        else:
            self.frozen_counter = 0
    
        navigation_time = time.time() - navigation_start
        nav_no_ai = navigation_time - sam_time - full_frontier_time
        nav_no_map = navigation_time - mapping_time
        nav_no_ai_no_map = navigation_time - sam_time - full_frontier_time - mapping_time
        self.timings["navigation_time"]["total_time"] += navigation_time
        self.timings["navigation_time"]["calls"] += 1
        self.timings["navigation_time_no_ai"]["total_time"] += nav_no_ai
        self.timings["navigation_time_no_ai"]["calls"] += 1
        self.timings["navigation_time_no_map"]["total_time"] += nav_no_map
        self.timings["navigation_time_no_map"]["calls"] += 1
        self.timings["navigation_time_no_ai_no_map"]["total_time"] += nav_no_ai_no_map
        self.timings["navigation_time_no_ai_no_map"]["calls"] += 1

        if self.json_path is not None:
            self.ft_manager.write_to_file(
                file_path=self.json_path, detected_objects=self.detected_objects
            )
        
        return True, "continue_navigation"

    def move(self, steps: int) -> None:
        """
        Execute up to `steps` motions along the path, acquire depth, and update mapper & manager.
        """
        start = time.time()
        self.rotated_degrees = 0

        if not self.path_to_go:
            self.log("info", self.logging_file, "No path to follow.")
            return

        for _ in range(steps):
            if not self.path_to_go:
                self.log("info", self.logging_file, "Path exhausted.")
                break

            if not self.is_pointnav:
                self.next_W_T_C = self.path_to_go.pop(0)

            move_start = time.time()
            self.move_simulation()
            move_end = time.time()
            self.timings["move_habitat"]["total_time"] += (move_end - move_start)
            self.timings["move_habitat"]["calls"] += 1

            self.log(
                "info",
                self.logging_file,
                f" -------Current navigation step: {self.navigation_steps} -------",
            )


            # Capture new depth
            read_start = time.time()
            _, depth = self.get_rgbd()
            read_end = time.time()
            self.timings["read_habitat"]["total_time"] += (read_end - read_start)
            self.timings["read_habitat"]["calls"] += 1

            # Insert into mapper
            C2_T_W = self.get_cam_extrinsic()
            W_T_C2 = np.linalg.inv(C2_T_W)

            if self.use_map:
                self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C2)

            # If we truly moved, update path bookkeeping
            if not is_same_pose(self.last_W_T_C2, W_T_C2):
                self.last_W_T_C2 = W_T_C2.copy()
                if self.ft_manager is not None:
                    self.ft_manager.add_robot_poses([W_T_C2])
                self.move_enough = True
                
        self.timings["move_time"]["total_time"] += (time.time() - start)
        self.timings["move_time"]["calls"] += 1


    def get_agent_pose(self, W_T_C2: np.ndarray) -> np.ndarray:
        """
        Get the agent's pose in world coordinates given the camera extrinsic.
        """
        W_T_A = W_T_C2 @ self.C_T_R
        return W_T_A

    def get_camera_pose(self, W_T_A: np.ndarray) -> np.ndarray:
        """
        Get the camera's pose in world coordinates given the agent extrinsic.
        """
        A_T_C2 = np.linalg.inv(self.C_T_R)
        W_T_C2 = W_T_A @ A_T_C2
        return W_T_C2

    def plot_3d(self, W_T_C2, depth=None) -> None:

        axes = plt.figure().add_subplot(111, projection="3d")
        for pose in [self.poses[-1]]:
            robot_pos = pose[:3, 3]
            robot_dir_x = pose[:3, 0]
            robot_dir_y = pose[:3, 1]
            robot_dir_z = pose[:3, 2]
            axes.quiver(
                robot_pos[0],
                robot_pos[1],
                robot_pos[2],
                robot_dir_x[0],
                robot_dir_x[1],
                robot_dir_x[2],
                length=0.2,
                color="r",
                arrow_length_ratio=0.3,
            )  # X axis
            axes.quiver(
                robot_pos[0],
                robot_pos[1],
                robot_pos[2],
                robot_dir_y[0],
                robot_dir_y[1],
                robot_dir_y[2],
                length=0.2,
                color="g",
                arrow_length_ratio=0.3,
            )  # Y axis
            axes.quiver(
                robot_pos[0],
                robot_pos[1],
                robot_pos[2],
                robot_dir_z[0],
                robot_dir_z[1],
                robot_dir_z[2],
                length=0.2,
                color="b",
                arrow_length_ratio=0.3,
            )  # Z axis

        # Set limits around bbox
        min_min = min(self.bbox)
        max_max = max(self.bbox)

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

        if depth is not None:
            # Plot the depth points
            height, width = depth.shape

            points = self.depth_to_point_cloud(
                depth, self.cam_intrinsic.intrinsic_matrix
            )  # Nx3

            points_homogeneous = np.hstack(
                (points, np.ones((points.shape[0], 1), dtype=points.dtype))
            )  # Nx4

            # Convert points to world coordinates
            points = (W_T_C2 @ points_homogeneous.T).T[:, :3]  # Nx3

            # Plot one every 10 points
            points_sparse = points[::10, :]

            axes.scatter(
                points_sparse[:, 0],
                points_sparse[:, 1],
                points_sparse[:, 2],
                c="gray",
                s=10,
            )

        if self.ft_manager is not None:
            # Plot frontiers as arrows along the z direction
            for ft in self.ft_manager.all_frontiers:
                pose = ft.pose6d
                ft_pos = pose[:3, 3]
                ft_dir = pose[:3, 2]  # Z direction
                axes.quiver(
                    ft_pos[0],
                    ft_pos[1],
                    ft_pos[2],
                    ft_dir[0],
                    ft_dir[1],
                    ft_dir[2],
                    length=0.3,
                    color="magenta",
                    arrow_length_ratio=0.3,
                )

        # View the plot so +x is up and +z is right
        axes.view_init(elev=90, azim=-90)

        # Draw bounding box
        if self.bbox is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = self.bbox
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

        occ_map = self.ft_manager.occ_map

        axes.scatter(
            occ_map[:, 0],
            occ_map[:, 1],
            occ_map[:, 2],
            c="red",
            s=10,
        )

        # Plot path to goal
        if self.path_to_go:
            path_points = np.array([pose[:3, 3] for pose in self.path_to_go])  # Nx3
            axes.plot(
                path_points[:, 0],
                path_points[:, 1],
                path_points[:, 2],
                c="orange",
                linewidth=2,
            )

        plt.show(block=True)

    def rotate(self):
        """
        Rotate in place 30 degrees
        """
        start = time.time()        

        self.next_W_T_C = np.linalg.inv(self.get_cam_extrinsic())
        R_curr = self.next_W_T_C[:3, :3]
        R_delta = R.from_euler("Z", -30, degrees=True).as_matrix()
        R_next = R_delta @ R_curr
        self.next_W_T_C[:3, :3] = R_next
        self.rotated_degrees += 30

        move_start = time.time()
        self.rotate_simulation()
        move_end = time.time()
        self.timings["move_habitat"]["total_time"] += (move_end - move_start)
        self.timings["move_habitat"]["calls"] += 1

        # Capture new depth
        read_start = time.time()
        _, depth = self.get_rgbd()
        read_end = time.time()
        self.timings["read_habitat"]["total_time"] += (read_end - read_start)
        self.timings["read_habitat"]["calls"] += 1

        # Insert into mapper
        C2_T_W = self.get_cam_extrinsic()
        W_T_C2 = np.linalg.inv(C2_T_W)
        if self.use_map:
            self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C2)

        # If we truly moved, update path bookkeeping
        if not np.allclose(W_T_C2, self.last_W_T_C2):
            self.last_W_T_C2 = W_T_C2
            if self.ft_manager is not None:
                self.ft_manager.add_robot_poses([W_T_C2])
            self.move_enough = True

        end = time.time()
        self.timings["rotate_time"]["total_time"] += (end - start)
        self.timings["rotate_time"]["calls"] += 1
        return self.rotated_degrees

    def save_rgb_image(self, image: np.ndarray, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        output = np.asarray(image)
        if np.max(output) <= 1:
            output = (output * 255).astype(np.uint8)
        else:
            output = output.astype(np.uint8)
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        elif output.shape[2] == 4:
            output = output[:, :, :3]
        cv2.imwrite(path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.composition_images.clear()
        self.termination_images.clear()
        self.composition_depths.clear()
        self.composition_viewpoints.clear()
        self.last_rgb = None
        self.last_som_img = None
        self.segmentation_image = None

        if self.mapper is not None:
            self.mapper.close()
            self.mapper = None

        gc.collect()

    def compose_images(self, image_array, compress=True) -> np.ndarray:
        """
        Compose multiple images into a grid for visualization.
        """
        rows, cols = self.composition_dims
        img_h, img_w = image_array[0].shape[:2]
        canvas_h = rows * img_h
        canvas_w = cols * img_w
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)  # black background

        for idx, img in enumerate(image_array):
            r = idx // cols
            c = idx % cols
            start_y = r * img_h
            start_x = c * img_w
            canvas[start_y : start_y + img_h, start_x : start_x + img_w, :] = img

        if compress:
            canvas = cv2.resize(
                canvas,
                None,
                fx=COMPRESSION,
                fy=COMPRESSION,
                interpolation=cv2.INTER_AREA,
            )
        return canvas

    def merge_objects(self, distance_threshold: float = 0.5) -> None:
        """
        Merge detected objects that are close to each other with DBSCAN
        """

        # Extract centroids
        centroids = []
        for obj in self.detected_objects:
            if obj.centroid is not None:
                centroids.append(obj.centroid)

        if len(centroids) == 0:
            return
        centroids = np.array(centroids)
        # DBSCAN clustering
        clustering = DBSCAN(
            eps=distance_threshold, min_samples=1, metric="euclidean"
        ).fit_predict(centroids)
        merged_objects = []
        for cluster_id in np.unique(clustering):
            cluster_indices = np.where(clustering == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            if len(cluster_indices) == 1:
                merged_objects.append(self.detected_objects[cluster_indices[0]])
                continue
            # Merge objects in this cluster
            merged_obj = self.detected_objects[
                cluster_indices[0]
            ]  # start with the first object's data

            merged_centroids = []
            merged_viewpoints = []
            for idx in cluster_indices:
                obj = self.detected_objects[idx]
                if obj.centroid is not None:
                    merged_centroids.append(obj.centroid)
                if obj.viewpoint is not None:
                    merged_viewpoints.append(obj.viewpoint)

                # All objects must be valid to keep the merged one valid
                merged_obj.is_valid = merged_obj.is_valid and obj.is_valid

                # if any object is verified true positive, mark merged as true positive
                if obj.verification_status == "true_positive":
                    merged_obj.verification_status = "true_positive"
                elif obj.verification_status == "false_positive":
                    if merged_obj.verification_status != "true_positive":
                        merged_obj.verification_status = "false_positive"

                if self.goal_object is not None and obj.id == self.goal_object.id:
                    self.goal_object = merged_obj

            if len(merged_centroids) > 0:
                merged_obj.centroid = np.mean(merged_centroids, axis=0)
            if len(merged_viewpoints) > 0:
                merged_obj.viewpoint = np.mean(merged_viewpoints, axis=0)

            merged_objects.append(merged_obj)

        self.detected_objects = merged_objects

    # ---------- setup ----------
    def setup_system(self) -> None:
        """Init mapper, detector, and manager."""
        # Mapper
        intr2 = self.cam_intrinsic

        params = {
            "min_cell_width": self.VOX_SIZE / 2.0,
            "width": intr2.width,
            "height": intr2.height,
            "fx": intr2.intrinsic_matrix[0, 0],
            "fy": intr2.intrinsic_matrix[1, 1],
            "cx": intr2.intrinsic_matrix[0, 2],
            "cy": intr2.intrinsic_matrix[1, 2],
            "min_range": 0.05,
            "max_range": (
                self.config["depth_range"]
                if self.config["depth_range"] is not None
                else 3.5
            ),
            "resolution": self.VOX_SIZE,
        }

        # save intrinsics for later
        json_intr_path = os.path.join(self.save_dir, "camera_intrinsics.json")
        o3d.io.write_pinhole_camera_intrinsic(json_intr_path, intr2)
        self.log(
            "info", self.logging_file, f"Saved camera intrinsics to {json_intr_path}"
        )

        if self.use_map:
            self.mapper = WaveMapper(params=params)

        # OpenFrontier
        unet = load_model(
            path=self.args.unet_weight,
            num_classes=self.config["num_classes"],
            use_depth=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ft_detector = FrontierDetector(
            model=unet,
            camera_intrinsic=intr2.intrinsic_matrix.copy(),
            use_depth=True,
            img_size_model=self.config["input_img_size"],
            device=device,
            log_level=self.args.log_level,
        )

        # Frontier Manager
        self.ft_manager = FrontierManager(
            params=self.config, log_level=self.args.log_level, planner=self.planner
        )
        self.ft_manager.logging_file = self.logging_file
        self.ft_manager.filter_bbox = self.bbox
        self.ft_manager.planner.set_bounds(self.bbox)
        self.ft_manager.planner.logging_file = self.logging_file

    def log(self, level: str, file: str, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        folder = os.path.dirname(file).split("/")
        folder = f"{folder[-2]}/{folder[-1]}" if len(folder) >= 2 else folder[0]
        message = f"[{folder}]\t[{timestamp}]\t[AGENT]\t[{self.navigation_steps}]\t[{level.upper()}]\t{message}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "a") as f:
            print(message)
            f.write(message + "\n")
