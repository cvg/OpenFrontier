import logging
from typing import Callable, Optional

import numpy as np

from planner.occ_rrt_point3d import OccupancyGrid3DPathPlanner
from utils.vis_utils import (
    capture_depth,
    capture_rgb,
    get_vis_state,
    set_vis_cam_ex,
)

from .agent import NavigationAgent


def metric_depth_from_rgb_metric3d(
    rgb_input, intrinsic_mat, camera_W, camera_H, local_model_path
):
    try:
        from mono_depth.Metric3D import metric_depth_from_rgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Metric3D depth source requires the optional Metric3D dependencies. "
            "Install Metric3D requirements or choose --depth_source GT or UniK3D."
        ) from exc

    return metric_depth_from_rgb(
        rgb_input=rgb_input,
        intrinsic_mat=intrinsic_mat,
        camera_W=camera_W,
        camera_H=camera_H,
        local_model_path=local_model_path,
    )


def metric_depth_from_rgb_unik3d(rgb_input, intrinsic_mat):
    try:
        from mono_depth.UniK3D import metric_depth_from_rgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "UniK3D depth source requires the optional 'unik3d' package. "
            "Install UniK3D or choose --depth_source GT or Metric3D."
        ) from exc

    return metric_depth_from_rgb(rgb_input=rgb_input, intrinsic_mat=intrinsic_mat)


class O3DAgent(NavigationAgent):
    DEPTH_GT = "GT"
    DEPTH_M3D = "Metric3D"
    DEPTH_UNIK3D = "UniK3D"

    def __init__(
        self,
        args,
        target: str,
        save_dir: str,
        config: dict,
        bbox: list[float],
        robot_vis,
        observer_vis=None,
        depth_source: str = DEPTH_GT,
        update_geometry_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        self.robot_vis = robot_vis
        self.observer_vis = observer_vis
        self.depth_source = depth_source
        self.update_geometry_fn = update_geometry_fn

        super().__init__(
            args=args,
            target=target,
            planner=OccupancyGrid3DPathPlanner(config),
            save_dir=save_dir,
            cam_intrinsic=get_vis_state(robot_vis)["cam_intrinsic"],
            bbox=bbox,
            get_rgbd=self.get_rgbd,
            get_cam_extrinsic=self.get_cam_extrinsic,
            move_fn=self.move_actions,
            rotate_fn=self.rotate_fn,
        )

    def get_rgbd(self):
        rgb = capture_rgb(self.robot_vis, return_rgb_type="np")
        img = rgb["image"]
        if np.max(img) <= 1:
            img = (img * 255).astype(np.uint8)

        K = get_vis_state(self.robot_vis)["cam_intrinsic"].intrinsic_matrix

        if self.depth_source == self.DEPTH_GT:
            depth = capture_depth(self.robot_vis, return_depth_type="np")["depth"]
        elif self.depth_source == self.DEPTH_UNIK3D:
            depth = metric_depth_from_rgb_unik3d(rgb_input=img, intrinsic_mat=K)
        elif self.depth_source == self.DEPTH_M3D:
            depth = metric_depth_from_rgb_metric3d(
                rgb_input=img,
                intrinsic_mat=K,
                camera_W=img.shape[1],
                camera_H=img.shape[0],
                local_model_path=None,
            )
        else:
            raise ValueError(f"Unknown depth_source: {self.depth_source}")

        return img, depth

    def get_cam_extrinsic(self):
        return get_vis_state(self.robot_vis)["cam_extrinsic"]

    def move_actions(self) -> None:
        self._apply_next_pose()

    def rotate_fn(self) -> None:
        self._apply_next_pose()

    def _apply_next_pose(self) -> None:
        if self.robot_vis is None:
            return

        observer_cam_extrinsic = None
        if self.observer_vis is not None:
            observer_cam_extrinsic = get_vis_state(self.observer_vis)["cam_extrinsic"]

        logging.debug("Moving Open3D robot camera to next pose:\n%s", self.next_W_T_C)
        set_vis_cam_ex(self.robot_vis, np.linalg.inv(self.next_W_T_C))

        self._refresh_viewers()

        if self.update_geometry_fn is not None:
            self.update_geometry_fn()

        self._refresh_viewers()
        if self.observer_vis is not None and observer_cam_extrinsic is not None:
            set_vis_cam_ex(self.observer_vis, observer_cam_extrinsic)

    def _refresh_viewers(self) -> None:
        if self.observer_vis is not None:
            self.observer_vis.update_renderer()
            self.observer_vis.poll_events()
        self.robot_vis.update_renderer()
        self.robot_vis.poll_events()
