import os
import time
import threading
import logging
import argparse
import sys
import traceback
from typing import List
from pathlib import Path
import numpy as np
import open3d as o3d
from utils.cv2_setup import configure_opencv_qt_fonts

configure_opencv_qt_fonts()
import cv2

from utils.vis_utils import (
    get_vis_state,
    set_vis_cam_ex,
    set_vis_cam_intr,
    is_vis_moving,
    camera_vis_with_cylinders,
    capture_rgb,
    create_camera,
    create_interactive_vis,
    load_mesh,
    register_basic_callbacks,
)

# OpenFrontier
from utils.frontier_utils import read_config_yaml

from nav.o3d_agent import O3DAgent

np.set_printoptions(precision=3, suppress=True)


class NavigationApp:
    """
    A lightweight wrapper for the navigation loop and state.
    """

    # ---------- constants / defaults ----------
    REFRESH_RATE = 50  # Hz
    VOX_SIZE = 0.1

    # Camera-1 (observer) defaults
    CAM1_H, CAM1_W, CAM1_F = 960, 1280, 700.0
    # Camera-2 (robot) defaults
    CAM2_H, CAM2_W, CAM2_F = 480, 480, 300.0
    SOM_WINDOW_NAME = "OpenFrontier SoM"

    # Depth sources
    DEPTH_GT = O3DAgent.DEPTH_GT
    DEPTH_M3D = O3DAgent.DEPTH_M3D
    DEPTH_UNIK3D = O3DAgent.DEPTH_UNIK3D

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mesh_path = args.mesh
        if args.write_path:
            self.save_dir = os.path.dirname(args.write_path) or "."
        else:
            self.save_dir = "navigation_output"
        os.makedirs(self.save_dir, exist_ok=True)

        # Config
        self.config = read_config_yaml(args.config)
        self.success_threshold: float = float(self.config.get("success_threshold", 1.0))

        self.initial_cam_extrinsic = self.config["initial_cam_extrinsic"]

        self.logging_file = os.path.join(self.save_dir, f"navigation_log.txt")

        # Config
        self.config = read_config_yaml(args.config)

        # Depth source
        self.depth_source = args.depth_source

        # Visualization windows
        self.vis_1 = None  # observer view
        self.vis_2 = None  # robot view
        self.som_window_ready = False
        self.agent = None

        # Sensors
        ## set custom camera parameters if needed
        # self.CAM2_H = args.image_height if args.image_height is not None else self.CAM2_H
        # self.CAM2_W = args.image_width if args.image_width is not None else self.CAM2_W
        # self.CAM2_F = args.focal_length if args.focal_length is not None else self.CAM2_F

        # Geometry caches (for visualization in o3d)
        self.geometry_vis_1: List[o3d.geometry.Geometry] = []
        self.ft_geometry_vis_1: List[o3d.geometry.Geometry] = []
        self.object_geometry_vis_1: List[o3d.geometry.Geometry] = []

        # mutex for vis updates
        self._lock = threading.Lock()

        self.video_writer = None
        self.video_writer_size = None
        self.video_path = os.path.join(self.save_dir, f"navigation_video.mp4")
        self.preview_path = os.path.join(self.save_dir, "topdown.png")
        self.video_finalized = False
        self.navigation_finished = False

    def setup(self) -> None:

        self.setup_viewers()

        self.agent = O3DAgent(
            args=self.args,
            target=self.args.target,
            save_dir=self.save_dir,
            config=self.config,
            bbox=self.bbox,
            robot_vis=self.vis_2,
            observer_vis=self.vis_1,
            depth_source=self.depth_source,
            update_geometry_fn=self.update_geometry_vis_1,
        )

        self.agent.setup_system()
        self.agent.log(
            "info",
            self.agent.logging_file,
            "Open3D navigation demo initialized.",
        )
        self.agent.log(
            "info",
            self.agent.vlm_log_file,
            "Open3D navigation demo VLM log initialized.",
        )

    # ---------- visualization helpers ----------

    def update_vis(self) -> None:
        """Update both viewers while preserving their camera extrinsics."""
        if self.vis_1 is None or self.vis_2 is None:
            return

        cam_ex_1 = get_vis_state(self.vis_1)["cam_extrinsic"]
        cam_ex_2 = get_vis_state(self.vis_2)["cam_extrinsic"]

        # If vis_2 moved enough, refresh frustum and overlays
        if is_vis_moving(
            self.vis_2, self.agent.last_W_T_C2, trans_thre=0.1, rot_thre=0.26
        ):
            # Store latest
            self.agent.last_W_T_C2 = np.linalg.inv(cam_ex_2)
            self.update_geometry_vis_1()

        # Render and restore cams
        self.vis_1.update_renderer()
        self.vis_2.update_renderer()
        set_vis_cam_ex(self.vis_1, cam_ex_1)
        set_vis_cam_ex(self.vis_2, cam_ex_2)
        self.update_som_window()

    def update_som_window(self) -> None:
        if self.args.headless:
            return

        if not self.som_window_ready:
            return

        som_img = self.agent.last_som_img
        if som_img is None:
            return

        if np.max(som_img) <= 1:
            som_img = (som_img * 255).astype(np.uint8)
        else:
            som_img = som_img.astype(np.uint8)

        if som_img.shape[0] != self.CAM2_H or som_img.shape[1] != self.CAM2_W:
            som_img = cv2.resize(som_img, (self.CAM2_W, self.CAM2_H))

        cv2.imshow(self.SOM_WINDOW_NAME, cv2.cvtColor(som_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def update_geometry_vis_1(self) -> None:
        """
        Update o3d geoms overlays in vis_1.
        """
        if self.vis_1 is None or self.vis_2 is None:
            return

        # vis_2 camera poses
        C2_T_W = get_vis_state(self.vis_2)["cam_extrinsic"]
        W_T_C2 = np.linalg.inv(C2_T_W)

        intr = get_vis_state(self.vis_2)["cam_intrinsic"]
        W2, H2 = intr.width, intr.height
        fx2 = intr.intrinsic_matrix[0, 0]
        # fy2 = intr.intrinsic_matrix[1, 1]

        wh_ratio = W2 / H2
        fovx_deg = 2.0 * np.degrees(np.arctan(W2 / (2.0 * fx2)))

        frustum_meshes = camera_vis_with_cylinders(
            W_T_C2,
            wh_ratio=wh_ratio,
            scale=0.8,
            weight=0.0,
            color=[0, 0, 1],
            fovx=fovx_deg,
            radius=0.04,
        )

        # Clear previous
        for g in self.geometry_vis_1:
            self.vis_1.remove_geometry(g)
        self.geometry_vis_1.clear()

        # Frontiers overlay
        self.visualize_all_frontiers(self.vis_1)

        # Frustum overlay
        for g in frustum_meshes:
            self.geometry_vis_1.append(g)
            self.vis_1.add_geometry(g, reset_bounding_box=False)

        if self.args.vis_graph:
            # (optional) topo-graph overlay
            graph_vis = self.agent.ft_manager.get_graph_vis()
            for g in graph_vis:
                self.geometry_vis_1.append(g)
                self.vis_1.add_geometry(g, reset_bounding_box=False)

        self.visualize_detected_objects()

    def visualize_detected_objects(self) -> None:
        if self.vis_1 is None or self.agent is None:
            return

        C_T_W = get_vis_state(self.vis_1)["cam_extrinsic"]

        for g in self.object_geometry_vis_1:
            self.vis_1.remove_geometry(g)
        self.object_geometry_vis_1.clear()

        for obj in self.agent.detected_objects:
            if obj.centroid is None or not obj.is_valid:
                continue
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
            sphere.paint_uniform_color([0, 1, 0])
            sphere.translate(obj.centroid)
            self.object_geometry_vis_1.append(sphere)
            self.vis_1.add_geometry(sphere, reset_bounding_box=False)

        set_vis_cam_ex(self.vis_1, C_T_W)

    # ---------- main navigation logic ----------

    def navigation(self) -> None:
        """
        Main navigation loop.
        """
        assert self.vis_1 is not None and self.vis_2 is not None
        self.agent.log(
            "info",
            self.agent.logging_file,
            "Open3D navigation loop started.",
        )

        self.vis_1.update_renderer()
        self.vis_2.update_renderer()

        self.agent.initialize()

        try:
            while True:
                should_continue, reason = self.agent.navigation(
                    save_images=self.args.save_images
                )
                self.update_som_window()
                self.visualize_detected_objects()
                if self.args.record_video or self.args.image_preview:
                    self.write_video()
                if not should_continue:
                    self.agent.ft_manager.write_to_file(
                        file_path=self.agent.json_path,
                        detected_objects=self.agent.detected_objects,
                    )
                    self.agent.log(
                        "info",
                        self.logging_file,
                        f"Navigation stopped: {reason}",
                    )
                    break
        finally:
            self.agent.log(
                "info",
                self.logging_file,
                f"Navigation finished, total steps: {len(self.agent.poses)}",
            )
            self.finalize_video()
            self.navigation_finished = True

    # ---------- visualization: frontiers ----------
    def normalize_video_image(self, image: np.ndarray) -> np.ndarray:
        output = np.asarray(image)
        if np.max(output) <= 1:
            output = (output * 255).astype(np.uint8)
        else:
            output = output.astype(np.uint8)
        if output.ndim == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        elif output.shape[2] == 4:
            output = output[:, :, :3]
        return output

    def draw_video_text(
        self, image: np.ndarray, text: str, origin: tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        output = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        padding = 6
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )
        left = max(origin[0] - padding, 0)
        top = max(origin[1] - text_height - padding, 0)
        right = min(origin[0] + text_width + padding, output.shape[1])
        bottom = min(origin[1] + baseline + padding, output.shape[0])
        cv2.rectangle(output, (left, top), (right, bottom), (0, 0, 0), -1)
        cv2.putText(
            output,
            text,
            origin,
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return output

    def annotate_video_panel(self, image: np.ndarray, label: str) -> np.ndarray:
        output = image.copy()
        output[:5, :, :] = 0
        output[-5:, :, :] = 0
        output[:, :5, :] = 0
        output[:, -5:, :] = 0
        return self.draw_video_text(output, label)

    def write_video(self):
        if not self.args.record_video and not self.args.image_preview:
            return

        if self.video_finalized:
            return

        topdown_image = capture_rgb(self.vis_1, return_rgb_type="np")["image"]
        rgb = capture_rgb(self.vis_2, return_rgb_type="np")["image"]

        if topdown_image is None or rgb is None:
            logging.warning("capture_rgb returned None, skipping video frame.")
            return

        topdown_image = self.normalize_video_image(topdown_image)
        rgb = self.normalize_video_image(rgb)

        topdown_image = self.draw_video_text(
            topdown_image,
            f"{Path(self.mesh_path).stem} | {self.agent.goal} | Step: {self.agent.navigation_steps}",
        )

        rgb = self.annotate_video_panel(rgb, "Camera View")

        if self.agent.last_som_img is not None:
            som_image = self.normalize_video_image(self.agent.last_som_img)
            som_image = cv2.resize(som_image, (rgb.shape[1], rgb.shape[0]))
            som_image = self.annotate_video_panel(som_image, "Frontiers")
            rgb = cv2.vconcat([rgb, som_image])

        if topdown_image.shape[0] > rgb.shape[0]:
            diff = topdown_image.shape[0] - rgb.shape[0]
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            rgb = cv2.copyMakeBorder(
                rgb,
                pad_top,
                pad_bottom,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )
        else:
            diff = rgb.shape[0] - topdown_image.shape[0]
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            topdown_image = cv2.copyMakeBorder(
                topdown_image,
                pad_top,
                pad_bottom,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )

        combined = cv2.hconcat([rgb, topdown_image])
        if self.args.image_preview:
            cv2.imwrite(self.preview_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        frame_size = (combined.shape[1], combined.shape[0])
        if not self.args.record_video:
            return

        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.video_path,
                fourcc,
                15.0,
                frame_size,
            )
            self.video_writer_size = frame_size
        elif frame_size != self.video_writer_size:
            combined = cv2.resize(combined, self.video_writer_size)

        if not self.video_writer.isOpened():
            logging.error("video_writer is not opened. Check codec and filepath.")
        else:
            self.video_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    def finalize_video(self) -> None:
        if self.video_finalized:
            return
        if self.video_writer is not None:
            print("Saving video to", self.video_path)
            self.video_writer.release()
            self.video_writer = None
            print("Video saved!")
        self.video_finalized = True

    def close_outputs(self) -> None:
        if (
            (self.args.record_video or self.args.image_preview)
            and
            not self.video_finalized
            and self.vis_1 is not None
            and self.vis_2 is not None
        ):
            try:
                self.write_video()
            except Exception as e:
                logging.warning("Failed to write final video frame: %s", e)
        self.finalize_video()
        if self.agent is not None:
            self.agent.close()
        cv2.destroyAllWindows()

    def visualize_all_frontiers(self, vis) -> None:
        """
        Overlay all valid frontiers in vis_1 (frustums + axes + goal marker).
        """
        if self.agent.ft_manager is None:
            return

        C_T_W = get_vis_state(vis)["cam_extrinsic"]  # stash/restore
        for g in self.ft_geometry_vis_1:
            vis.remove_geometry(g)
        self.ft_geometry_vis_1.clear()

        if len(self.agent.ft_manager.all_frontiers) == 0:
            logging.debug("No frontiers to visualize.")
            set_vis_cam_ex(vis, C_T_W)
            return

        # Get normalization factor to color frontiers by utility from 0 to 1
        utilities = [ft.utility for ft in self.agent.ft_manager.valid_frontiers]
        min_utility = min(utilities)
        max_utility = max(utilities)
        utility_range = max_utility - min_utility if max_utility > min_utility else 1.0

        # Draw each frontier frustum
        for ft in self.agent.ft_manager.valid_frontiers:
            W_T_C = ft.pose6d  # already W_T_C
            frustum = camera_vis_with_cylinders(
                W_T_C,
                wh_ratio=self.CAM2_W / self.CAM2_H,
                scale=0.6,
                weight=(ft.utility - min_utility) / utility_range,
                fovx=2 * np.degrees(np.arctan(self.CAM2_W / (2 * self.CAM2_F))),
                radius=0.04,
                return_mesh=False,
            )
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0]
            )
            axis.transform(W_T_C)
            frustum.append(axis)
            for g in frustum:
                self.ft_geometry_vis_1.append(g)
                vis.add_geometry(g, reset_bounding_box=False)

        # Mark current goal
        try:
            if self.agent.ft_manager.current_goal_pose is not None:
                goal_pose = self.agent.ft_manager.current_goal_pose[:3, 3]
                if goal_pose is not None:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
                    sphere.paint_uniform_color([0, 0, 1])
                    sphere.translate(goal_pose)
                    vis.add_geometry(sphere, reset_bounding_box=False)
                    self.ft_geometry_vis_1.append(sphere)
        except Exception as e:
            logging.error(f"Error visualizing next goal frontier: {e}")

        set_vis_cam_ex(vis, C_T_W)

    # ---------- setup ----------

    def setup_viewers(self) -> None:
        """Create viewers and add the scene mesh & initial overlays."""
        # Logging level
        # Cameras
        cam_intr_1 = create_camera(self.CAM1_H, self.CAM1_W, self.CAM1_F)
        cam_intr_2 = create_camera(self.CAM2_H, self.CAM2_W, self.CAM2_F)

        self.vis_1 = create_interactive_vis(
            self.CAM1_H,
            self.CAM1_W,
            cam_intr_1,
            show_back_face=False,
            light_on=False,
            z_near=0.02,
            z_far=50.0,
            visible=not self.args.headless,
        )
        self.vis_2 = create_interactive_vis(
            self.CAM2_H,
            self.CAM2_W,
            cam_intr_2,
            show_back_face=True,
            light_on=False,
            z_near=0.02,
            z_far=50.0,
            visible=not self.args.headless,
        )

        # Load scene mesh
        scene_mesh = load_mesh(f"{self.mesh_path}")
        # Get bounding box from scene
        bbox = scene_mesh.get_axis_aligned_bounding_box()
        self.bbox = [
            bbox.min_bound[0],
            bbox.max_bound[0],
            bbox.min_bound[1],
            bbox.max_bound[1],
            bbox.min_bound[2],
            bbox.max_bound[2],
        ]

        self.vis_1.add_geometry(scene_mesh, reset_bounding_box=True)
        self.vis_2.add_geometry(scene_mesh, reset_bounding_box=True)
        if not self.args.headless:
            cv2.namedWindow(self.SOM_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.SOM_WINDOW_NAME, self.CAM2_W, self.CAM2_H)
            blank_som = np.full((self.CAM2_H, self.CAM2_W, 3), 255, dtype=np.uint8)
            cv2.imshow(self.SOM_WINDOW_NAME, blank_som)
            cv2.waitKey(1)
            self.som_window_ready = True

        # Initial frustum of vis_2, drawn in vis_1
        C2_T_W = get_vis_state(self.vis_2)["cam_extrinsic"]  # (W→C2)
        frustum = camera_vis_with_cylinders(
            C2_T_W,
            wh_ratio=self.CAM2_W / self.CAM2_H,
            scale=0.8,
            weight=0.0,
            fovx=2 * np.degrees(np.arctan(self.CAM2_W / (2 * self.CAM2_F))),
            radius=0.04,
        )
        for g in frustum:
            self.geometry_vis_1.append(g)
            self.vis_1.add_geometry(g)

        # Global axis
        world_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        self.vis_1.add_geometry(world_axis, reset_bounding_box=False)

        # Register basic callbacks
        if not self.args.headless:
            register_basic_callbacks(self.vis_1)
            register_basic_callbacks(self.vis_2)

        # Reset intrinsics
        set_vis_cam_intr(self.vis_1, cam_intr_1)
        set_vis_cam_intr(self.vis_2, cam_intr_2)

        # Observer & robot poses from config
        obs_C_T_W = np.asarray(self.config["observer_cam_extrinsic"], dtype=float)
        rob_C_T_W = np.asarray(self.initial_cam_extrinsic, dtype=float)
        set_vis_cam_ex(self.vis_1, obs_C_T_W)
        set_vis_cam_ex(self.vis_2, rob_C_T_W)

    def run(self) -> int:
        """Run the navigation loop until terminated."""
        self.setup()
        exit_code = 0

        try:
            if self.args.auto_start:
                self.navigation()

            else:
                logging.info(
                    " --- You are in manual mode, you can move the camera using WASD(translation), JL(rotation), QZ(height) keys in the small window ---"
                )

                def on_space(vis):
                    self.navigation()
                    return False

                self.vis_1.register_key_callback(32, on_space)
                self.vis_2.register_key_callback(32, on_space)
                logging.info(" --- PRESS SPACE TO START navigation --- ")

            while (
                not self.navigation_finished
                and self.vis_1.poll_events()
                and self.vis_2.poll_events()
            ):
                with self._lock:
                    self.update_vis()
                time.sleep(1.0 / self.REFRESH_RATE)
        except KeyboardInterrupt:
            print("Shutting down...")
        except Exception:
            exit_code = 1
            traceback.print_exc()
        finally:
            self.close_outputs()
            if self.vis_1 is not None:
                self.vis_1.destroy_window()
            if self.vis_2 is not None:
                self.vis_2.destroy_window()

        return exit_code


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--target", type=str, required=True, help="Target object to search for"
    )

    p.add_argument("--mesh", type=str, required=True, help="Path to the mesh file")
    p.add_argument(
        "--config",
        type=str,
        default="config/hm3d_navigation.yaml",
        help="OpenFrontier configuration file",
    )
    p.add_argument(
        "--write_path", type=str, help="JSON file to write the ftmanager state"
    )
    p.add_argument(
        "--auto_start",
        action="store_true",
        default=False,
        help="Auto-start navigation loop",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI windows and record navigation_video.mp4",
    )
    p.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of navigation steps",
    )
    p.add_argument(
        "--max_time", type=int, default=3600, help="Maximum navigation time in seconds"
    )
    p.add_argument(
        "--vis_graph",
        action="store_true",
        default=False,
        help="Visualize the topological graph",
    )
    p.add_argument(
        "--save-images",
        "--save_images",
        dest="save_images",
        action="store_true",
        default=False,
        help="Save intermediate navigation images",
    )
    p.add_argument(
        "--record-video",
        "--record_video",
        dest="record_video",
        action="store_true",
        default=False,
        help="Record navigation_video.mp4",
    )
    p.add_argument(
        "--image_preview",
        action="store_true",
        default=False,
        help="Write the latest composed video frame to topdown.png",
    )
    p.add_argument(
        "--unet_weight",
        type=Path,
        default=Path("model_weights/rgbd_11cls.pth"),
        help="Path to UNet model weights",
    )
    p.add_argument(
        "--depth_source",
        type=str,
        default=NavigationApp.DEPTH_GT,
        choices=[
            NavigationApp.DEPTH_GT,
            NavigationApp.DEPTH_M3D,
            NavigationApp.DEPTH_UNIK3D,
        ],
        help="Depth source",
    )
    p.add_argument(
        "--log_level",
        "-ll",
        type=int,
        default=20,
        help="logging level (0=notset, 10=debug, 20=info...)",
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.headless:
        args.auto_start = True
        args.record_video = True
        args.image_preview = True

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=args.log_level,
    )
    print(f"Open3D version: {o3d.__version__}")

    app = NavigationApp(args)
    exit_code = app.run()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
