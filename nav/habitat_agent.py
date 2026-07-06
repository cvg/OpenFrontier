import habitat
import open3d as o3d

from .agent import NavigationAgent
from utils.transform import habitat_camera_intrinsic
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
from habitat_sim.agent import AgentState

from planner.habitat_planner import HabitatPlanner
from utils.transform import (
    from_habitat_position,
    from_habitat_rotation,
    to_habitat_position,
    to_habitat_rotation,
)
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2
from pathlib import Path


class HabitatAgent(NavigationAgent):
    def __init__(
        self,
        habitat_env: habitat.Env,
        args: dict,
        save_dir: str,
        openfrontier_config: dict,
        habitat_config,
        scene: str,
    ):
        episode = habitat_env.current_episode

        scene_node = habitat_env.sim.get_active_scene_graph().get_root_node()
        scene_bb = scene_node.cumulative_bb
        self.bbox = [
            scene_bb.min[0],
            scene_bb.max[0],
            -scene_bb.max[2],
            -scene_bb.min[2],
            scene_bb.min[1],
            scene_bb.max[1],
        ]

        self.planner = HabitatPlanner(sim=habitat_env.sim, params=openfrontier_config)

        self.habitat_env = habitat_env

        agent_state = habitat_env.sim.get_agent_state()
        cam_state = agent_state.sensor_states["rgb"]
        height_diff = cam_state.position[1] - agent_state.position[1]
        cam_to_agent = np.eye(4)
        cam_to_agent[2, 3] = height_diff

        target = episode.goals[0].object_category

        self.target = target
        self.scene = scene

        fix_view_level = False

        if scene == "6s7QHgap2fW":
            if target == "bed":
                target = "queen_bed"
        elif scene == "5cdEh9F2hJL":
            if target == "tv_monitor":
                target = "computer_monitor"
        elif scene == "MHPLjHsuG27":
            if target == "bed":
                target = "queen_bed"
            elif target == "plant":
                target = "flowers"
        elif scene == "Dd4bFSTQ8gi":
            if target == "chair":
                target = "brown chair"
        elif scene == "HY1NcmCgn3n":
            fix_view_level = True
            if target == "plant":
                target = "plant_special_no_purple_flowers"

        if target == "tv_monitor":
            target = "television_screen"
        elif target == "plant":
            target = "potted_plant"
        elif target == "sofa":
            target = "loveseat"

        super().__init__(
            args=args,
            target=target,
            planner=self.planner,
            save_dir=save_dir,
            cam_intrinsic=habitat_camera_intrinsic(habitat_config),
            bbox=self.bbox,
            get_rgbd=self.get_rgbd,
            get_cam_extrinsic=self.get_cam_extrinsic,
            move_fn=self.move_actions,
            rotate_fn=self.rotate_fn,
            cam_to_agent=cam_to_agent,
            fix_view_level=fix_view_level,
        )

        self.video_frames = []

    def draw_topdown(self, topdown_image):
        for pos in self.ft_manager._unreachable_positions:
            x, y = self.position_to_topdown(pos, topdown_image)
            cv2.circle(topdown_image, (x, y), 10, (50, 50, 50), -1)

        if self.detected_objects is not None and len(self.detected_objects) > 0:
            for obj in self.detected_objects:
                obj_pose = obj.centroid
                x, y = self.position_to_topdown(obj_pose, topdown_image)
                status = obj.verification_status
                if status == "unverified":
                    cv2.circle(topdown_image, (x, y), 10, (0, 0, 255), -1)
                elif status == "true_positive":
                    cv2.circle(topdown_image, (x, y), 10, (0, 255, 0), -1)
                elif status == "no_frontier":
                    cv2.line(
                        topdown_image, (x - 7, y - 7), (x + 7, y + 7), (50, 50, 50), 3
                    )
                    cv2.line(
                        topdown_image, (x - 7, y + 7), (x + 7, y - 7), (50, 50, 50), 3
                    )
                elif status == "false_positive":
                    cv2.line(
                        topdown_image, (x - 7, y - 7), (x + 7, y + 7), (0, 0, 255), 3
                    )
                    cv2.line(
                        topdown_image, (x - 7, y + 7), (x + 7, y - 7), (0, 0, 255), 3
                    )

        if self.ft_manager is not None and len(self.ft_manager.valid_frontiers) > 0:
            utilities = [
                ft.utility if ft.utility is not None else 0
                for ft in self.ft_manager.valid_frontiers
            ]
            min_utility = min(utilities)
            max_utility = max(utilities)
            utility_range = (
                max_utility - min_utility if max_utility > min_utility else 1.0
            )
            yellow_red_cmap = LinearSegmentedColormap.from_list(
                "yellow_red", ["#F6FF55", "#FF0202"]
            )

            goal_ft = self.ft_manager.goal_frontier

            for ft in self.ft_manager.all_frontiers:
                weight = (ft.utility - min_utility) / utility_range
                color_value = min(weight, 1.0)
                color = yellow_red_cmap(color_value)[:3]
                if goal_ft is not None and ft.id == goal_ft.id:
                    # draw a circle around the goal frontier
                    x, y = self.position_to_topdown(ft.pos3d, topdown_image)
                    cv2.circle(
                        topdown_image,
                        (x, y),
                        15,
                        (0, 255, 0),
                        8,
                    )

                ft_pose = ft.pose6d
                x, y, angle = self.pose_to_topdown(ft_pose, topdown_image)
                cv2.arrowedLine(
                    topdown_image,
                    (
                        int(x - 10 * np.cos(angle)),
                        int(y - 10 * np.sin(angle)),
                    ),
                    (
                        int(x + 10 * np.cos(angle)),
                        int(y + 10 * np.sin(angle)),
                    ),
                    (
                        int(color[2] * 255),
                        int(color[1] * 255),
                        int(color[0] * 255),
                    ),
                    6,
                    tipLength=1,
                )
                # write down the utility to two decimals
                if ft.utility < 1e5:
                    util = f"{ft.utility:.2f} | {int(ft.probability*100)}%"
                else:
                    util = "inf"

                cv2.putText(
                    topdown_image,
                    str(util),
                    (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return topdown_image

    def draw_custom(self, topdown_image):
        if self.path_to_go is not None and len(self.path_to_go) > 0:
            last_pose = self.path_to_go[-1]

            x, y, angle = self.pose_to_topdown(last_pose, topdown_image)
            cv2.arrowedLine(
                topdown_image,
                (
                    int(x - 10 * np.cos(angle)),
                    int(y - 10 * np.sin(angle)),
                ),
                (
                    int(x + 10 * np.cos(angle)),
                    int(y + 10 * np.sin(angle)),
                ),
                (255, 0, 0),
                6,
                tipLength=1,
            )
            for pose in self.path_to_go:
                position = pose[:3, 3]
                x, y = self.position_to_topdown(position, topdown_image)
                cv2.circle(topdown_image, (x, y), 3, (255, 0, 0), -1)

        return topdown_image

    def update_video(self) -> None:

        self.metrics = self.habitat_env.get_metrics()

        topdown_image = cv2.cvtColor(
            colorize_draw_agent_and_fit_to_height(self.metrics["top_down_map"], 1024),
            cv2.COLOR_BGR2RGB,
        )

        topdown_image = self.draw_custom(topdown_image)

        topdown_image = self.draw_topdown(topdown_image)

        topdown_image = cv2.putText(
            topdown_image,
            f"{self.scene} | {self.habitat_env.current_episode.episode_id} | {self.goal} | Step: {self.navigation_steps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        bgr, _ = self.get_rgbd()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Draw a black border around the rgb image
        rgb[:5, :, :] = 0
        rgb[-5:, :, :] = 0
        rgb[:, :5, :] = 0
        rgb[:, -5:, :] = 0

        # Write "Camera View" on the rgb image
        cv2.putText(
            rgb,
            "Camera View",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        som_image = self.last_som_img

        if som_image is not None:
            som_image = np.array(som_image)

            # resize som_image to match rgb
            som_image = cv2.resize(som_image, (rgb.shape[1], rgb.shape[0]))
            som_image_rgb = cv2.cvtColor(som_image, cv2.COLOR_BGR2RGB)

            # Draw a black border around the som image
            som_image_rgb[:5, :, :] = 0
            som_image_rgb[-5:, :, :] = 0
            som_image_rgb[:, :5, :] = 0
            som_image_rgb[:, -5:, :] = 0

            # Write "Frontiers" on the som image
            cv2.putText(
                som_image_rgb,
                "Frontiers",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            rgb = cv2.vconcat([rgb, som_image_rgb])

        topdown_shape = topdown_image.shape
        rgb_shape = rgb.shape

        # If topdown is larger, fill the height of rgb with black, else vice versa
        if topdown_shape[0] > rgb_shape[0]:
            diff = topdown_shape[0] - rgb_shape[0]
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            rgb = cv2.copyMakeBorder(
                rgb, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            diff = rgb_shape[0] - topdown_shape[0]
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
        combined_image = cv2.hconcat([rgb, topdown_image])

        cv2.imwrite("topdown.png", combined_image)

        self.video_frames.append(combined_image)

    def get_rgbd(self):
        obs = self.habitat_env.sim.get_sensor_observations()
        img = obs["rgb"]
        depth = obs["depth"]

        if np.max(img) <= 1:
            img = (img * 255).astype(np.uint8)

        if img.shape[2] == 4:
            img = img[:, :, :3]

        return img, depth

    def get_cam_extrinsic(self):
        position_hab = (
            self.habitat_env.sim.get_agent_state().sensor_states["rgb"].position
        )
        rotation_hab = (
            self.habitat_env.sim.get_agent_state().sensor_states["rgb"].rotation
        )
        T = np.eye(4)
        T[:3, :3] = from_habitat_rotation(rotation_hab)
        T[:3, 3] = from_habitat_position(position_hab)
        return np.linalg.inv(T)

    def move_actions(self):
        actions = self.planner.actions
        if actions is None or len(actions) == 0 or actions[0] is None:
            print("No action from planner, stopping.")
            return
        action = actions.pop(0)
        self.habitat_env.step(action)

    def move_state(self):
        next_W_T_C = self.next_W_T_C

        position = to_habitat_position(self.get_agent_pose(next_W_T_C)[:3, 3])
        rotation = to_habitat_rotation(next_W_T_C[:3, :3])

        self.habitat_env.sim.get_agent(0).set_state(
            AgentState(
                position=position,
                rotation=rotation,
            )
        )

    def rotate_fn(self):
        self.habitat_env.step("turn_right")

    def save_trajectory(self, dir="output_objnav/"):
        import imageio
        import os

        os.makedirs(dir, exist_ok=True)

        video_dir = Path.joinpath(dir, "metrics.mp4")

        metric_writer = imageio.get_writer(video_dir, fps=15)
        for met in self.video_frames:
            metric_writer.append_data(cv2.cvtColor(met, cv2.COLOR_BGR2RGB))

        metric_writer.close()

    def position_to_topdown(self, position, topdown_image):
        map_size_x, map_size_y, _ = topdown_image.shape
        x_box = [self.bbox[0], self.bbox[1]]
        y_box = [self.bbox[2], self.bbox[3]]
        map_width = x_box[1] - x_box[0]
        map_height = y_box[1] - y_box[0]
        x_ratio = (position[0] - x_box[0]) / (map_width)
        y_ratio = (position[1] - y_box[0]) / (map_height)

        if map_width > map_height:
            x_pixel = int((x_ratio) * map_size_y)
            y_pixel = int((1 - y_ratio) * map_size_x)
        else:
            y_pixel = int((1 - x_ratio) * map_size_x)
            x_pixel = int((1 - y_ratio) * map_size_y)

        return x_pixel, y_pixel

    def pose_to_topdown(self, pose, topdown_image):
        rotation = pose[:3, :3]
        position = pose[:3, 3]
        x_axis = rotation[:, 0]
        angle = np.arctan2(x_axis[1], x_axis[0])

        map_size_x, map_size_y, _ = topdown_image.shape
        x_box = [self.bbox[0], self.bbox[1]]
        y_box = [self.bbox[2], self.bbox[3]]
        map_width = x_box[1] - x_box[0]
        map_height = y_box[1] - y_box[0]
        x_ratio = (position[0] - x_box[0]) / (map_width)
        y_ratio = (position[1] - y_box[0]) / (map_height)

        if map_width > map_height:
            x_pixel = int((x_ratio) * map_size_y)
            y_pixel = int((1 - y_ratio) * map_size_x)
            angle = -angle - np.pi / 2
        else:
            angle = -angle + np.pi
            y_pixel = int((1 - x_ratio) * map_size_x)
            x_pixel = int((1 - y_ratio) * map_size_y)

        return x_pixel, y_pixel, angle
