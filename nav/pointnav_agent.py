import habitat
import open3d as o3d

from nav.habitat_agent import HabitatAgent
from utils.transform import habitat_camera_intrinsic
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

from planner.pointnav_planner import PointnavPlanner

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2
from pathlib import Path


class PointnavAgent(HabitatAgent):
    def __init__(
        self,
        habitat_env: habitat.Env,
        args: dict,
        save_dir: str,
        openfrontier_config: dict,
        habitat_config,
        scene: str,
    ):
        super().__init__(
            habitat_env, args, save_dir, openfrontier_config, habitat_config, scene
        )

        self.planner = PointnavPlanner(params=openfrontier_config)
        self.move_simulation = self.move_actions

    def draw_custom(self, topdown_image: np.ndarray) -> np.ndarray:
        if self.planner.goal_pose is not None:
            x, y = self.position_to_topdown(
                self.planner.goal_pose[:3, 3], topdown_image
            )

            green_yellow_red_cmap = LinearSegmentedColormap.from_list(
                "green_yellow_red", ["#00FF00", "#FFFF00", "#FF0000"]
            )

            max_forward_heat = self.planner.MAX_FORWARD_HEAT
            current_forward_heat = self.planner.forward_failure_heat
            forward_heat_ratio = min(current_forward_heat / max_forward_heat, 1.0)

            max_rotation_heat = self.planner.MAX_ROTATION_HEAT
            current_rotation_heat = self.planner.rotation_heat
            rotation_heat_ratio = min(current_rotation_heat / max_rotation_heat, 1.0)

            heat = max(forward_heat_ratio, rotation_heat_ratio)

            color = green_yellow_red_cmap(heat)[:3]

            cv2.circle(
                topdown_image,
                (x, y),
                15,
                (
                    int(color[2] * 255),
                    int(color[1] * 255),
                    int(color[0] * 255),
                ),
                8,
            )

        rho, theta = self.planner.rho, self.planner.theta
        theta += np.pi / 2
        if self.planner.start_pose is not None:
            pose6d = self.planner.start_pose
            start = pose6d[:3, 3]
            calc_end = (
                start[0] + rho * np.cos(theta + np.arctan2(pose6d[1, 0], pose6d[0, 0])),
                start[1] + rho * np.sin(theta + np.arctan2(pose6d[1, 0], pose6d[0, 0])),
                start[2],
            )
            x_start, y_start = self.position_to_topdown(start, topdown_image)
            x_end, y_end = self.position_to_topdown(calc_end, topdown_image)
            cv2.line(
                topdown_image,
                (x_start, y_start),
                (x_end, y_end),
                (255, 0, 0),
                4,
            )

        return topdown_image

    def move_actions(self):
        action = self.planner.next_action()
        if action is not None:
            self.habitat_env.step(action)
        else:
            print("No action from planner!")
