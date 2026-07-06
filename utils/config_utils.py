import os
from pathlib import Path

import habitat
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.environ.get("OPENFRONTIER_DATA_ROOT", PROJECT_ROOT / "data")
).expanduser()
HABITAT_LAB_ROOT = Path(
    os.environ.get("HABITAT_LAB_ROOT", PROJECT_ROOT / "habitat-lab")
).expanduser()

HM3D_CONFIG_PATH = os.environ.get(
    "HABITAT_HM3D_CONFIG",
    str(
        HABITAT_LAB_ROOT
        / "habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
    ),
)

MP3D_CONFIG_PATH = os.environ.get(
    "HABITAT_MP3D_CONFIG",
    str(
        HABITAT_LAB_ROOT
        / "habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml"
    ),
)

DATA_PATH = str(DATA_ROOT) + os.sep


def hm3d_config(path: str = HM3D_CONFIG_PATH, stage: str = "val", episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = DATA_PATH + "scene_datasets/"
        habitat_config.habitat.dataset.data_path = (
            DATA_PATH + "objectnav_hm3d_v2/val/content/{split}.json.gz"
        )
        habitat_config.habitat.simulator.scene_dataset = (
            DATA_PATH
            + "scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        )
        habitat_config.habitat.environment.iterator_options.num_episode_sample = (
            episodes
        )
        habitat_config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = (
            3.5
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = (
            False
        )
        habitat_config.habitat.task.measurements.success.success_distance = 1.0
    return habitat_config

def mp3d_config(path: str = MP3D_CONFIG_PATH, stage: str = "val", episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = DATA_PATH + "scene_datasets/"
        habitat_config.habitat.dataset.data_path = (
            DATA_PATH + "objectnav_mp3d_v1/val/content/{split}.json.gz"
        )
        habitat_config.habitat.simulator.scene_dataset = (
            DATA_PATH
            + "scene_datasets/mp3d/mp3d.scene_dataset_config.json"
        )
        habitat_config.habitat.environment.iterator_options.num_episode_sample = (
            episodes
        )
        habitat_config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = (
            3.5
        )
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = (
            False
        )
        habitat_config.habitat.task.measurements.success.success_distance = 1.0
    return habitat_config


def ovon_config(path: str = HM3D_CONFIG_PATH, stage: str = "val_unseen", episodes=200):
    import ovon.dataset
    import ovon.task.simulator
    import ovon.task.sensors
    import ovon.measurements.nav
    from ovon.config import (
        NavmeshSettings,
        OVONDistanceToGoalConfig,
        ClipObjectGoalSensorConfig,
        OVONObjectGoalIDMeasurementConfig,
    )

    habitat_config = hm3d_config(path, stage, episodes)

    with read_write(habitat_config):
        habitat_config.habitat.dataset.data_path = (
            DATA_PATH + "ovon/val_unseen/content/{split}.json.gz"
        )
        habitat_config.habitat.dataset.type = "OVON-v1"
        habitat_config.habitat.simulator.type = "OVONSim-v0"
        habitat_config.habitat.simulator.navmesh_settings = NavmeshSettings()
        habitat_config.habitat.task.lab_sensors.update(
            {"clip_objectgoal_sensor": ClipObjectGoalSensorConfig()}
        )

        habitat_config.habitat.task.measurements.success.success_distance = 0.25
        habitat_config.habitat.task.measurements.distance_to_goal.type = (
            "OVONDistanceToGoal"
        )

        if "objectgoal_sensor" in habitat_config.habitat.task.lab_sensors:
            del habitat_config.habitat.task.lab_sensors["objectgoal_sensor"]

        habitat_config.habitat.task.measurements.update(
            {"ovon_object_goal_id": OVONObjectGoalIDMeasurementConfig()}
        )

        habitat_config.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=False,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    return habitat_config
