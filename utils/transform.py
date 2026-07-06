import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from habitat_sim.utils.common import quat_from_coeffs


def habitat_camera_intrinsic(config):
    assert (
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width
        == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width
    ), "The configuration of the depth camera should be the same as rgb camera."
    assert (
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height
        == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height
    ), "The configuration of the depth camera should be the same as rgb camera."
    assert (
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov
        == config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov
    ), "The configuration of the depth camera should be the same as rgb camera."
    width = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width
    height = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height
    hfov = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov
    xc = (width - 1.0) / 2.0
    zc = (height - 1.0) / 2.0
    f = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))

    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(width, height, f, f, xc, zc)

    return intrinsics


def from_habitat_position(position):
    return np.array([position[0], -position[2], position[1]])


def to_habitat_position(position):
    return np.array([position[0], position[2], -position[1]])


def from_habitat_rotation(rotation):
    rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
    rotation_matrix = R.from_quat(rotation).as_matrix()
    # Reflect Y and Z axes
    reflect = np.diag([1, -1, -1])
    rotation_matrix = rotation_matrix @ reflect
    correction = R.from_euler("x", 90, degrees=True).as_matrix()
    rotation_matrix = correction @ rotation_matrix
    return rotation_matrix


def to_habitat_rotation(matrix):
    # Inverse of the correction
    correction_inv = R.from_euler("x", -90, degrees=True).as_matrix()
    rotation_matrix = correction_inv @ matrix
    # Reflect Y and Z axes
    reflect = np.diag([1, -1, -1])
    rotation_matrix = rotation_matrix @ reflect
    r = R.from_matrix(rotation_matrix)
    x, y, z, w = r.as_quat()
    quat = quat_from_coeffs([x, y, z, w])
    return quat
