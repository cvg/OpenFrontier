import numpy as np
from typing import Optional


class DetectedObject:

    _next_id = 1

    def __init__(
        self,
    ):

        self.id = DetectedObject._next_id
        DetectedObject._next_id += 1
        self.label = ""
        self.mask = None
        self.image_index = 0
        self.depth = None
        self.viewpoint = None
        self.centroid = None
        self.is_valid = True
        self.verification_status = "unverified"
        self.frontier = None

    @classmethod
    def from_mask(
        cls,
        mask: dict,
        depth: np.ndarray,
        viewpoint: np.ndarray,
        intrinsic_mat: np.ndarray,
    ):
        obj = cls()
        obj.label = mask.get("label", "")
        obj.image_index = mask.get("image_index", 0)

        mask_array = np.array(mask["mask"])
        mask_depth = mask_array * depth

        obj.depth = depth
        obj.viewpoint = viewpoint
        obj.centroid = cls.get_object_location(mask_depth, viewpoint, intrinsic_mat)
        return obj

    @classmethod
    def get_object_location(
        cls,
        mask_depth: np.ndarray,
        viewpoint: np.ndarray,
        intrinsic_mat: np.ndarray,
    ) -> Optional[np.ndarray]:
        points_3d = cls.depth_to_point_cloud(mask_depth, intrinsic_mat)  # Nx3
        if points_3d.shape[0] == 0:
            return None
        # Transform to world coordinates
        ones = np.ones((points_3d.shape[0], 1))
        points_homogeneous = np.hstack((points_3d, ones))  # Nx4
        points_world = (viewpoint @ points_homogeneous.T).T  # Nx4
        centroid = np.mean(points_world[:, :3], axis=0)
        return centroid

    @classmethod
    def depth_to_point_cloud(
        cls, depth: np.ndarray, intrinsic_mat: np.ndarray
    ) -> np.ndarray:
        """
        Convert depth map to 3D point cloud in camera coordinates.
        """
        mask = depth > 0
        indices = np.array(np.nonzero(mask)).T  # Nx2 (v,u)
        if indices.shape[0] == 0:
            return np.zeros((0, 3))
        us = indices[:, 1]
        vs = indices[:, 0]
        zs = depth[vs, us]

        xs = (us - intrinsic_mat[0, 2]) * zs / intrinsic_mat[0, 0]
        ys = (vs - intrinsic_mat[1, 2]) * zs / intrinsic_mat[1, 1]

        points_3d = np.vstack((xs, ys, zs)).T  # Nx3
        return points_3d

    def to_dict(self) -> dict:
        """JSON-serializable summary for navigation state snapshots."""

        def _list(x):
            return None if x is None else np.asarray(x, dtype=float).tolist()

        return {
            "id": self.id,
            "label": self.label,
            "position": _list(self.centroid),
            "centroid": _list(self.centroid),
            "image_index": self.image_index,
            "viewpoint": _list(self.viewpoint),
            "is_valid": self.is_valid,
            "verification_status": self.verification_status,
            "frontier_id": (
                None if self.frontier is None else getattr(self.frontier, "id", None)
            ),
        }
