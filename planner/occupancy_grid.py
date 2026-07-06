import os
import numpy as np
from typing import Dict, List, Optional, Sequence, Union
from scipy.spatial import KDTree
import time


class OccupancyGrid:
    """
    3D occupancy-grid
    """

    def __init__(self, params: Optional[Dict] = None) -> None:
        params = params or {}

        self._free_kdt: Optional[KDTree] = None
        self._occ_kdt: Optional[KDTree] = None

        self._use_freegrid: bool = bool(params.get("use_free_grid", False))
        self._use_occgrid: bool = bool(params.get("use_occ_grid", False))
        self._min_dist2occ: float = float(params.get("min_dist2occ", 0.1))
        self._max_dist2free: float = float(params.get("max_dist2free", 0.1))
        self.logging_file: str = "navigation_log.txt"

    @property
    def use_freegrid(self) -> bool:
        return self._use_freegrid

    @use_freegrid.setter
    def use_freegrid(self, val: bool) -> None:
        self._use_freegrid = bool(val)

    @property
    def use_occgrid(self) -> bool:
        return self._use_occgrid

    @use_occgrid.setter
    def use_occgrid(self, val: bool) -> None:
        self._use_occgrid = bool(val)

    @property
    def min_dist2occ(self) -> float:
        return self._min_dist2occ

    @min_dist2occ.setter
    def min_dist2occ(self, val: float) -> None:
        self._min_dist2occ = float(val)

    @property
    def max_dist2free(self) -> float:
        return self._max_dist2free

    @max_dist2free.setter
    def max_dist2free(self, val: float) -> None:
        self._max_dist2free = float(val)

    def isfree(self, point: Sequence[float]) -> bool:
        """
        True if sample is within max_dist2free of free voxelgrid (if KDTree set).
        If no free KDTree is set, returns True.
        """
        if self._free_kdt is None:
            return True
        dist, _ = self._free_kdt.query(point)
        return dist < self._max_dist2free

    def isoccupied(self, point: Sequence[float]) -> bool:
        """
        True if sample is within min_dist2occ of occupied voxelgrid (if KDTree set).
        If no occ KDTree is set, returns False.
        """
        if self._occ_kdt is None:
            return False
        dist, _ = self._occ_kdt.query(point)
        return dist < self._min_dist2occ

    def is_state_valid(self, state: np.ndarray) -> bool:
        p = np.array([state[0], state[1], state[2]], dtype=float)

        in_free = True
        away_from_occ = True

        if self._use_freegrid and self._free_kdt is not None:
            dist, _ = self._free_kdt.query(p)
            in_free = dist < self._max_dist2free

        if self._use_occgrid and self._occ_kdt is not None:
            dist, _ = self._occ_kdt.query(p)
            away_from_occ = dist > self._min_dist2occ

        return in_free and away_from_occ

    def update_space(
        self, free_vx: Optional[np.ndarray] = None, occ_vx: Optional[np.ndarray] = None
    ) -> None:
        if free_vx is not None:
            free_vx = np.asarray(free_vx, dtype=float)
            assert (
                free_vx.ndim == 2 and free_vx.shape[1] == 3
            ), "free_vx should be (N,3)"
            self._free_kdt = KDTree(free_vx)

        if occ_vx is not None:
            occ_vx = np.asarray(occ_vx, dtype=float)
            assert occ_vx.ndim == 2 and occ_vx.shape[1] == 3, "occ_vx should be (M,3)"
            self._occ_kdt = KDTree(occ_vx)

    def snap_to_free(self, point: Sequence[float]) -> Optional[np.ndarray]:
        if (not self.isfree(point)) or self.isoccupied(point):
            if self._free_kdt is None:
                print("No free KDTree available; cannot adjust start.")
                return point

            # k nearest (bounded by dataset size)
            k = min(500, self._free_kdt.n)
            dists, idxs = self._free_kdt.query(point, k=k)

            # Ensure arrays for uniform handling
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)

            original_point = np.asarray(point, dtype=float)

            for d, idx in zip(dists, idxs):
                if d < 3.0 * self._max_dist2free:
                    candidate = self._free_kdt.data[idx]
                    if not self.isoccupied(candidate):
                        point = candidate.astype(float, copy=True)
                        return point
            return original_point

        return np.asarray(point, dtype=float)

    def log(self, level: str, file: str, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        folder = os.path.dirname(file).split("/")
        folder = f"{folder[-2]}/{folder[-1]}" if len(folder) >= 2 else folder[0]
        message = f"[{folder}]\t[{timestamp}]\t[PLANNER]\t[{level.upper()}]\t{message}"
        with open(file, "a") as f:
            print(message)
            f.write(message + "\n")
