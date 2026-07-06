from utils.server_wrapper import ServerMixin, host_agent

from typing import Dict, Optional
from planner.occ_rrt_point3d import OccupancyGrid3DPathPlanner
from utils.frontier_utils import read_config_yaml
import numpy as np

if __name__ == "__main__":

    class PlannerServer(ServerMixin):

        def __init__(self, params: Optional[Dict] = None) -> None:
            super().__init__()
            self.params = params or {}

            self.config = read_config_yaml("config/hm3d_navigation.yaml")

            self.planner = OccupancyGrid3DPathPlanner(self.params)

        def process_payload(self, payload: dict) -> dict:
            function = payload["function"]
            data = payload["data"]
            print(f"Received request for function: {function}")
            print(f"Data keys: {list(data.keys())}")

            try:
                if function == "reset":
                    self.planner = OccupancyGrid3DPathPlanner(self.params)
                    return {"result": "success"}

                if function == "get_use_freegrid":
                    return self.planner.use_freegrid

                elif function == "set_use_freegrid":
                    self.planner.use_freegrid = bool(data["val"])
                    return {"result": "success"}

                elif function == "get_use_occgrid":
                    return {"result": "success", "value": self.planner.use_occgrid}

                elif function == "set_use_occgrid":
                    self.planner.use_occgrid = bool(data["val"])
                    return {"result": "success"}

                elif function == "get_min_dist2occ":
                    return {"result": "success", "value": self.planner.min_dist2occ}

                elif function == "set_min_dist2occ":
                    self.planner.min_dist2occ = float(data["val"])
                    return {"result": "success"}

                elif function == "get_max_dist2free":
                    return {"result": "success", "value": self.planner.max_dist2free}

                elif function == "set_max_dist2free":
                    self.planner.max_dist2free = float(data["val"])
                    return {"result": "success"}

                elif function == "isfree":
                    return {
                        "result": "success",
                        "value": self.planner.isfree(data["point"]),
                    }

                elif function == "isoccupied":
                    return {
                        "result": "success",
                        "value": self.planner.isoccupied(data["point"]),
                    }

                elif function == "is_state_valid":
                    return {
                        "result": "success",
                        "value": self.planner.is_state_valid(data["state"]),
                    }

                elif function == "get_bounds":
                    return {"result": "success", "value": self.planner.get_bounds()}

                elif function == "set_bounds":
                    self.planner.set_bounds(data["bounds"])
                    return {"result": "success"}

                elif function == "update_space":
                    self.planner.update_space(
                        free_vx=data.get("free_vx"),
                        occ_vx=data.get("occ_vx"),
                    )
                    return {"result": "success"}

                elif function == "update_start_goal":
                    result = self.planner.update_start_goal(
                        start=data.get("start"),
                        goal=data.get("goal"),
                    )
                    return {"result": "success", "value": result}

                elif function == "get_start_goal":
                    return {"result": "success", "value": self.planner.get_start_goal()}

                elif function == "get_motion_check_resolution":
                    return {
                        "result": "success",
                        "value": self.planner.get_motion_check_resolution(),
                    }

                elif function == "solve":
                    method = data.get("method", "rrtstar")
                    time_limit = float(data.get("time_limit", 1.0))
                    return {
                        "result": "success",
                        "value": self.planner.solve(
                            method=method, time_limit=time_limit
                        ),
                    }

                elif function == "interpolate_path":
                    num_interp_points = int(data.get("num_interp_points", 10))
                    external_waypoints = data.get("external_waypoints", None)

                    path = self.planner.interpolate_path(
                        num_interp_points=num_interp_points,
                        external_waypoints=external_waypoints,
                    )

                    for point in path:
                        if isinstance(point["quat"], np.ndarray):
                            point["quat"] = point["quat"].tolist()
                        if isinstance(point["pos"], np.ndarray):
                            point["pos"] = point["pos"].tolist()

                    return {
                        "result": "success",
                        "value": path,
                    }

                elif function == "get_solution_path":
                    return_type = data.get("return_type", "dict")
                    path = self.planner.get_solution_path(return_type=return_type)

                    if return_type == "dict":
                        for point in path:
                            if isinstance(point["quat"], np.ndarray):
                                point["quat"] = point["quat"].tolist()
                            if isinstance(point["pos"], np.ndarray):
                                point["pos"] = point["pos"].tolist()
                    else:
                        path = [pt.tolist() for pt in path]

                    return {
                        "result": "success",
                        "value": path,
                    }

                else:
                    return {
                        "result": "error",
                        "message": f"Unknown function: {function}",
                    }
            except Exception as e:
                return {
                    "result": "error",
                    "message": str(e),
                }

    server = PlannerServer()
    print("Agent loaded!")
    host_agent(server, name="planner", port=12184)
