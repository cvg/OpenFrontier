from scipy.spatial import KDTree
import numpy as np

import heapq


import numpy as np
from scipy.spatial import KDTree
import heapq
import math


def astar(
    nav_kdt: KDTree, start: np.ndarray, goal: np.ndarray, neighbor_radius: float = 0.30
):
    """
    A* pathfinding on a 2D navigable KDTree.

    nav_kdt: KDTree of navigable points (N,2)
    start: (2,) array, start position
    goal: (2,) array, goal position
    neighbor_radius: radius to consider neighbors

    Returns: Nx2 array of path points, or None if no path found
    """
    nav_points = nav_kdt.data
    N = len(nav_points)

    neighbor_radius = neighbor_radius * 1.9

    if N == 0:
        return None

    # Precompute neighbors for each point
    neighbors = [nav_kdt.query_ball_point(p, neighbor_radius) for p in nav_points]

    # Map start and goal to nearest navigable points
    _, start_idx = nav_kdt.query(start)
    _, goal_idx = nav_kdt.query(goal)

    open_set = []
    heapq.heappush(
        open_set,
        (0 + np.linalg.norm(nav_points[start_idx] - nav_points[goal_idx]), start_idx),
    )

    came_from = {}
    g_score = {i: float("inf") for i in range(N)}
    g_score[start_idx] = 0
    f_score = {i: float("inf") for i in range(N)}
    f_score[start_idx] = np.linalg.norm(nav_points[start_idx] - nav_points[goal_idx])

    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal_idx:
            # Reconstruct path
            path = [nav_points[current]]
            while current in came_from:
                current = came_from[current]
                path.append(nav_points[current])
            return np.array(path[::-1])

        for neighbor in neighbors[current]:
            tentative_g = g_score[current] + np.linalg.norm(
                nav_points[current] - nav_points[neighbor]
            )
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + np.linalg.norm(
                    nav_points[neighbor] - nav_points[goal_idx]
                )
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # no path found


def heading(p_from, p_to):
    v = p_to - p_from
    return np.degrees(np.arctan2(v[1], v[0])) % 360


def quantize_heading(deg):
    return round(deg / 30) * 30 % 360  # 12 possible headings


def astar_min_turns(
    nav_kdt: KDTree, start: np.ndarray, goal: np.ndarray, neighbor_radius: float = 0.30
):

    nav = nav_kdt.data
    N = len(nav)
    if N == 0:
        return None

    # Precompute neighbors
    neighbor_radius = neighbor_radius * 1.9
    neighbors = [nav_kdt.query_ball_point(p, neighbor_radius) for p in nav]

    _, start_idx = nav_kdt.query(start)
    _, goal_idx = nav_kdt.query(goal)

    # Cost weights (very important)
    W_STEP = 10000
    W_TURN = 100
    W_DIST = 1

    # A* states are (node_idx, heading)
    start_heading = quantize_heading(heading(start, nav[start_idx]))

    open_heap = []
    heapq.heappush(open_heap, (0, start_idx, start_heading))

    came_from = {}

    # g[(idx,heading)] stores cost
    g = {}
    g[(start_idx, start_heading)] = 0

    visited = set()

    while open_heap:
        _, cur_idx, cur_hd = heapq.heappop(open_heap)
        state = (cur_idx, cur_hd)

        if state in visited:
            continue
        visited.add(state)

        if cur_idx == goal_idx:
            # reconstruct path ignoring headings
            path = [nav[cur_idx]]
            key = state
            while key in came_from:
                key = came_from[key][0]  # extract previous state
                path.append(nav[key[0]])
            return np.array(path[::-1])

        for nxt_idx in neighbors[cur_idx]:
            nxt_hd_raw = heading(nav[cur_idx], nav[nxt_idx])
            nxt_hd = quantize_heading(nxt_hd_raw)

            step_cost = 1  # always 1 per move
            turn_cost = 1 if nxt_hd != cur_hd else 0
            dist_cost = np.linalg.norm(nav[cur_idx] - nav[nxt_idx])

            new_cost = (
                g[state] + step_cost * W_STEP + turn_cost * W_TURN + dist_cost * W_DIST
            )

            nxt_state = (nxt_idx, nxt_hd)

            if nxt_state not in g or new_cost < g[nxt_state]:
                g[nxt_state] = new_cost

                h = np.linalg.norm(nav[nxt_idx] - nav[goal_idx]) * W_DIST
                f = new_cost + h

                came_from[nxt_state] = (state,)

                heapq.heappush(open_heap, (f, nxt_idx, nxt_hd))

    return None
