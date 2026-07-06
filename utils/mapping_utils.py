import torch
import torch.nn.functional as F
import math
import numpy as np
from scipy.spatial import KDTree
from collections import deque
from scipy.ndimage import gaussian_filter, sobel
import cv2


def scatter_reduce_amin(src, index, dim=-1, out=None):
    if src.numel() == 0:
        # If source is empty, return out initialized to +inf or empty
        if out is None:
            size = list(src.shape)
            if index.numel() > 0:
                size[dim] = int(index.max()) + 1
            else:
                size[dim] = 0
            return torch.full(size, float("inf"), dtype=src.dtype, device=src.device)
        return out

    # determine output shape
    if out is None:
        size = list(src.shape)
        size[dim] = int(index.max()) + 1 if index.numel() > 0 else 0
        out = torch.full(size, float("inf"), dtype=src.dtype, device=src.device)

    # align dimension
    if index.shape != src.shape:
        index = index.expand_as(src)

    # transpose to bring dim to front
    flat_out = out.transpose(dim, 0).contiguous()
    flat_src = src.transpose(dim, 0).contiguous()
    flat_idx = index.transpose(dim, 0).contiguous()

    # manual scatter amin
    for i in range(flat_src.shape[0]):
        if flat_idx.numel() > 0:
            flat_out[flat_idx[i]] = torch.minimum(flat_out[flat_idx[i]], flat_src[i])

    return flat_out.transpose(dim, 0)


def rasterize_voxel_depth(
    vx_centers,
    radius,
    K,
    T_cams_world,
    image_size,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Fast batched voxel rasterization using scatter_reduce and physically-based radius expansion.

    Args:
        vx_centers (np.ndarray): (N, 3) world-space voxel centers
        radius (float): 3D voxel radius (in meters)
        K (np.ndarray): (3, 3) camera intrinsics
        T_cams_world (np.ndarray): (B, 4, 4) camera extrinsics (camera_T_world)
        image_size (tuple): (H, W)
        device (str): 'cuda' or 'cpu'

    Returns:
        depth_imgs: (B, H, W) numpy array of depth images
    """
    B = T_cams_world.shape[0]
    H, W = image_size

    # Early return if no voxels
    if vx_centers is None or len(vx_centers) == 0:
        return np.full((B, H, W), 100, dtype=np.float32)

    N = vx_centers.shape[0]

    # Prepare tensors
    vx_centers = torch.tensor(vx_centers, dtype=torch.float32, device=device)  # (N, 3)
    K = torch.tensor(K, dtype=torch.float32, device=device)
    T = torch.tensor(T_cams_world, dtype=torch.float32, device=device)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Homogeneous coordinates
    ones = torch.ones((N, 1), device=device)
    vx_centers_h = (
        torch.cat([vx_centers, ones], dim=1).unsqueeze(0).expand(B, N, 4)
    )  # (B, N, 4)

    # Transform world to camera frame
    xyz_cam = torch.bmm(vx_centers_h, T.transpose(1, 2))[:, :, :3]  # (B, N, 3)
    x, y, z = xyz_cam.unbind(-1)  # (B, N)

    # Keep only points in front of the camera
    valid = z > 0.1
    cam_idx, pt_idx = valid.nonzero(as_tuple=True)
    x, y, z = x[valid], y[valid], z[valid]

    # Project to 2D
    u = (fx * x / z + cx).round().long()
    v = (fy * y / z + cy).round().long()

    # Filter projections within image bounds
    valid_u = (u >= 0) & (u < W)
    valid_v = (v >= 0) & (v < H)
    valid_proj = valid_u & valid_v

    cam_idx = cam_idx[valid_proj]
    u = u[valid_proj]
    v = v[valid_proj]
    z = z[valid_proj]

    # Allocate full flattened image buffer for all B batches
    depth_flat = torch.full((B * H * W,), float("inf"), device=device)

    # Offset each pixel index by batch ID
    offsets = cam_idx * (H * W)
    pixel_idx_global = (v * W + u) + offsets

    # Perform a single scatter_reduce over all batches

    if hasattr(depth_flat, "scatter_reduce_"):
        depth_flat.scatter_reduce_(
            dim=0, index=pixel_idx_global, src=z, reduce="amin", include_self=True
        )
    else:
        # Fallback for older PyTorch versions without scatter_reduce_
        depth_flat = scatter_reduce_amin(
            src=z, index=pixel_idx_global, dim=0, out=depth_flat
        )

    # Reshape back to (B, H, W)
    depth_imgs = depth_flat.view(B, H, W)
    depth_imgs[depth_imgs == float("inf")] = 0.0

    # Compute Physical Radius in Pixels
    if z.numel() == 0:
        # nothing was projected -> return current image as-is
        return (
            depth_imgs.cpu().numpy()
            if device.startswith("cuda")
            else depth_imgs.numpy()
        )

    projected_r_pix = fx * radius / z.clamp(min=1e-4)
    max_r_pix = projected_r_pix.max().item()
    kernel_size = int(2 * math.ceil(max_r_pix) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius_px = kernel_size // 2

    # Radius Expansion with Min-Pooling
    assert radius_px > 0
    depth = depth_imgs.clone()
    depth[depth == 0.0] = float("inf")

    depth_inv = -depth.unsqueeze(1)  # (B, 1, H, W)
    depth_expanded_inv = F.max_pool2d(
        depth_inv, kernel_size=kernel_size, stride=1, padding=radius_px
    )
    depth_expanded = -depth_expanded_inv.squeeze(1)
    depth_expanded[depth_expanded == float("inf")] = 0.0

    return (
        depth_expanded.cpu().numpy()
        if device.startswith("cuda")
        else depth_expanded.numpy()
    )


def render_voxel_depth(occ_pts, camera_extrinsics, render_params):
    """
    Render depth images from the occupied voxels
    camera_extrinsics: (B, 4, 4) numpy array of camera extrinsics (world to camera)
    """
    depth_imgs = rasterize_voxel_depth(
        occ_pts,
        radius=render_params["radius"],
        K=render_params["K"],
        T_cams_world=camera_extrinsics,
        image_size=(render_params["H"], render_params["W"]),
    )
    return depth_imgs


def select_visible_points(points, K, T, img_size, max_depth=np.inf, min_depth_map=None):
    """
    Project 3D points to 2D image coordinates using camera intrinsics and extrinsics,
    returning only those that are visible and optionally closer than a depth map.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 3)), np.array([])

    points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)
    valid_points = np.ones(points.shape[0], dtype=bool)

    # Transform to camera coordinates
    points_cam = (T @ points_h.T).T[:, :3]  # (N, 3)
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

    # Filter by depth range
    valid_points &= (z > 0) & (z <= max_depth)

    # Project to image plane
    u = np.round(K[0, 0] * x / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * y / z + K[1, 2]).astype(int)

    # Filter out-of-bounds pixels
    in_bounds = (u >= 0) & (u < img_size[1]) & (v >= 0) & (v < img_size[0])
    valid_points &= in_bounds

    # Filter based on depth map
    if min_depth_map is not None:
        depth_map_values = np.full(points.shape[0], np.nan)
        depth_map_values[in_bounds] = min_depth_map[v[in_bounds], u[in_bounds]]
        visible_mask = np.isnan(depth_map_values) | (z <= depth_map_values)
        valid_points &= visible_mask

    valid_indices = np.where(valid_points)[0]
    return points[valid_indices], valid_indices


def compute_visible_voxels(
    K, T, img_size, voxel_size=0.1, max_depth=np.inf, min_depth_map=None
):
    """
    Calculate the maximum number of voxels that can be visible in an image based on camera parameters and optional depth map.
    """

    # create voxel grid in camera coordinates
    z = np.arange(voxel_size, max_depth, voxel_size)
    x_range = int(0.8 * (max_depth * (img_size[1] / 2)) / K[0, 0])
    y_range = int(0.8 * (max_depth * (img_size[0] / 2)) / K[1, 1])
    x = np.arange(-x_range, x_range + voxel_size, voxel_size)
    y = np.arange(-y_range, y_range + voxel_size, voxel_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    voxel_centers = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  # (N, 3)

    # transform voxel centers to world coordinates
    voxel_centers_h = np.hstack((voxel_centers, np.ones((voxel_centers.shape[0], 1))))
    voxel_centers_world = (np.linalg.inv(T) @ voxel_centers_h.T).T[:, :3]

    # handle strict zero depth
    if min_depth_map is not None:
        min_depth_map = np.array(min_depth_map, dtype=np.float32)
        min_depth_map[abs(min_depth_map) < 1e-3] = max_depth

    # project to image and get visible ones
    valid_voxels, valid_indices = select_visible_points(
        voxel_centers_world,
        K=K,
        T=T,
        img_size=img_size,
        max_depth=max_depth,
        min_depth_map=min_depth_map,
    )

    return voxel_centers_world[valid_indices], valid_voxels.shape[0]


def voxel_floor_fov_from_WTC(
    W_T_C: np.ndarray,
    voxel_size: float,
    max_radius: float,
    floor_level: float,
    intrinsics,
):
    C_pos = W_T_C[:3, 3].copy()
    R_W_C = W_T_C[:3, :3]

    # Floor-level camera position
    C_pos[2] = floor_level

    # Camera forward projected on floor
    forward = R_W_C @ np.array([0, 0, 1])
    forward[2] = 0
    forward /= np.linalg.norm(forward)

    # Horizontal FOV
    fx = intrinsics.intrinsic_matrix[0, 0]
    img_width = intrinsics.width
    fov_x = 2 * np.arctan(img_width / (2 * fx))

    # Determine grid bounds in local camera frame
    y_min, y_max = 0, max_radius
    x_extent = max_radius * np.tan(fov_x / 2)
    x_min, x_max = -x_extent, x_extent

    nx = int(np.ceil((x_max - x_min) / voxel_size))
    ny = int(np.ceil((y_max - y_min) / voxel_size))

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    local_points = np.array([[x, y, 0] for y in ys for x in xs])

    # Filter points inside cone
    angles = np.arctan2(local_points[:, 0], local_points[:, 1])
    distances = np.linalg.norm(local_points[:, :2], axis=1)
    mask = (np.abs(angles) <= fov_x / 2) & (distances <= max_radius)
    local_points = local_points[mask]

    # Rotate to world frame
    angle = -np.arctan2(forward[0], forward[1])
    R_align = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    world_points = (R_align @ local_points.T).T + C_pos

    # --- Snap to voxel grid ---
    world_points[:, :2] = np.floor(world_points[:, :2] / voxel_size) * voxel_size

    # Remove duplicates after snapping
    _, idx = np.unique(world_points[:, :2], axis=0, return_index=True)
    world_points = world_points[idx]

    # add 4 voxel sqquare around the camera
    cam_x = np.floor(C_pos[0] / voxel_size) * voxel_size
    cam_y = np.floor(C_pos[1] / voxel_size) * voxel_size
    for dx in [-voxel_size, 0, voxel_size]:
        for dy in [-voxel_size, 0, voxel_size]:
            world_points = np.vstack(
                (world_points, np.array([[cam_x + dx, cam_y + dy, floor_level]]))
            )

    return world_points


def visible_floor_voxels(
    camera_pos, floor_voxels, occupied_points, wall_voxel_size=0.1, nav_voxel_size=0.2
):
    """
    Returns 2D floor voxels visible from camera,
    blocking rays passing through or between wall voxels.
    """
    floor_level = floor_voxels[0, 2]  # all floor voxels share same Z

    occupied_points = occupied_points[occupied_points[:, 2] < camera_pos[2]]

    wall_points = extract_walls(
        points=occupied_points,
        grid_res=wall_voxel_size,
    )

    if len(wall_points) == 0:
        return floor_voxels

    wall_points_2d = np.unique(wall_points[:, :2], axis=0)

    wall_set = KDTree(wall_points_2d)

    # --- 3. Raytrace each floor voxel ---
    visible = []

    camera_pos_2d = camera_pos[:2]
    floor_voxels_2d = floor_voxels[:, :2]

    for fv in floor_voxels_2d:
        direction = np.array(fv - camera_pos_2d)
        dist = np.linalg.norm(direction)
        if dist == 0:
            visible_3d = np.array([fv[0], fv[1], floor_level])
            visible.append(visible_3d)
            continue

        steps = int(round(dist / nav_voxel_size))

        if steps == 0:
            visible_3d = np.array([fv[0], fv[1], floor_level])
            visible.append(visible_3d)
            continue

        step_vec = direction / steps
        blocked = False

        for s in range(1, steps + 1):
            sample_point = camera_pos_2d + step_vec * s

            # Check neighborhood
            if wall_set.query_ball_point(sample_point, nav_voxel_size * 1.5):
                blocked = True
                break

        if not blocked:
            visible_3d = np.array([fv[0], fv[1], floor_level])
            visible.append(visible_3d)

    visible = np.array(visible)

    wall_points = np.array(wall_points)

    return np.array(visible), np.array(wall_points)


def compute_elevation_map(points, grid_res=0.1):
    """
    points: (N,3) array
    grid_res: size of each cell in meters
    Returns:
        min_z: HxW array (elevation)
        var_z: HxW array (roughness)
        valid: mask where at least 1 point fell into cell
        x_grid, y_grid: coordinate grids
    """

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # grid size
    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()

    W = int(np.ceil((max_x - min_x) / grid_res))
    H = int(np.ceil((max_y - min_y) / grid_res))

    # index for each point
    ix = ((x - min_x) / grid_res).astype(int)
    iy = ((y - min_y) / grid_res).astype(int)

    ix = np.clip(ix, 0, W - 1)
    iy = np.clip(iy, 0, H - 1)

    # accumulators
    min_z = np.full((H, W), np.inf)
    max_z = np.full((H, W), -np.inf)
    sum_z = np.zeros((H, W))
    count = np.zeros((H, W))

    # fill stats
    for px, py, pz in zip(ix, iy, z):
        if 0 <= px < W and 0 <= py < H:
            if pz < min_z[py, px]:
                min_z[py, px] = pz
            if pz > max_z[py, px]:
                max_z[py, px] = pz
            sum_z[py, px] += pz
            count[py, px] += 1

    # compute variance
    mean_z = np.zeros_like(min_z)
    valid = count > 0
    mean_z[valid] = sum_z[valid] / count[valid]

    # second moment
    var_acc = np.zeros((H, W))
    for px, py, pz in zip(ix, iy, z):
        if 0 <= px < W and 0 <= py < H:
            if valid[py, px]:
                diff = pz - mean_z[py, px]
                var_acc[py, px] += diff * diff
    var_z = np.zeros_like(min_z)
    var_z[valid] = var_acc[valid] / count[valid]

    return min_z, var_z, valid, (min_x, min_y)


def slope_from_elevation(min_z):
    """
    Computes slope magnitude per grid cell from elevation map.
    Uses Sobel gradient.
    """
    # replace invalid with a large constant so they don't create gradients
    z = np.copy(min_z)
    z[np.isinf(z)] = np.nan
    z = gaussian_filter(np.nan_to_num(z, nan=0.0), sigma=1.0)

    gx = sobel(z, axis=1)
    gy = sobel(z, axis=0)

    slope = np.sqrt(gx**2 + gy**2)
    return slope


def extract_walls(
    points,
    grid_res=0.1,
    max_slope=0.3,
    max_variance=0.03,  # meters / meter, ~11°
):  # adjust for noise
    """
    Returns a boolean mask of the elevation map where floor is true.
    Entirely dynamic: no absolute height threshold.
    """

    min_z, var_z, valid, (min_x, min_y) = compute_elevation_map(points, grid_res)
    slope = slope_from_elevation(min_z)

    wall_mask = np.zeros_like(min_z, dtype=bool)
    wall_mask[valid] = (slope[valid] > max_slope) | (var_z[valid] > max_variance)
    wall_indices = np.argwhere(wall_mask)
    wall_points = []
    for iy, ix in wall_indices:
        x_coord = min_x + ix * grid_res
        y_coord = min_y + iy * grid_res
        z_coord = min_z[iy, ix]
        wall_points.append([x_coord, y_coord, z_coord])
    return np.array(wall_points)


def prune_nav_kdtree(nav_kdt: KDTree, voxel_size: float = 0.2) -> KDTree:
    """
    Remove disconnected islands from a 2D navigable KDTree and return a new KDTree.

    nav_kdt: KDTree containing 2D points
    voxel_size: approximate spacing of points

    Returns: KDTree containing only the largest connected component
    """
    nav_points = nav_kdt.data
    if nav_points.shape[0] == 0:
        return KDTree(nav_points)  # empty tree

    # Build temporary tree for neighbor search
    tree = KDTree(nav_points)
    visited = np.zeros(len(nav_points), dtype=bool)
    components = []

    # BFS for connected components
    for i in range(len(nav_points)):
        if visited[i]:
            continue
        component = []
        queue = deque([i])
        visited[i] = True

        while queue:
            idx = queue.popleft()
            component.append(idx)
            neighbors = tree.query_ball_point(nav_points[idx], r=voxel_size * 1.1)
            for n in neighbors:
                if not visited[n]:
                    visited[n] = True
                    queue.append(n)

        components.append(component)

    # Keep only the largest component
    largest = max(components, key=len)
    filtered_points = nav_points[largest]

    # Return a new KDTree
    return KDTree(filtered_points)
