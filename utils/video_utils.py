import numpy as np
import cv2
from matplotlib.colors import LinearSegmentedColormap

VIEW_WIDTH = 320
BORDER_WIDTH = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX

TITLE_FONT_SIZE = 0.7
TITLE_FONT_THICKNESS = 1

FRONTIER_FONT_SIZE = 0.6
FRONTIER_FONT_THICKNESS = 1

TITLE_BACKGROUND_COLOR = (0, 0, 0)
TITLE_TEXT_COLOR = (255, 255, 255)

BORDER_COLOR = (0, 0, 0)

def position_to_topdown(position, topdown_image, bbox):
        map_size_x, map_size_y, _ = topdown_image.shape
        x_box = [bbox[0], bbox[1]]
        y_box = [bbox[2], bbox[3]]
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

def pose_to_topdown(pose, topdown_image, bbox):
    rotation = pose[:3, :3]
    position = pose[:3, 3]
    x_axis = rotation[:, 0]
    angle = np.arctan2(x_axis[1], x_axis[0])

    map_size_x, map_size_y, _ = topdown_image.shape
    x_box = [bbox[0], bbox[1]]
    y_box = [bbox[2], bbox[3]]
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

def img_msg_to_img(img_msg):
    img_data = np.frombuffer(img_msg.data, dtype=np.uint8)
    image = img_data.reshape(img_msg.height, img_msg.width, -1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (VIEW_WIDTH, VIEW_WIDTH))
    return image

def draw_pointcloud(topic, bbox, topdown_image, color):
    if topic is not None:
        # get the points from the pointcloud
        raw_data = np.frombuffer(topic.data, dtype=np.float32)
        occ_map = raw_data.reshape(-1, 3)[:, :3]
        occ_map = occ_map[occ_map[:, 2] > bbox[4] + 0.5]  # filter by z min)
        occ_map = occ_map[occ_map[:, 2] < bbox[4] + 2]
        #  get one point per x, y
        occ_map = occ_map[np.unique(occ_map[:, :2], axis=0, return_index=True)[1]]
    else:
        occ_map = np.empty((0, 3))

    for point in occ_map:
      x_pixel, y_pixel = position_to_topdown(point, topdown_image, bbox)
      cv2.circle(topdown_image, (x_pixel, y_pixel), 2, color, -1)

def draw_title(img, title, font_size = TITLE_FONT_SIZE):
    (text_width, text_height), _ = cv2.getTextSize(title, FONT, font_size, TITLE_FONT_THICKNESS)
    cv2.rectangle(img, (0, 0), (int(text_width * 1.2), int(text_height * 2)), TITLE_BACKGROUND_COLOR, -1)
    cv2.putText(img, title, (10, int(text_height * 1.5)), FONT, font_size, TITLE_TEXT_COLOR, TITLE_FONT_THICKNESS, cv2.LINE_AA)

def draw_border(img):
    border = BORDER_WIDTH
    img[:border, :, :] = BORDER_COLOR
    img[-border:, :, :] = BORDER_COLOR
    img[:, :border, :] = BORDER_COLOR
    img[:, -border:, :] = BORDER_COLOR

def img_from_file(file, fallback):
  if file.exists():
      img = cv2.imread(str(file))
      img = cv2.resize(img, (VIEW_WIDTH, VIEW_WIDTH))
  else:
      img = fallback
  return img

def draw_frontiers(frontiers, topdown_image, bbox):
    utilities = [
        ft["utility"] if ft["utility"] is not None else 0
        for ft in frontiers
    ]

    if len(utilities) > 1:
      min_utility = min(utilities)
      max_utility = max(utilities)
    else:
      min_utility = 0
      if len(utilities) == 1:
        max_utility = utilities[0]
      else:
        max_utility = 0

    utility_range = (
        max_utility - min_utility if max_utility > min_utility else 1.0
    )
    yellow_red_cmap = LinearSegmentedColormap.from_list(
        "yellow_red", ["#F6FF55", "#FF0202"]
    )

    object_frontiers = [ft for ft in frontiers if ft["is_object"]]
    non_object_frontiers = [ft for ft in frontiers if not ft["is_object"]]
    for ft in non_object_frontiers + object_frontiers:

        if ft["utility"] is None:
            ft["utility"] = 0.0

        weight = (ft["utility"] - min_utility) / utility_range

        color_value = min(weight, 1.0)
        color = yellow_red_cmap(color_value)[:3]


        ft_pose = np.array(ft["6d_pose"])

        x, y, angle = pose_to_topdown(ft_pose, topdown_image, bbox)
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
        if ft["utility"] < 1e5:
            util = f"{ft['utility']:.2f} | {int(ft['probability']*100)}%"
        elif ft["utility"] >= 1e5 and ft["utility"] < 1e15:
            utility = float(ft["u_gain"]) * float(ft["probability"]) / 0.1
            util = f"{utility:.2f} | {int(ft['probability']*100)}%"
        else:
            util = "object"

        # get size of text box
        (text_width, text_height), _ = cv2.getTextSize(util, FONT, FRONTIER_FONT_SIZE, FRONTIER_FONT_THICKNESS)
        # draw white rectangle behind text
        cv2.rectangle(
            topdown_image,
            (x + 13, y - text_height // 2),
            (x + 17 + text_width, y + text_height // 2 * 3),
            (255, 255, 255),
            -1,
        )

        cv2.putText(
            topdown_image,
            str(util),
            (x + 15, y + 10),
            FONT,
            FRONTIER_FONT_SIZE,
            (0, 0, 0),
            FRONTIER_FONT_THICKNESS,
            cv2.LINE_AA,
        )