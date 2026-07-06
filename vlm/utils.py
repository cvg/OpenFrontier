import json
import numpy as np
from typing import Tuple
import PIL.Image
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
from vlm.client import VLMClient
import base64
import io

from vlm.models import VLMModel, SegmentationModel, is_google_api, is_gemma_api

COMPOSITIONS = {1: (1, 1), 2: (2, 1), 4: (2, 2), 6: (2, 3), 8: (2, 4), 9: (3, 3)}
COMPRESSION = 0.5


GEMINI_CONFIG = types.GenerateContentConfig(response_mime_type="application/json")


def detect_frontier_probabilities(
    rgb_image: np.ndarray,
    labels: list,
    target_object: str,
    vlm_model: VLMModel,
    api_key: str = None,
) -> Tuple[bool, list[dict], str]:
    """
    Given an image and a target object (e.g., 'bathroom', 'toy'), use Gemini to estimate
    the probability and reasoning for each labeled frontier leading to the object.

    Args:
        rgb_image (np.ndarray): Input RGB image (H, W, 3).
        target_object (str): The object to be searched for (e.g., "bathroom").
        api_key (str): API key for Gemini.

    Returns:
        bool: Success
        list[dict]: A list with one dictionary mapping labels (e.g., "A", "B", "C")
                    to [probability, reason] format.
        str: Raw response
    """
    # Convert NumPy image to PIL
    image = PIL.Image.fromarray(rgb_image)

    if isinstance(vlm_model, str):
        vlm_model = VLMModel(vlm_model)

    split = target_object.split("_special_")

    if len(split) > 0:
        target_object = split[0]

    # Prompt template
    prompt = (
        f" Assume the labeled frontiers {labels} represent possible places to go. Each frontier is a detected boundary between explored and unexplored space. "
        f"Estimate the probability that each frontier leads to (or is already around) a {target_object} when moving towards it and continue exploring from there, along with reasoning. Unseen labels should have a probability of 0.5."
        f"Note that also consider longer-term navigation possibilities, not just immediate visibility, and that some frontiers may lead to larger unknown regions. "
        f"Also pay attention to the neighborhood context around each frontier, since each frontier is already confirmed to lead to some unexplored space. "
        f"Return only a JSON list with one dictionary. Each key should be the frontier label, "
        f"and the value should be a list: the first item is the probability (0 to 1), "
        f"and the second is a short explanation. Format: "
        f'{{"A": [0.3, "reason"], "B": [0.2, "reason"], ...}}'
        f"Stop after you have covered frontiers {labels}"
    )

    # Initialize Gemini API

    if is_google_api(vlm_model):
        if api_key is None:
            raise ValueError("API key must be provided for Gemini models.")

        # Send to Gemini model
        client = genai.Client(api_key=api_key)
        model_name = vlm_model.value.replace("-api", "")

        if is_gemma_api(vlm_model):
            response = client.models.generate_content(
                model=model_name, contents=[prompt, image]
            )
        else:
            response = client.models.generate_content(
                model=model_name, contents=[prompt, image], config=GEMINI_CONFIG
            )
        response_text = response.text
    else:
        # Send to local VLM server
        client = VLMClient("vlm", port=12185)
        response = client.send_request(image=rgb_image, prompt=prompt)
        response_text = response.get("response", "")


    # Strip formatting artifacts if present
    raw_response = response_text.strip()

    # Find ```json
    if "```json" in raw_response:
        raw_response = raw_response.split("```json")[-1]
    if "```" in raw_response:
        raw_response = raw_response.split("```")[0].strip()
 
    # Parse JSON
    try:
        output = json.loads(raw_response)
        if isinstance(output, dict):
            return True, output, raw_response
        elif (
            isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict)
        ):
            return True, output[0], raw_response
        else:
            print("Unexpected JSON structure.")
            return False, {}, raw_response
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        return False, {}, raw_response


def segment_target_object(
    rgb_composition: np.ndarray,
    target_object: str,
    segmentation_model: SegmentationModel,
    n_images: int = 1,
    api_key: str = None,
) -> list[dict]:

    if isinstance(segmentation_model, str):
        segmentation_model = SegmentationModel(segmentation_model)

    # Convert NumPy image to PIL
    image = PIL.Image.fromarray(rgb_composition)

    split = target_object.split("_special_")

    if len(split) > 0:
        target_object = split[0]

    if segmentation_model == SegmentationModel.GEMINI_2_5_FLASH:
        if api_key is None:
            raise ValueError("API key must be provided for Gemini segmentation model.")

        prompt = (
            f" Give the segmentation masks for the {target_object} object, unless it is reflected in a mirror."
            # f" Only consider {target_object} for which the majority of the {target_object} is visible and not occluded."
            # f" Only consider {target_object} that is with all certainty a {target_object}."
            # f" Limit the output to a maximum of three {target_object} instances."
            f" Output a JSON list of segmentation masks where each entry contains the 2D"
            f' bounding box in the key "box_2d", the segmentation mask in key "mask", and'
            f' the text label in the key "label". Use descriptive labels. If there is no {target_object}, return an empty list.'
        )

        # Initialize Gemini API
        client = genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            )  # set thinking_budget to 0 for better results in object detection
        )
        response = client.models.generate_content(
            model=segmentation_model.value, contents=[prompt, image], config=config
        )
        output_masks, combined = _parse_gemini_segmentation_output(
            response.text, image, n_images
        )

    elif segmentation_model == SegmentationModel.SAM3:
        client = VLMClient("sam3", port=12184)
        image_array = np.array(image)
        response = client.send_request(image=image_array, prompt=target_object)
        if response["result"] != "success":
            print("SAM3 segmentation failed:", response.get("message", "Unknown error"))
            return [], image

        masks = response["masks"]
        boxes = response["boxes"]
        output_masks, combined = _parse_sam3_segmentation_output(
            masks, boxes, image, n_images, target_object
        )
    else:
        raise ValueError(f"Unsupported segmentation model: {segmentation_model.value}")

    return (output_masks, combined)


def detect_target_object(
    rgb: np.ndarray,
    target_object: str,
    vlm_model: VLMModel,
    api_key: str = None,
) -> Tuple[bool, list[dict], str]:
    """
    Given an image and a target object (e.g., 'bathroom', 'toy'), use Gemini to estimate
    the probability and reasoning for the target object being present.

    Args:
        rgb: (np.ndarray): Input RGB image (H, W, 3).
        target_object (str): The object to be searched for (e.g., "bathroom").
        api_key (str): API key for Gemini.

    Returns:
        bool: Success
        tuple[dict]: The probability and reasoning for the target object being present.
        str: raw_response
    """
    # Convert NumPy image to PIL
    image = PIL.Image.fromarray(rgb)

    if isinstance(vlm_model, str):
        vlm_model = VLMModel(vlm_model)

    split = target_object.split("_special_")
    special = None

    if len(split) > 1:
        special = split[1]
        target_object = split[0]

    addition = ""
    if target_object.lower() == "sofa" or target_object.lower() == "loveseat":
        target_object = "sofa"
        addition = ", consider only wide sofas, loveseats and sectionals, that can clearly seat two or more people, not single-seat chairs or armchairs"
    elif "bed" in target_object.lower():
        addition = ", consider only full beds with clearly visible mattresses and/or bedding, not sofas or couches"
    elif target_object.lower() == "chair":
        addition = ", consider only single-seat chairs and stools, not sofas, armchairs, long wooden benches or exercising chairs"
    elif "screen" in target_object.lower() or "monitor" in target_object.lower():
        addition = ", consider only televisions and computer monitors, not electronic displays like tablets or kitchen appliances"
    elif "plant" in target_object.lower():
        addition = ", consider only plants and flowers in pots and vases, artificial flower arrangements are acceptable too. Do not consider paintings or photos of plants, or decorative branch arrangements without leaves or flowers"
    elif "toilet" in target_object.lower():
        addition = ", consider only adult toilets, not child or portable toilets"

    if "queen" in target_object.lower():
        target_object = "bed"

    if special is not None and special == "no_purple_flowers":
        addition += ", do not consider purple flowers on top of a golden table"

    prompt = (
        f"Based on this image, estimate the probability that a {target_object} is in the field of view of the camera, in-frame and within a distance of five meters{addition}."
        f" If the {target_object} is a photo or painting, reflected on a mirror, behind a glass window or door, overally unreachable, barely visible or mostly occluded it should not be considered present."
        f" Keep probabilities either close to 0 for absent or close to 1 for present. "
        f" Add one sentence of reasoning. "
        f"Return a JSON list with one dictionary. Format: "
        f'{{"probability": 0.9, "reason": "reason"}}'
    )

    if is_google_api(vlm_model):
        if api_key is None:
            raise ValueError("API key must be provided for Gemini models.")

        model_name = vlm_model.value.replace("-api", "")

        client = genai.Client(api_key=api_key)

        if is_gemma_api(vlm_model):
            response = client.models.generate_content(
                model=model_name, contents=[prompt, image]
            )
        else:
            response = client.models.generate_content(
                model=model_name, contents=[prompt, image], config=GEMINI_CONFIG
            )

        response_text = response.text
    else:
        client = VLMClient("vlm", port=12185)
        response = client.send_request(image=rgb, prompt=prompt)
        response_text = response.get("response", "")


    # Strip formatting artifacts if present
    raw_response = response_text.strip()


    # Find ```json
    if "```json" in raw_response:
        raw_response = raw_response.split("```json")[-1]
    if "```" in raw_response:
        raw_response = raw_response.split("```")[0].strip()

    # Parse JSON
    try:
        output = json.loads(raw_response)
        if isinstance(output, dict):
            return True, output, raw_response
        elif (
            isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict)
        ):
            return True, output[0], raw_response
        else:
            print("Unexpected JSON structure.")
            return (
                False,
                {"probability": 0.0, "reason": "Unexpected JSON structure."},
                raw_response,
            )
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        return (
            False,
            {"probability": 0.0, "reason": "Failed to decode JSON."},
            raw_response,
        )


def _parse_json(json_output: str):
    output = json_output.strip()

    if "```json" in output:
        output = output.split("```json")[-1]
    if "```" in output:
        output = output.split("```")[0].strip()

    return output


def _parse_gemini_segmentation_output(
    response_text: str, image: PIL.Image.fromarray, n_images: int
) -> list[dict]:
    try:
        parsed = _parse_json(response_text)
        if parsed is None:
            print(response_text)
            return [], image
        items = json.loads(parsed)

    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        print(response_text)
        return [], image

    output_masks = []

    combined = image.convert("RGBA")

    # Process each mask
    for _, item in enumerate(items):
        try:

            # Get bounding box coordinates
            box = item["box_2d"]
            y0 = int(box[0] / 1000 * image.size[1])
            x0 = int(box[1] / 1000 * image.size[0])
            y1 = int(box[2] / 1000 * image.size[1])
            x1 = int(box[3] / 1000 * image.size[0])

            mask_object = {}

            # Skip invalid boxes
            if y0 >= y1 or x0 >= x1:
                continue

            # Process mask
            png_str = item["mask"]
            if not png_str.startswith("data:image/png;base64,"):
                continue

            # Remove prefix
            png_str = png_str.removeprefix("data:image/png;base64,")
            mask_data = base64.b64decode(png_str)
            mask = Image.open(io.BytesIO(mask_data))

            # Resize mask to match bounding box
            mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)

            # Convert mask to numpy array for processing
            mask_array = np.array(mask)

            # Create overlay for this mask
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            orig_w, orig_h = image.size

            rows, cols = COMPOSITIONS[n_images]

            width_ratio = cols * COMPRESSION
            height_ratio = rows * COMPRESSION

            orig_h = int(orig_h / height_ratio)
            orig_w = int(orig_w / width_ratio)

            # Create overlay for the mask
            color = (255, 0, 0, 200)
            thresholded_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

            center = [int((x0 + x1) / 2), int((y0 + y1) / 2)]

            # Find which of the image in the grid the center falls into
            grid_w = image.size[0] / cols
            grid_h = image.size[1] / rows
            grid_row = min(int(center[1] / grid_h), rows - 1)
            grid_col = min(int(center[0] / grid_w), cols - 1)
            grid_index = grid_row * cols + grid_col

            for y in range(y0, y1):
                for x in range(x0, x1):
                    if mask_array[y - y0, x - x0] > 128:  # Threshold for mask
                        overlay_draw.point((x, y), fill=color)

                        orig_x = int((x / COMPRESSION) % orig_w)
                        orig_y = int((y / COMPRESSION) % orig_h)
                        thresholded_mask[orig_y, orig_x] = 1

            combined = Image.alpha_composite(combined, overlay)

            mask_object["label"] = item["label"]
            mask_object["mask"] = thresholded_mask
            mask_object["image_index"] = grid_index

            output_masks.append(mask_object)

            print_object = {
                "label": item["label"],
                "image_index": grid_index,
                "box_2d": [x0, y0, x1, y1],
            }
            print("Detected Object:", print_object)

        except Exception as e:
            print(f"Error processing mask: {e}")
            print("Item causing error:", item)
            continue

    return output_masks, combined


def _parse_sam3_segmentation_output(
    masks: list,
    boxes: list,
    image: PIL.Image.fromarray,
    n_images: int,
    prompt: str,
) -> list[dict]:

    output_masks = []

    combined = np.array(image.convert("RGBA"))

    # Process each mask
    for i, mask in enumerate(masks):
        try:
            mask = np.array(mask)
            if mask.ndim == 3:
                mask = mask[0]
            box = boxes[i]
            x0 = np.clip(int(box[0]), 0, image.size[0])
            y0 = np.clip(int(box[1]), 0, image.size[1])
            x1 = np.clip(int(box[2]), 0, image.size[0])
            y1 = np.clip(int(box[3]), 0, image.size[1])

            mask_object = {}

            # Skip invalid boxes
            if y0 >= y1 or x0 >= x1:
                continue

            orig_w, orig_h = image.size

            rows, cols = COMPOSITIONS[n_images]

            width_ratio = cols * COMPRESSION
            height_ratio = rows * COMPRESSION

            orig_h = int(orig_h / height_ratio)
            orig_w = int(orig_w / width_ratio)

            # Create overlay for the mask
            color = [255, 0, 0, 200]

            center = [int((x0 + x1) / 2), int((y0 + y1) / 2)]

            # Find which of the image in the grid the center falls into
            grid_w = image.size[0] / cols
            grid_h = image.size[1] / rows
            grid_row = min(int(center[1] / grid_h), rows - 1)
            grid_col = min(int(center[0] / grid_w), cols - 1)
            grid_index = grid_row * cols + grid_col

            thresholded_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

            for y in range(y0, y1):
                for x in range(x0, x1):
                    if mask[y, x]:  # Threshold for mask
                        combined[y, x, :4] = color

                        orig_x = int((x / COMPRESSION) % orig_w)
                        orig_y = int((y / COMPRESSION) % orig_h)
                        thresholded_mask[orig_y, orig_x] = 1

            mask_object["label"] = prompt
            mask_object["mask"] = thresholded_mask
            mask_object["image_index"] = grid_index

            output_masks.append(mask_object)

            print_object = {
                "label": prompt,
                "image_index": grid_index,
                "box_2d": [y0, x0, y1, x1],
            }
            print("Detected Object:", print_object)

        except Exception as e:
            print(f"Error processing mask: {e}")
            print("Item causing error:", i)
            continue

    combined = PIL.Image.fromarray(combined)
    return output_masks, combined
