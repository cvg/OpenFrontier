import os
import sys
from pathlib import Path

import torch
#################################### For Image ####################################
from PIL import Image

SAM3_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "sam3"
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import cv2
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_img', type=str, required=True, help='Path to the input image')
parser.add_argument('--target_object', type=str, required=True, help='Text prompt for segmentation')

args = parser.parse_args()

# parent dir from input image
parent_dir = os.path.dirname(args.input_img)



# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open(
args.input_img
)
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt=args.target_object)

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# move to cpu
masks = masks.cpu()
boxes = boxes.cpu()
scores = scores.cpu()

print(f"Found {len(masks)} masks for the prompt '{args.target_object}'")

combined = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA)


# Draw the masks on the image
# Process each mask
for i, mask in enumerate(masks):

    mask = mask.numpy()[0]

    color = (255, 0, 0, 200)

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row, col]:
                combined[row, col] = color

# Save the combined image
image_name = os.path.basename(args.input_img)
output_path = os.path.join(parent_dir, image_name.split('.')[0] + '_segmentation.png')
cv2.imwrite(output_path, combined)
print(f"Segmentation result saved to {output_path}")
