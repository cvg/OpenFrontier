from vlm.utils import detect_frontier_probabilities
from utils.api_key import get_google_api_key
import PIL.Image
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Detect frontier probabilities using Gemini"
    )
    parser.add_argument(
        "--input_img", type=str, required=True, help="Path to input RGB image"
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Target object to search for"
    )

    args = parser.parse_args()

    # Load the input image
    image = np.array(PIL.Image.open(args.input_img))

    # Detect frontier probabilities
    probabilities = detect_frontier_probabilities(
        rgb_image=image, target_object=args.target, api_key=get_google_api_key()
    )
    num_frontiers = len(probabilities[0])
    obj_prob_list = []
    for i in range(num_frontiers):
        ft_label = chr(65 + i)  # 'A', 'B', 'C', ...
        try:
            ext_gain = probabilities[0][ft_label][0]  # Extract probability
            print(
                f"Frontier {ft_label}: Probability of leading to {args.target}: {ext_gain:.2f}"
            )
        except KeyError:
            ext_gain = 0.0
            print(
                f"Frontier {ft_label}: Probability data not available, defaulting to {ext_gain:.2f}"
            )
        obj_prob_list.append(ext_gain)

    # save the obj_prob_list as a npy file, under same directory as input image, named as input_img name + _{target}_prob.npy
    out_path = (
        args.input_img.replace(".png", f"_{args.target}_prob.npy")
        .replace(".jpg", f"_{args.target}_prob.npy")
        .replace(".jpeg", f"_{args.target}_prob.npy")
    )
    np.save(out_path, np.array(obj_prob_list))

    print("Detected Frontier Probabilities:", probabilities)
