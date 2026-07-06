import gc
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SAM3_ROOT = Path(__file__).resolve().parent / "third_party" / "sam3"
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from utils.server_wrapper import ServerMixin, host_agent

SAM3_BPE_PATH = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

if __name__ == "__main__":

    class Sam3Server(ServerMixin):

        def __init__(self) -> None:
            super().__init__()

            # Load the model
            self.model = build_sam3_image_model(str(SAM3_BPE_PATH))
            self.processor = Sam3Processor(self.model)
            self.use_cuda_autocast = torch.cuda.is_available()

            self.cleanup_count = 0

        def process_payload(self, payload: dict) -> dict:
            self.cleanup_count += 1
            if self.cleanup_count % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            try:
                image = np.array(payload["image"], dtype=np.uint8)
                image = Image.fromarray(image)
                prompt = payload["prompt"]

                inference_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self.use_cuda_autocast
                    else nullcontext()
                )
                with inference_context:
                    inference_state = self.processor.set_image(image)
                    # Prompt the model with text
                    output = self.processor.set_text_prompt(
                        state=inference_state, prompt=prompt
                    )

                # Get the masks, bounding boxes, and scores
                masks, boxes, scores = (
                    output["masks"],
                    output["boxes"],
                    output["scores"],
                )

                # NumPy cannot serialize bfloat16 tensors produced under autocast.
                masks = masks.detach().to(dtype=torch.bool, device="cpu")
                boxes = boxes.detach().to(dtype=torch.float32, device="cpu")
                scores = scores.detach().to(dtype=torch.float32, device="cpu")

                return {
                    "result": "success",
                    "masks": masks.numpy().tolist(),
                    "boxes": boxes.numpy().tolist(),
                    "scores": scores.numpy().tolist(),
                }
            except Exception as e:
                return {"result": "error", "message": str(e)}

    server = Sam3Server()
    print("Sam3 loaded!")
    host_agent(server, name="sam3", port=12184)
