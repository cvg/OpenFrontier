import gc
import torch
import argparse
import numpy as np
from PIL import Image

from vlm.models import VLMModel
from vlm.inference_llava import InferenceLlava
from vlm.inference_gemma3 import InferenceGemma3
from vlm.inference_internvl import InferenceInternVL
from utils.server_wrapper import ServerMixin, host_agent

if __name__ == "__main__":

    class VLMServer(ServerMixin):

        def __init__(self, model: str) -> None:
            super().__init__()

            try:
                model = VLMModel(model)
            except ValueError:
                raise ValueError(f"Unsupported VLM model: {model}")

            if model.value.lower().startswith("gemma"):
                self.model = InferenceGemma3(model.value)

            elif model.value.lower().startswith("intern"):
                self.model = InferenceInternVL(model)
                
            elif "llava" in model.value.lower():
                self.model = InferenceLlava(model.value)
            else:
                raise ValueError(f"Unsupported local VLM model: {model.value}")

            self.cleanup_count = 0

        def process_payload(self, payload: dict) -> dict:
            self.cleanup_count += 1
            if self.cleanup_count % 10 == 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()

            # try:
            
            prompt = payload["prompt"]
            
            try:
                image = payload["image"]
                image = np.array(image, dtype=np.uint8)
                image = Image.fromarray(image)
            except:
                image = "none"
            
            try:
                system_prompt = payload["system_prompt"]
            except:
                system_prompt = "none"
                
            response = self.model.predict(image, prompt, system_prompt)
            return {"result": "success", "response": response}

            # except Exception as e:
            # return {"result": "error", "message": str(e)}

    parser = argparse.ArgumentParser(description="VLM Server for OpenFrontier")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="VLM model to use",
    )

    model = parser.parse_args().model

    server = VLMServer(model)
    print(f"{model} loaded!")
    host_agent(server, name="vlm", port=12185)
