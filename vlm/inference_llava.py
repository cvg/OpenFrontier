import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, pipeline
from PIL import Image
from vlm.inference_base import InferenceBase



class InferenceLlava(InferenceBase):
    def __init__(self, model_name: str) -> None:
        if not model_name.lower().startswith("llava"):
            raise ValueError(
                f"Unsupported model for LLAVA: {model_name}"
            )

        path = "llava-hf/" + model_name

        self.pipe = pipeline("image-text-to-text", model=path)

    def predict(self, image: Image.Image, prompt: str) -> str:
        messages = [
        {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
            ],
        }]
        out = self.pipe(text=messages, images=image, max_new_tokens=500)
        gen_text = out[0]["generated_text"][-1]["content"]
        return gen_text
