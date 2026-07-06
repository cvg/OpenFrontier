from transformers import pipeline
import torch
from PIL import Image
from vlm.inference_base import InferenceBase


class InferenceGemma3(InferenceBase):
    def __init__(self, model_name: str) -> None:
        if not model_name.lower().startswith("gemma"):
            raise ValueError(
                f"Unsupported model for InferenceGemma3: {model_name}"
            )
            
        model = model_name.lower().replace("-local", "")

        path = "google/" + model
        self.pipe = pipeline(
            "image-text-to-text",
            model=path,
            device="cuda",
            torch_dtype=torch.bfloat16,
        )

    def predict(self, image: Image.Image, prompt: str, system_prompt: str = "none") -> str:
        
        if system_prompt == "none":
            system_prompt = "You are an assistant for an object-goal navigation pipeline based on visual frontiers. You will help the agent by answering questions about frontiers that could lead to the target object, and by identifying whether the target object is present in the image. You will always return a JSON list with one dictionary and when requested reasoning, you will be concise and to the point."
        
        if image != "none":
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        else:
            content = [
                {"type": "text", "text": prompt},
            ]
        
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": content,
            },
        ]

        output = self.pipe(text=messages, max_new_tokens=1000)
        return output[0]["generated_text"][-1]["content"]
