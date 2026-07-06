import numpy as np
from typing import Optional
from utils.server_wrapper import send_request


class VLMClient:
    def __init__(self, endpoint: str, port: int):
        self.url = f"http://localhost:{port}/{endpoint}"

    def send_request(
        self,
        prompt: str,
        image: np.ndarray = "none",
        system_prompt: str = "none",
    ) -> tuple[Optional[np.ndarray], str]:

        result = send_request(
            self.url,
            image=image,
            prompt=prompt,
            system_prompt=system_prompt,
        )

        if result["result"] == "error":
            raise RuntimeError(f"{self.url} client error: {result['message']}")

        return result
