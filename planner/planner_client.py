import numpy as np
from typing import Optional
from utils.server_wrapper import send_request
import logging


class PlannerClient:
    def __init__(self, port: int = 12184):
        self.url = f"http://localhost:{port}/sam3"

    def send_request(
        self,
        image: np.ndarray,
        prompt: str,
    ) -> tuple[Optional[np.ndarray], str]:

        result = send_request(
            self.url,
            image=image,
            prompt=prompt,
        )

        if result["result"] == "error":
            logging.error(f"PlannerClient error: {result['message']}")
            return None

        return result
