from PIL import Image


class InferenceBase:
    def predict(self, prompt: str, image: Image.Image) -> str:
        raise NotImplementedError
