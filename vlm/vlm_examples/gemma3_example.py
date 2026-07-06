from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-12b-pt",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

output = pipe(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    text="<start_of_image> in this image, there is",
)

print(output)
# [{'input_text': '<start_of_image> in this image, there is',
# 'generated_text': '<start_of_image> in this image, there is a bumblebee on a pink flower.\n\n'}]
