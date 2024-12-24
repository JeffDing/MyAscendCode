import os
import argparse

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from openmind import is_torch_npu_available
from openmind_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to the model",
        default="jeffding/SSD-1B-openmind",
        )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    model_path = args.model_name_or_path
    
    if is_torch_npu_available():
        device = "npu:0"
    else:
        device = "cpu"

    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to(device)

    prompt = "An astronaut riding a green horse."
    image = pipe(prompt=prompt).images[0]

    image.save("astronaut_rides_horse.png")


if __name__ == '__main__':
    main()

