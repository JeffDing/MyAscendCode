import argparse

import torch
from openmind import pipeline, is_torch_npu_available


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to model",
        default="pretrainmodel/opus-mt-en-es",
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
        
    start_time = time.time()

    translator = pipeline("translation", model=model_path, framework="pt", device=device)
    
    output = translator("translate English to Spanish: Her name is Sarsh and she lives in London.")
    print(output)
    
    end_time = time.time()
    print(f"硬件环境：{device},推理执行时间：{end_time - start_time}秒")

if __name__ == "__main__":
    main()
