from openmind import pipeline, is_torch_npu_available
from openmind import AutoTokenizer, AutoModelForCausalLM
from openmind_hub import snapshot_download
import torch.nn.functional as F
from torch import Tensor
import openmind
import torch
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to model",
        default="pretrainmodel/bert-base-parsbert-ner-uncased",
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
    
    token_classifier = pipeline(task="token-classification", model=model_path, framework="pt", device=device)
    
    output = token_classifier("This is a test !")
    print(output)
    
    end_time = time.time()
    print(f"硬件环境：{device},推理执行时间：{end_time - start_time}秒")
    
if __name__ == "__main__":
    main()
