from openmind import AutoTokenizer, pipeline, is_torch_npu_available
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
        default="pretrainmodel/I-SOLAR-10.7B-dpo-sft-v1.0",
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
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    start_time = time.time()
    
    pipe = pipeline("text-generation", model=model_path, torch_dtype=torch.bfloat16, device_map=device)
    messages = [
        {
            "role": "system",
            "content": "당신은 친절한 채팅 로봇, 항상 해적 스타일로 응답",
        },
        {"role": "user", "content": "당신은 친절한 채팅 로봇, 항상 해적 스타일로 응답?"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs)
    
    end_time = time.time()
    print(f"硬件环境：{device},推理执行时间：{end_time - start_time}秒")
    
if __name__ == "__main__":
    main()
