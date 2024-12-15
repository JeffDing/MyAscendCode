from openmind import AutoModelForCausalLM, AutoTokenizer
from openmind import is_torch_npu_available
import torch
import argparse
import torch.nn.functional as F
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to model",
        default="pretrainmodel/Mistral-7B-Instruct-v0.1-GGUF",
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
    #device = "cpu"  
    filename = "mistral-7b-instruct-v0.1.Q2_K.gguf"
    tokenizer = AutoTokenizer.from_pretrained(model_path, gguf_file=filename, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, gguf_file=filename, device_map=device
    )
    
    start_time = time.time()
    
    prompt = "Q: hello who are you?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=32)

    print(tokenizer.decode(generation_output[0]))
    end_time = time.time()
    print(f"硬件环境：{device},推理执行时间：{end_time - start_time}秒")


if __name__ == "__main__":
    main()
