from openmind import AutoTokenizer, AutoModelForCausalLM, is_torch_npu_available
from openmind_hub import snapshot_download
import torch.nn.functional as F
from torch import Tensor
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to model",
        default="jeffding/japanese-gpt2-xsmall-openmind",
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device)
    
    start_time = time.time()
    
    input_text = "BCCI ने टी-20 वर्ल्ड कप के बीच जिम्बाब्वे सीरीज "
    input_ids = tokenizer.encode(input_text,
                return_tensors="pt").to(device)

    outputs = model.generate(input_ids, max_new_tokens=100,
              do_sample=True, top_k=50,
              top_p=0.95, temperature=0.7)

    print(tokenizer.decode(outputs[0]))
    
    end_time = time.time()
    print(f"硬件环境：{device},推理执行时间：{end_time - start_time}秒")
    
if __name__ == "__main__":
    main()
