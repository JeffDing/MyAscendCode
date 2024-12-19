import torch.nn.functional as F
from torch import Tensor
from openmind import AutoTokenizer, AutoModel
from transformers import AutoModelForSeq2SeqLM
import torch
import sys
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to model",
        default="pretrainmodel/t5_translate_en_ru_zh_large_1024_v2",
    )
    args = parser.parse_args()
    return args

def evaluate_on_device(model_path, input_texts, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                                 
    start_time = time.time()
    data = "translate English to Russian：That is good."
    encoded = tokenizer([data], return_tensors="pt").to(device)
    translation = model.generate(**encoded)
    print(translation)
    end_time = time.time()
    print(f"硬件环境：{device}上推理执行时间：{end_time - start_time}秒")
    return translation

args = parse_args()
model_path = args.model_name_or_path
  
input_texts =["Give me a short introduction to large language model."]
#Evaluate on CPU
cpu_embeddings =evaluate_on_device(model_path,input_texts,"cpu")

print(f"CPU Embeddings:{cpu_embeddings}")
# Evaluate on NPu if available
if torch.npu.is_available():
    npu_embeddings=evaluate_on_device(model_path, input_texts,"npu:0")
    
    print(f"NPU Embeddings:{npu_embeddings}")
    #Move CPu embeddings to NPU
    cpu_embeddings_on_npu=cpu_embeddings.to("npu:0")
else:
    print("NPU is not available.")

