import torch.nn.functional as F
from torch import Tensor
from openmind import AutoTokenizer, AutoModelForCausalLM
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
        default="pretrainmodel/Mistral-7B-Instruct-v0.1-GGUF",
    )
    args = parser.parse_args()
    return args

def evaluate_on_device(model_path, input_texts, device):
    filename = "mistral-7b-instruct-v0.1.Q2_K.gguf"
    tokenizer = AutoTokenizer.from_pretrained(model_path, gguf_file=filename, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, gguf_file=filename, device_map=device
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                                 
    start_time = time.time()
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**batch_dict)
    embeddings =outputs.last_hidden_state[:,0]
    #(Optionally)normalizeembeddings
    embeddings =F.normalize(embeddings, p=2, dim=1)
    end_time = time.time()
    print(f"硬件环境：{device}上推理执行时间：{end_time - start_time}秒")
    return embeddings

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

    # Calculate the cosine similarity between CPu and NPu embeddings
    
    cosine_similarity =F.cosine_similarity(cpu_embeddings_on_npu, npu_embeddings, dim=1)
    print(f"Cosine Similarity Between CPU and NPU embeddings: {cosine_similarity}")
else:
    print("NPU is not available.")
