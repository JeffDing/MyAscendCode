import numpy as np
import torch

def cosine_similarity_torch(vec1, vec2):
    """计算两个张量的余弦相似度，避免 NaN"""
    dot_product = torch.sum(vec1 * vec2, dim=-1)
    norm_vec1 = torch.norm(vec1, dim=-1)
    norm_vec2 = torch.norm(vec2, dim=-1)
    # 避免分母为零
    denominator = norm_vec1 * norm_vec2
    denominator = torch.where(denominator == 0, torch.tensor(1.0, device=denominator.device), denominator)
    return dot_product / denominator

# 加载 .npy 文件，并转换为 Torch 张量
file1 = "./forward_out/hf_internlm2_5_7b_chat_logits_fp16.npy"  # 替换为你的第一个 .npy 文件路径
file2 = "./forward_out/modellink_hf_internlm2_5_7b_chat_logits_fp16.npy"  # 替换为你的第二个 .npy 文件路径

data1 = torch.tensor(np.load(file1), dtype=torch.float16)
data2 = torch.tensor(np.load(file2), dtype=torch.float16)

# 提升精度到 float32
data1 = data1.to(torch.float32)
data2 = data2.to(torch.float32)

# 检查形状是否一致
if data1.shape != data2.shape:
    raise ValueError(f"两个文件的形状不一致: {data1.shape} vs {data2.shape}")

# 输出两个文件的张量内容（显示前 5 个元素）
print("文件 1 的张量内容（前 5 个元素）：")
print(data1.flatten()[:5])  # 展平后取前 5 个
print("文件 2 的张量内容（前 5 个元素）：")
print(data2.flatten()[:5])

# 逐特征对比 (对 2048 个特征逐一计算余弦相似度)
if data1.shape[0] == 1:  # Batch 为 1
    similarities = cosine_similarity_torch(data1[0], data2[0])  # 逐特征比较
    for i, sim in enumerate(similarities):
        # 限制相似度范围到 [-1, 1]
        sim_clamped = torch.clamp(sim, -1, 1).item()
        similarity_percentage = ((sim_clamped + 1) / 2) * 100  # 转换为百分比
        print(f"特征 {i+1} 的相似度: {similarity_percentage:.2f}%")

# 整体对比 (将后两维展平为大向量)
data1_flat = data1.view(1, -1)
data2_flat = data2.view(1, -1)
similarity = cosine_similarity_torch(data1_flat, data2_flat)

# 限制整体相似度范围到 [-1, 1]
similarity_clamped = torch.clamp(similarity, -1, 1).item()
similarity_percentage = ((similarity_clamped + 1) / 2) * 100  # 转换为百分比
print(f"整体相似度: {similarity_percentage:.2f}%")