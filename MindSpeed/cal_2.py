import numpy as np
from scipy.spatial.distance import cosine

def load_npy_file(file_path):
    """
    加载 .npy 文件并返回其内容。
    """
    return np.load(file_path)

def compute_cosine_similarity(array1, array2):
    """
    计算两个数组的余弦相似度。
    """
    # 确保数组形状相同
    if array1.shape != array2.shape:
        raise ValueError("两个数组的形状不匹配，无法计算余弦相似度。")
    
    # 计算余弦相似度
    # 余弦相似度的范围是 [-1, 1]，1 表示完全相似，-1 表示完全相反，0 表示正交
    return 1 - cosine(array1.flatten(), array2.flatten())

def main():
    # 文件路径
    file1_path = ""
    file2_path = ""

    # 加载 .npy 文件
    array1 = load_npy_file(file1_path)
    array2 = load_npy_file(file2_path)

    # 计算余弦相似度
    similarity = compute_cosine_similarity(array1, array2)

    # 输出结果
    print(f"余弦相似度: {similarity}")

if __name__ == "__main__":
    main()
