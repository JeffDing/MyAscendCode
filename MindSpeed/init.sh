# 创建环境
# python3.8
conda create -n ModelLink python=3.8
conda activate ModelLink

pip install torch==2.1.0 torch_npu==2.1.0.post8 torchvision==0.16.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -U setuptools

git clone https://gitee.com/ascend/apex
yum install patch
cd apex
bash scripts/build.sh --python=3.8
cd apex/dist/
pip3 uninstall apex
pip3 install --upgrade apex-0.1

# 拉取环境
conda activate ModelLink
git clone https://gitee.com/ascend/MindSpeed-LLM.git 
git clone https://github.com/nvidia/Megatron-LM.git
# 如果下载不了github的话使用国产镜像 
git clone https://openi.pcl.ac.cn/JeffDing/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.7.0
cp -r megatron ../MindSpeed-LLM/
cd ..
cd MindSpeed-LLM
mkdir logs
mkdir model_from_hf
mkdir dataset
mkdir ckpt

# 安装依赖
cd ..
git clone https://gitee.com/ascend/MindSpeed.git
rm -rf /home/ma-user/anaconda3/envs/ModelLink/lib/python3.8/site-packages/mindspeed.egg-link
cd MindSpeed
# checkout commit from MindSpeed core_r0.7.0 in 2024.11.04
git checkout c9d20b5 
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -e .
cd ../MindSpeed-LLM
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install binutils
export PATH=/home/ma-user/anaconda3/envs/ModelLink/bin:$PATH
cp /home/ma-user/anaconda3/envs/ModelLink/bin/aarch64-conda-linux-gnu-ld /home/ma-user/anaconda3/envs/ModelLink/bin/ld # 非必要


# 转换模型
bash examples/mcore/internlm25_chat/ckpt_convert_internlm25_chat_7b_hf2mcore.sh
    
# 数据集下载
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download lsb/enwiki20230101 --local-dir ./enwiki20230101 --local-dir-use-symlinks False
huggingface-cli download --repo-type dataset --resume-download datasets/hendrycks_test --local-dir ./hendrycks_test --local-dir-use-symlinks False



# 转换数据集
bash examples/mcore/internlm25_chat/data_convert_internlm25_pretrain.sh

# 预训练
bash examples/mcore/internlm25_chat/pretrain_internlm25_chat_7b_32k_ptd.sh

# 推理
bash examples/mcore/internlm25_chat/generate_internlm25_chat_7b_ptd.sh 

# 评估
bash examples/mcore/internlm25_chat/evaluate_internlm25_chat_7b_ptd.sh 
