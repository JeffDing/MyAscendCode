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
# checkout commit from MindSpeed core_r0.7.0 in 2024.12.13
git checkout 4045864e6df
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install -e .
cd ../MindSpeed-LLM
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install binutils
export PATH=/home/ma-user/anaconda3/envs/ModelLink/bin:$PATH
cp /home/ma-user/anaconda3/envs/ModelLink/bin/aarch64-conda-linux-gnu-ld /home/ma-user/anaconda3/envs/ModelLink/bin/ld # 非必要

# 下载模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download meta-llama/Llama-2-7b --local-dir Llama-2-7b --local-dir-use-symlinks False

# 转换数据集
mkdir ./finetune_dataset
python ./preprocess_data.py \
    --input ./dataset/alpaca/data \
    --tokenizer-name-or-path ./model_from_hf/llama2-7b-hf \
    --output-prefix ./finetune_dataset/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type llama2
    
# 转换模型
bash examples/mcore/llama2/ckpt_convert_llama2_hf2mcore.sh

# 微调
bash examples/mcore/llama2/tune_llama2_7b_full_ptd.sh

# 推理
bash examples/mcore/llama2/generate_llama2_7b_ptd.sh


# 复制配置文件
rm -rf llama2-7b/.ipynb_checkpoints/
cp -r llama2-7b/* MindSpeed-LLM/examples/mcore/llama2/
