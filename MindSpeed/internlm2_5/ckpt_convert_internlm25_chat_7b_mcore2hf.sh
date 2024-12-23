# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置并行策略
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf internlm2 \
    --load-model-type mg \
    --save-model-type hf \
    --w-pack True \
    --params-dtype bf16 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./model_weights/internlm2_5-7b-chat-mcore-tp1-pp1/ \
    --save-dir ./model_from_hf/internlm2_5-7b-chat/ # 需要填入原始HF模型路径，新权重会存于./model_from_hf/internlm25_7b_hf/mg2hg/