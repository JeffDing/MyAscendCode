# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换，设置需要的并行配置，--num-layers-per-virtual-pipeline-stage 5，--params-dtype bf16 结合需要使用
python convert_ckpt.py \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 1 \
    --load-dir ./ckpt/ \
    --save-dir ./model_from_hf/llama2-7b-hf_tune/ \
    --use-mcore-models
