# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 权重格式转换
python convert_ckpt.py \
   --use-mcore-models \
   --model-type-hf internlm2 \
   --model-type GPT \
   --load-model-type hf \
   --save-model-type mg \
   --w-pack True \
   --params-dtype bf16 \
   --target-tensor-parallel-size 8 \
   --target-pipeline-parallel-size 1 \
   --load-dir ./model_from_hf/internlm2_5-7b-chat/ \
   --save-dir ./model_weights/internlm2_5-7b-chat-mcore-tp8-pp1/ \
   --tokenizer-model ./model_from_hf/internlm25_7b_hf/tokenizer.model
