#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# distributed config
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0

# modify script model path and tokenizer path
TOKENIZER_PATH="./model_from_hf/internlm2_5-7b-chat/"
CHECKPOINT="./ckpt/"

# configure task and data path
DATA_PATH="/tmp/code/dataset/mmlu/test/"
TASK="mmlu"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

torchrun ${DISTRIBUTED_ARGS} evaluation.py   \
       --no-chat-template \
       --task-data-path ${DATA_PATH} \
       --task ${TASK} \
       --use-mcore-models \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 32  \
       --hidden-size 4096 \
       --ffn-hidden-size 14336 \
       --group-query-attention \
       --num-query-groups 8 \
       --position-embedding-type rope \
       --norm-epsilon 1e-5 \
       --rotary-base 50000000 \
       --seq-length 32768 \
       --max-new-tokens 1 \
       --micro-batch-size 1 \
       --num-attention-heads 32 \
       --max-position-embeddings 32768 \
       --padded-vocab-size 92544 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --swiglu \
       --load ${CHECKPOINT} \
       --disable-bias-linear \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --normalization RMSNorm \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --make-vocab-size-divisible-by 1 \
       --seed 42 \
       | tee logs/evaluate_internlm25_7b_mcore_${TASK}.log
