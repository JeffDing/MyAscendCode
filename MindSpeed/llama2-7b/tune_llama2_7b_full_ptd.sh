#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6004
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="./ckpt"
DATA_PATH="./finetune_dataset/alpaca"
TOKENIZER_MODEL="./model_from_hf/llama2-7b-hf"
CKPT_LOAD_DIR="./model_weights/Llama2-mcore"

TP=8
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --load ${CKPT_LOAD_DIR} \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --tokenizer-not-use-fast \
    --prompt-type llama2 \
    --variable-seq-lengths \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 4 \
    --global-batch-size 128 \
    --make-vocab-size-divisible-by 1 \
    --lr 5e-5 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 1 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --bf16 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"


OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    | tee logs/tune_llama2_7b.log
