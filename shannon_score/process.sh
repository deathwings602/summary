#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=13672
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

export CUDA_VISIBLE_DEVICES=6

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"


BASE_PATH=$(cd $(dirname "${BASH_SOURCE[0]}") >/dev/null && pwd)
DATASET="EcoNewSum"
INPUT_FILE="guangMingDaily-2.json"
MODEL_CONFIG_DIR=${CPM_CACHE_PATH}/cpm1-small
EPOCH=3
CKPT_STEPS=0
OUTPUT_FILE=${BASE_PATH}/infer_results/${INPUT_FILE}/${EPOCH}-${CKPT_STEPS}.jsonl

if [ ! -d ${BASE_PATH}/infer_results ]; then
    mkdir ${BASE_PATH}/infer_results
fi

if [ ! -d ${BASE_PATH}/infer_results/${INPUT_FILE} ]; then
    mkdir ${BASE_PATH}/infer_results/${INPUT_FILE}
fi

OPTS=""
OPTS+=" --max-length 1536"
OPTS+=" --dataset ${DATASET}"
OPTS+=" --model-config ${MODEL_CONFIG_DIR}/config.json"
OPTS+=" --vocab-file ${MODEL_CONFIG_DIR}/vocab.txt"
OPTS+=" --load /data/disk3/private/zhaoxinhao/CPM1/cpm1-small"
OPTS+=" --input-file ${CPM_TRAIN_DATA_PATH}/${DATASET}/${INPUT_FILE}"
OPTS+=" --output-file ${OUTPUT_FILE}"
OPTS+=" --span-length 100"
OPTS+=" --temperature 1"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0"
OPTS+=" --no-repeat-ngram-size 0"
OPTS+=" --repetition-penalty 2"
OPTS+=" --length-penalty 1.5"
OPTS+=" --beam-size 5"
OPTS+=" --batch-size 1"
# OPTS+=" --random-sample"

export CUDA_VISIBLE_DEVICES=6

CMD="python3 -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/shannon_entropy.py ${OPTS}"
echo ${CMD}

${CMD} 2>&1 | tee ${BASE_PATH}/infer_results/${INPUT_FILE}/infer-${EPOCH}-${CKPT_STEPS}.log

cat ${OUTPUT_FILE}.* > ${OUTPUT_FILE}
#rm ${OUTPUT_FILE}.*
