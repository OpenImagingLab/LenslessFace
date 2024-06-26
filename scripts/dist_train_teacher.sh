#!/usr/bin/env bash

check_port() {
    port=$1
    result=$(netstat -tln | awk '{print $4}' | grep ":$port$")

    if [ -n "$result" ]; then
        return 1
    else
        return 0
    fi
}

find_available_port() {
    start_port=$1
    end_port=$((start_port + 10))

    for ((port=start_port; port<=end_port; port++))
    do
        check_port $port
        if [ $? -eq 0 ]; then
            echo $port
            return $port
        fi
    done
}


CONFIG=$1
GPUS=`nvidia-smi --list-gpus | wc -l`
# WORRDIR is logs + the file name of CONFIG without .py

WORRDIR=$(echo logs/${CONFIG%.*} | sed 's/configs\///g')
mkdir -p $WORRDIR
echo "work dir: $WORRDIR"
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=`find_available_port 29500`
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/mmcls/train_teacher.py \
    $CONFIG \
    --seed 0 \
    --work-dir $WORRDIR \
    --launcher pytorch ${@:3}\
    
