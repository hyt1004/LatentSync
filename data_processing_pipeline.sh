#!/bin/bash

python -m preprocess.data_processing_pipeline \
    --total_num_workers 20 \
    --per_gpu_num_workers 20 \
    --resolution 256 \
    --sync_conf_threshold 3 \
    --temp_dir ./datas/temp/ \
    --input_dir /disk4/huyutao/modelzoo/digital_human/LatentSync/0finetune_datas/
