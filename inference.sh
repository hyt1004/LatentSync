#!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/second_stage.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --guidance_scale 1.0 \
    --video_path "/disk4/huyutao/test_sample/zhulang/zhulang_1080p10s.mp4" \
    --audio_path "/disk4/huyutao/test_sample/audio/kanghui_train_30s.mp3" \
    --video_out_path "kh_30s_out.mp4"
