#!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/second_stage.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --guidance_scale 1.0 \
    --video_path "/disk4/huyutao/test_sample/video/05-10s-1080p.mp4" \
    --audio_path "/disk4/huyutao/test_sample/audio/kanghui_train_10s.mp3" \
    --video_out_path "latentsync-05.mp4"
