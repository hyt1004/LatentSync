# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import argparse
import shutil
import datetime
import logging
from omegaconf import OmegaConf

from tqdm.auto import tqdm
from einops import rearrange

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.logging import get_logger
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed

from latentsync.data.unet_dataset import UNetDataset
from latentsync.models.unet import UNet3DConditionModel
from latentsync.models.syncnet import SyncNet
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.utils.util import (
    init_dist,
    cosine_loss,
    reversed_forward,
)
from latentsync.utils.util import plot_loss_chart, gather_loss
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.trepa import TREPALoss
from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector
from eval.eval_sync_conf import syncnet_eval
import lpips


logger = get_logger(__name__)


def main(config):
    # 初始化分布式训练
    # local_rank = init_dist()  # 获取本地进程的排名
    # global_rank = dist.get_rank()  # 获取全局进程的排名
    # num_processes = dist.get_world_size()  # 获取总进程数
    # is_main_process = global_rank == 0  # 判断是否为主进程

    # seed = config.run.seed + global_rank  # 设置随机种子
    set_seed(config.run.seed)

    # 日志文件夹
    folder_name = "train" + datetime.datetime.now().strftime(f"-%Y_%m_%d-%H:%M:%S")
    output_dir = os.path.join(config.data.train_output_dir, folder_name)

    # 在每个进程上记录配置以便调试
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 处理输出文件夹的创建
    # if is_main_process:
    diffusers.utils.logging.set_verbosity_info()
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)  # 创建检查点目录
    os.makedirs(f"{output_dir}/val_videos", exist_ok=True)  # 创建验证视频目录
    os.makedirs(f"{output_dir}/loss_charts", exist_ok=True)  # 创建损失图表目录
    shutil.copy(config.unet_config_path, output_dir)  # 复制配置文件
    shutil.copy(config.data.syncnet_config_path, output_dir)  # 复制同步网络配置文件
    # endif

    # device = torch.device(local_rank)  # 设置设备
    device = torch.device("cuda:0")  # 设置设备
    trepa_device = torch.device("cuda:1")  # 设置TREPA设备

    noise_scheduler = DDIMScheduler.from_pretrained("configs")  # 初始化噪声调度器

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)  # 加载VAE模型
    vae.config.scaling_factor = 0.18215  # 设置缩放因子
    vae.config.shift_factor = 0  # 设置偏移因子
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)  # 计算VAE缩放因子
    vae.requires_grad_(False)  # 冻结VAE参数
    vae.to(device)  # 将VAE移动到设备上

    syncnet_eval_model = SyncNetEval(device=device)  # 初始化同步网络评估模型
    syncnet_eval_model.loadParameters("checkpoints/auxiliary/syncnet_v2.model")  # 加载syncnet模型参数

    syncnet_detector = SyncNetDetector(device=device, detect_results_dir="detect_results")  # 初始化同步网络检测器

    # 根据cross_attention_dim选择whisper模型路径
    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device=device,
        audio_embeds_cache_dir=config.data.audio_embeds_cache_dir,
        num_frames=config.data.num_frames,
    )  # 初始化音频编码器

    unet, resume_global_step = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        config.ckpt.resume_ckpt_path,  # 加载检查点
        device=device,
    )  # 加载UNet模型

    # 如果配置中需要音频层并且使用同步网络
    if config.model.add_audio_layer and config.run.use_syncnet:
        syncnet_config = OmegaConf.load(config.data.syncnet_config_path)  # 加载同步网络配置
        if syncnet_config.ckpt.inference_ckpt_path == "":
            raise ValueError("SyncNet path is not provided")  # 检查同步网络路径是否提供
        syncnet = SyncNet(OmegaConf.to_container(syncnet_config.model)).to(device=device, dtype=torch.float16)  # 初始化同步网络
        syncnet_checkpoint = torch.load(syncnet_config.ckpt.inference_ckpt_path, map_location=device)  # 加载同步网络检查点
        syncnet.load_state_dict(syncnet_checkpoint["state_dict"])  # 加载状态字典
        syncnet.requires_grad_(False)  # 冻结同步网络参数

    unet.requires_grad_(False)  # 禁止UNet参数更新
    # 选择部分参数允许更新
    # 部分层使用smaller lr
    trainable_params = []
    for name, param in unet.named_parameters():
        if "mid_block" in name or "up_blocks" in name or "conv_norm_out" in name or "conv_out" in name:
            param.requires_grad = True  # 允许更新
            trainable_params.append(param)

    # 如果需要缩放学习率
    # if config.optimizer.scale_lr:
        # config.optimizer.lr = config.optimizer.lr * num_processes  # 根据进程数缩放学习率

    optimizer = torch.optim.AdamW(trainable_params, lr=config.optimizer.lr)  # 初始化优化器

    # if is_main_process:
    logger.info(f"trainable params number: {len(trainable_params)}")  # 记录可训练参数数量
    logger.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")  # 记录可训练参数规模
    # end if

    # 启用xformers
    if config.run.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()  # 启用xformers内存高效注意力
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")  # 检查xformers是否可用

    # 启用梯度检查点
    if config.run.enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()  # 启用梯度检查点

    # 获取train dataset
    train_dataset = UNetDataset(config.data.train_data_dir, config)  # 初始化数据集
    # distributed_sampler = DistributedSampler(
    #     train_dataset,
    #     num_replicas=num_processes,
    #     rank=global_rank,
    #     shuffle=True,
    #     seed=config.run.seed,
    # )  # 初始化分布式采样器

    # 创建dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True, # False,
        # sampler=distributed_sampler,
        num_workers=config.data.num_workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
    )

    # 获取训练迭代次数
    if config.run.max_train_steps == -1:
        assert config.run.max_train_epochs != -1
        config.run.max_train_steps = config.run.max_train_epochs * len(train_dataloader)  # 计算最大训练步数

    # 学习率调度器
    lr_scheduler = get_scheduler(
        config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.lr_warmup_steps,
        num_training_steps=config.run.max_train_steps,
    )

    # 如果需要感知损失和像素空间监督
    if config.run.perceptual_loss_weight != 0 and config.run.pixel_space_supervise:
        lpips_loss_func = lpips.LPIPS(net="vgg").to(device)  # 初始化LPIPS损失函数

    if config.run.trepa_loss_weight != 0 and config.run.pixel_space_supervise:
        trepa_loss_func = TREPALoss(device=trepa_device, ckpt_path="./checkpoints/auxiliary/vit_g_hybrid_pt_1200e_ssv2_ft.pth")  # 初始化TREPA损失函数

    # validation pipeline
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=noise_scheduler,
    ).to(device)  # 初始化Lipsync管道
    pipeline.set_progress_bar_config(disable=True)  # 禁用进度条

    # DDP包装器
    # unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)  # 包装UNet以支持分布式数据并行

    # 重新计算每个epoch的更新步骤
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))  # 计算每个epoch的更新步骤
    num_train_epochs = math.ceil(config.run.max_train_steps / num_update_steps_per_epoch)  # 计算训练epoch数量

    # 开始训练
    # total_batch_size = config.data.batch_size * num_processes  # 计算总批量大小
    total_batch_size = config.data.batch_size

    # if is_main_process:
    logger.info("***** Running training *****")  # 记录训练开始信息
    logger.info(f"  Num examples = {len(train_dataset)}")  # 记录训练样本数量
    logger.info(f"  Num Epochs = {num_train_epochs}")  # 记录训练epoch数量
    logger.info(f"  Instantaneous batch size per device = {config.data.batch_size}")  # 记录每个设备的批量大小
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")  # 记录总批量大小
    logger.info(f"  Total optimization steps = {config.run.max_train_steps}")  # 记录总优化步骤
    # end if

    resume_global_step = 0
    global_step = resume_global_step  # 初始化全局步骤
    first_epoch = resume_global_step // num_update_steps_per_epoch  # 计算第一个epoch

    # 只在每台机器上显示一次进度条
    progress_bar = tqdm(
        range(0, config.run.max_train_steps),
        initial=resume_global_step,
        desc="Steps",
        # disable=not is_main_process,
    )

    train_step_list = []  # 训练步骤列表
    sync_loss_list = []  # 同步损失列表
    recon_loss_list = []  # 重建损失列表

    val_step_list = []  # 验证步骤列表
    sync_conf_list = []  # 同步置信度列表

    # 支持混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config.run.mixed_precision_training else None  # 初始化梯度缩放器

    for epoch in range(first_epoch, num_train_epochs):
        # train_dataloader.sampler.set_epoch(epoch)  # 设置当前epoch
        unet.train()  # 设置UNet为训练模式

        for step, batch in enumerate(train_dataloader):
            ### >>>> 训练 >>>> ###

            if config.model.add_audio_layer:
                if batch["mel"] != []:
                    # mel 用于计算 sync loss
                    mel = batch["mel"].to(device, dtype=torch.float16)  # 将音频数据移动到设备

                # audio embeds 用于输入unet中进行特征融合
                audio_embeds_list = []  # 初始化音频嵌入列表
                try:
                    for idx in range(len(batch["video_path"])):
                        video_path = batch["video_path"][idx]  # 获取视频路径
                        start_idx = batch["start_idx"][idx]  # 获取开始索引

                        with torch.no_grad():
                            audio_feat = audio_encoder.audio2feat(video_path)  # 提取音频特征
                        audio_embeds = audio_encoder.crop_overlap_audio_window(audio_feat, start_idx)  # 裁剪音频嵌入
                        audio_embeds_list.append(audio_embeds)  # 添加到列表
                except Exception as e:
                    logger.info(f"{type(e).__name__} - {e} - {video_path}")  # 记录异常信息
                    continue
                audio_embeds = torch.stack(audio_embeds_list)  # (B, 16, 50, 384)
                audio_embeds = audio_embeds.to(device, dtype=torch.float16)  # 移动到设备
            else:
                audio_embeds = None  # 如果没有音频层，设置为None

            # 将视频转换为潜在空间
            gt_images = batch["gt"].to(device, dtype=torch.float16)  # 获取真实图像
            gt_masked_images = batch["masked_gt"].to(device, dtype=torch.float16)  # 获取掩码图像
            mask = batch["mask"].to(device, dtype=torch.float16)  # 获取掩码
            ref_images = batch["ref"].to(device, dtype=torch.float16)  # 获取参考图像

            gt_images = rearrange(gt_images, "b f c h w -> (b f) c h w")  # 重排真实图像
            gt_masked_images = rearrange(gt_masked_images, "b f c h w -> (b f) c h w")  # 重排掩码图像
            mask = rearrange(mask, "b f c h w -> (b f) c h w")  # 重排掩码
            ref_images = rearrange(ref_images, "b f c h w -> (b f) c h w")  # 重排参考图像

            # 使用vae进行encode
            with torch.no_grad():
                gt_latents = vae.encode(gt_images).latent_dist.sample()  # 编码真实图像
                gt_masked_images = vae.encode(gt_masked_images).latent_dist.sample()  # 编码掩码图像
                ref_images = vae.encode(ref_images).latent_dist.sample()  # 编码参考图像

            mask = torch.nn.functional.interpolate(mask, size=config.data.resolution // vae_scale_factor)  # 插值掩码

            gt_latents = (
                rearrange(gt_latents, "(b f) c h w -> b c f h w", f=config.data.num_frames) - vae.config.shift_factor
            ) * vae.config.scaling_factor  # 处理潜在变量
            gt_masked_images = (
                rearrange(gt_masked_images, "(b f) c h w -> b c f h w", f=config.data.num_frames)
                - vae.config.shift_factor
            ) * vae.config.scaling_factor  # 处理掩码图像
            ref_images = (
                rearrange(ref_images, "(b f) c h w -> b c f h w", f=config.data.num_frames) - vae.config.shift_factor
            ) * vae.config.scaling_factor  # 处理参考图像
            mask = rearrange(mask, "(b f) c h w -> b c f h w", f=config.data.num_frames)  # 处理掩码

            # 生成要添加到潜在变量的噪声
            if config.run.use_mixed_noise:
                # 参考论文: https://arxiv.org/abs/2305.10474
                noise_shared_std_dev = (config.run.mixed_noise_alpha**2 / (1 + config.run.mixed_noise_alpha**2)) ** 0.5
                noise_shared = torch.randn_like(gt_latents) * noise_shared_std_dev  # 生成共享噪声
                noise_shared = noise_shared[:, :, 0:1].repeat(1, 1, config.data.num_frames, 1, 1)  # 重复噪声

                noise_ind_std_dev = (1 / (1 + config.run.mixed_noise_alpha**2)) ** 0.5
                noise_ind = torch.randn_like(gt_latents) * noise_ind_std_dev  # 生成独立噪声
                noise = noise_ind + noise_shared  # 合并噪声
            else:
                noise = torch.randn_like(gt_latents)  # 生成噪声
                noise = noise[:, :, 0:1].repeat(
                    1, 1, config.data.num_frames, 1, 1
                )  # 对所有帧使用相同的噪声，参考论文: https://arxiv.org/abs/2308.09716

            bsz = gt_latents.shape[0]  # 获取批量大小

            # 为每个视频采样一个随机时间步
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=gt_latents.device)
            timesteps = timesteps.long()  # 转换为长整型

            # 根据每个时间步的噪声幅度向潜在变量添加噪声
            # （这是前向扩散过程）
            noisy_tensor = noise_scheduler.add_noise(gt_latents, noise, timesteps)

            # 根据预测类型获取损失目标
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise  # 目标为噪声
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError  # 未实现的预测类型
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")  # 未知的预测类型

            unet_input = torch.cat([noisy_tensor, mask, gt_masked_images, ref_images], dim=1)  # 拼接输入

            # 预测噪声并计算损失
            # 混合精度训练
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.run.mixed_precision_training):
                pred_noise = unet(unet_input, timesteps, encoder_hidden_states=audio_embeds).sample  # 预测噪声

            if config.run.recon_loss_weight != 0:
                recon_loss = F.mse_loss(pred_noise.float(), target.float(), reduction="mean")  # 计算重建损失
            else:
                recon_loss = 0  # 如果不需要重建损失，设置为0

            pred_latents = reversed_forward(noise_scheduler, pred_noise, timesteps, noisy_tensor)  # 还原预测的加噪声前的latent

            if config.run.pixel_space_supervise:
                pred_images = vae.decode(
                    rearrange(pred_latents, "b c f h w -> (b f) c h w") / vae.config.scaling_factor
                    + vae.config.shift_factor
                ).sample  # 解码预测图像 [16, 3, 256, 256]

            # LPIPS Loss
            if config.run.perceptual_loss_weight != 0 and config.run.pixel_space_supervise:
                pred_images_perceptual = pred_images[:, :, pred_images.shape[2] // 2 :, :]  # 获取感知图像 [16, 3, 128, 256]
                gt_images_perceptual = gt_images[:, :, gt_images.shape[2] // 2 :, :]  # 获取真实图像 [16, 3, 128, 256]
                lpips_loss = lpips_loss_func(pred_images_perceptual.float(), gt_images_perceptual.float()).mean()  # 计算LPIPS损失
            else:
                lpips_loss = 0  # 如果不需要感知损失，设置为0

            # TREPA Loss
            if config.run.trepa_loss_weight != 0 and config.run.pixel_space_supervise:
                trepa_pred_images = rearrange(pred_images, "(b f) c h w -> b c f h w", f=config.data.num_frames)  # 重排预测图像 [1, 3, 16, 256, 256]
                trepa_gt_images = rearrange(gt_images, "(b f) c h w -> b c f h w", f=config.data.num_frames)  # 重排真实图像 [1, 3, 16, 256, 256]
                trepa_loss = trepa_loss_func(trepa_pred_images.to(trepa_device), trepa_gt_images.to(trepa_device))  # 计算TREPA损失
                trepa_loss = trepa_loss.to(device)
            else:
                trepa_loss = 0  # 如果不需要TREPA损失，设置为0

            # Sync Loss
            if config.model.add_audio_layer and config.run.use_syncnet:
                if config.run.pixel_space_supervise:
                    syncnet_input = rearrange(pred_images, "(b f) c h w -> b (f c) h w", f=config.data.num_frames)  # 重排同步网络输入
                else:
                    syncnet_input = rearrange(pred_latents, "b c f h w -> b (f c) h w")  # 重排潜在变量输入

                if syncnet_config.data.lower_half:
                    height = syncnet_input.shape[2]  # 获取高度
                    syncnet_input = syncnet_input[:, :, height // 2 :, :]  # 仅使用下半部分
                ones_tensor = torch.ones((config.data.batch_size, 1)).float().to(device=device)  # 创建全1张量
                vision_embeds, audio_embeds = syncnet(syncnet_input, mel)  # 获取视觉和音频嵌入
                sync_loss = cosine_loss(vision_embeds.float(), audio_embeds.float(), ones_tensor).mean()  # 计算同步损失
                # sync_loss_list.append(gather_loss(sync_loss, device))  # 收集同步损失
                sync_loss_list.append(sync_loss.item())

            else:
                sync_loss = 0  # 如果不需要同步损失，设置为0

            loss = (
                recon_loss * config.run.recon_loss_weight
                + sync_loss * config.run.sync_loss_weight
                + lpips_loss * config.run.perceptual_loss_weight
                + trepa_loss * config.run.trepa_loss_weight
            )  # 计算总损失

            train_step_list.append(global_step)  # 添加当前步骤到训练步骤列表
            if config.run.recon_loss_weight != 0:
                # recon_loss_list.append(gather_loss(recon_loss, device))  # 收集重建损失
                recon_loss_list.append(recon_loss.item())

            optimizer.zero_grad()  # 清零梯度

            # 反向传播
            if config.run.mixed_precision_training:
                scaler.scale(loss).backward()  # 混合精度，缩放损失，反向传播
                """ >>> 梯度裁剪 >>> """
                scaler.unscale_(optimizer)  # 反缩放梯度
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config.optimizer.max_grad_norm)  # 裁剪梯度，防止梯度爆炸，稳定训练
                """ <<< 梯度裁剪 <<< """
                scaler.step(optimizer)  # 更新梯度
                scaler.update()  # 更新缩放器
            else:
                loss.backward()  # 反向传播
                """ >>> 梯度裁剪 >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), config.optimizer.max_grad_norm)  # 裁剪梯度
                """ <<< 梯度裁剪 <<< """
                optimizer.step()  # 更新梯度

            # 检查注意力块的梯度以进行调试
            # print(unet.module.up_blocks[3].attentions[2].transformer_blocks[0].audio_cross_attn.attn.to_q.weight.grad)

            lr_scheduler.step()  # 更新学习率
            progress_bar.update(1)  # 更新进度条
            global_step += 1  # 增加全局步骤

            ### <<<< 训练 <<<< ###

            # 保存检查点并进行验证
            # if is_main_process and (global_step % config.ckpt.save_ckpt_steps == 0):
            if global_step % config.ckpt.save_ckpt_steps == 0:
                if config.run.recon_loss_weight != 0:
                    plot_loss_chart(
                        os.path.join(output_dir, f"loss_charts/recon_loss_chart-{global_step}.png"),
                        ("Reconstruction loss", train_step_list, recon_loss_list),  # 绘制重建损失图表
                    )
                if config.model.add_audio_layer:
                    if sync_loss_list != []:
                        plot_loss_chart(
                            os.path.join(output_dir, f"loss_charts/sync_loss_chart-{global_step}.png"),
                            ("Sync loss", train_step_list, sync_loss_list),  # 绘制同步损失图表
                        )
                model_save_path = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}.pt")  # 检查点保存路径
                if isinstance(unet, torch.nn.DataParallel) or  isinstance(unet, torch.nn.parallel.DistributedDataParallel):
                    state_dict = unet.module.state_dict()  # 获取模型状态字典
                else:
                    state_dict = unet.state_dict()  # 获取模型状态字典
                try:
                    torch.save(state_dict, model_save_path)  # 保存模型状态字典
                    logger.info(f"Saved checkpoint to {model_save_path}")  # 记录保存信息
                except Exception as e:
                    logger.error(f"Error saving model: {e}")  # 记录保存错误

                # 验证
                logger.info("Running validation... ")  # 记录验证开始信息

                validation_video_out_path = os.path.join(output_dir, f"val_videos/val_video_{global_step}.mp4")  # 验证视频输出路径
                validation_video_mask_path = os.path.join(output_dir, f"val_videos/val_video_mask.mp4")  # 验证视频掩码路径

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pipeline(
                        config.data.val_video_path,
                        config.data.val_audio_path,
                        validation_video_out_path,
                        validation_video_mask_path,
                        num_frames=config.data.num_frames,
                        num_inference_steps=config.run.inference_steps,
                        guidance_scale=config.run.guidance_scale,
                        weight_dtype=torch.float16,
                        width=config.data.resolution,
                        height=config.data.resolution,
                        mask=config.data.mask,
                    )  # 运行验证

                logger.info(f"Saved validation video output to {validation_video_out_path}")  # 记录验证视频保存信息

                val_step_list.append(global_step)  # 添加当前步骤到验证步骤列表

                if config.model.add_audio_layer:
                    try:
                        _, conf = syncnet_eval(syncnet_eval_model, syncnet_detector, validation_video_out_path, "temp")  # 运行同步网络评估
                    except Exception as e:
                        logger.info(e)  # 记录异常信息
                        conf = 0  # 设置置信度为0
                    sync_conf_list.append(conf)  # 添加置信度到列表
                    plot_loss_chart(
                        os.path.join(output_dir, f"loss_charts/sync_conf_chart-{global_step}.png"),
                        ("Sync confidence", val_step_list, sync_conf_list),  # 绘制同步置信度图表
                    )

            logs = {"step_loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}  # 记录损失和学习率
            progress_bar.set_postfix(**logs)  # 更新进度条信息

            if global_step >= config.run.max_train_steps:
                break  # 如果达到最大训练步骤，退出循环

    progress_bar.close()  # 关闭进度条
    dist.destroy_process_group()  # 销毁进程组


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 配置文件路径
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/second_stage.yaml")
    parser.add_argument("--finetune", type=bool, default=False)
    parser.add_argument("--finetune_checkpoint_path", type=str, default="./finetune_outputs/train01/stage1/checkpoints/checkpoint-1000.pt")
    parser.add_argument("--train_data_dir", type=str, default="./0finetune_datas/high_visual_quality/")
    parser.add_argument("--val_video_path", type=str, default="./0finetune_datas/c0118-1080p-10s.mp4")
    parser.add_argument("--val_audio_path", type=str, default="./0finetune_datas/kanghui_train_30s.mp3")
    parser.add_argument("--output_dir", type=str, default="./finetune_outputs/train01/stage2-notrepa/")

    args = parser.parse_args()  # 解析命令行参数
    config = OmegaConf.load(args.unet_config_path)  # 加载配置
    config.unet_config_path = args.unet_config_path  # 设置配置文件路径

    if torch.cuda.device_count() < 2 and args.unet_config_path.endwith("second_stage.yaml"):
        raise ValueError("At least 2 GPUs are required for training")

    if args.finetune:
        config.data.train_output_dir = args.output_dir  # 设置训练输出目录
        config.data.train_data_dir = args.train_data_dir  # 设置训练数据目录
        config.data.val_video_path = args.val_video_path  # 设置验证视频路径
        config.data.val_audio_path = args.val_audio_path  # 设置验证音频路径

        config.run.finetune = True
        config.optimizer.lr = 1e-5
        config.run.max_train_steps = 1000  # 设置最大训练步数
        config.ckpt.save_ckpt_steps = 500  # 设置保存检查点步数
        config.ckpt.resume_ckpt_path = args.finetune_checkpoint_path
        
    import time
    st = time.time()
    main(config)  # 运行主函数
    print("training cost: ", time.time() - st)
