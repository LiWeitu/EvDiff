#!/usr/bin/env python
# coding=utf-8
import argparse
import copy
import datetime
import logging
import math
import os
import sys
import itertools
sys.path.append(os.getcwd())
import shutil
from contextlib import nullcontext
from pathlib import Path
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large
import torch
import torch.utils.checkpoint
import transformers
import traceback
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
import pyiqa
import diffusers
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.image_processor import  VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from training.models.autoencoder_kl import AutoencoderKL
from training.utils.util import load_lora_state_dict
from training.data.data_h5_color_EVDiff import H5_ESIM_Dataset_color_merge
from training.data.data_mem import MemMapDatasetVal
from training.models.voxel_model import OptimizedEventVAEEncoderv5

if is_wandb_available():
    import wandb
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)

class BatchEncoder():
    def __init__(self, device="cuda", sd2_path="checkpoints/sd3-medium"):
        super().__init__()
        self.weight_dtype = torch.float16

        # 初始化VAE
        self.vae = AutoencoderKL.from_pretrained(sd2_path, subfolder="vae").to(device)
        self.vae.requires_grad_(False)
        self.device = device

    def encode_batch(self, batch):
        """
        批量编码函数，支持序列数据，实际可以处理B，T，或者T，B，因为VAE时间无关
        输入:
            batch: DataLoader返回的批次数据，包含:
                "lr_img": [sequence_length, batch_size, C, H, W] 张量
                "hr_img": [sequence_length, batch_size, C, H, W] 张量
                "events": [sequence_length, batch_size, num_bins, H, W] 张量
                "path": 字符串列表

        返回:
            添加了编码结果的相同批次数据，新增:
                "latent_hr": [sequence_length, batch_size, latent_channels, H//8, W//8] 张量
        """
        with torch.no_grad():
            # 获取序列和批次信息
            hr_imgs = batch["hr_img"]  # [seq_len, batch_size, C, H, W]
            sequence_length, batch_size = hr_imgs.shape[0], hr_imgs.shape[1]

            # 初始化latent列表
            latent_hr_list = []

            # 逐时间步编码（也可以reshape后一次性编码，看显存情况）
            for t in range(sequence_length):
                # 获取当前时间步的所有图像
                hr_imgs_t = hr_imgs[t].to(self.device, dtype=self.weight_dtype)  # [batch_size, C, H, W]

                # 编码
                latent_hr_t = self.vae.encode(hr_imgs_t).latent_dist.sample()
                latent_hr_t = latent_hr_t * self.vae.config.scaling_factor

                latent_hr_list.append(latent_hr_t)

            # 堆叠成序列维度
            batch["latent_hr"] = torch.stack(latent_hr_list, dim=0)  # [seq_len, batch_size, ...]

            return batch


def warp_with_flow(x, flow):
    """
    双线性插值将 x warp 到 flow 指定位置
    x:    [B, C, H, W]
    flow: [B, 2, H, W], 表示 (dx, dy)
    """
    B, C, H, W = x.size()
    # 生成归一化网格
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
    grid = torch.stack((grid_x, grid_y), 2).float()  # [H,W,2]
    grid = grid.unsqueeze(0).repeat(B,1,1,1)         # [B,H,W,2]
    flow = flow.permute(0,2,3,1)                     # [B,H,W,2]
    coords = grid + flow
    # 归一化到 [-1,1]
    coords_x = 2.0 * coords[:,:,:,0] / max(W-1,1) - 1.0
    coords_y = 2.0 * coords[:,:,:,1] / max(H-1,1) - 1.0
    grid_norm = torch.stack([coords_x, coords_y], dim=3)
    return F.grid_sample(x, grid_norm, mode='bilinear', padding_mode='border', align_corners=True)

def log_validation(
    log_dict,
    args,
    accelerator,
    vae_scale,
):
    logger.info(
        f"Running validation... \n Generating images with prompt:"
    )

    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        hq = torch.chunk(log_dict["hq"], args.train_batch_size, dim=0)[0]
        image_stu = torch.chunk(log_dict["image_stu"], args.train_batch_size, dim=0)[0]

        image_processor = VaeImageProcessor(vae_scale_factor=2 ** (vae_scale - 1))
        image = image_processor.postprocess(hq.detach())[0]
        images.append(image)

        image = image_processor.postprocess(image_stu.detach())[0]
        images.append(image)

    labels = ["hq", "image_stu"]
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{labels[i]}: {log_dict['prompt'][0]}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--E2VID_sequence_length",
        type=int,
        default=40,
        help="E2VID sequence length",
    )
    parser.add_argument(
        "--E2VID_skip",
        type=int,
        default=30,
        help="E2VID need initialization, skip the first N steps"
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=5,
        help="E2VID need initialization, skip the first N steps"
    )
    parser.add_argument(
        "--temporal_fusion",
        type=str,
        default="gated",
        help="Path to pretrained E2VID model",
    )
    parser.add_argument(
        "--lambda_temp",
        type=float,
        default=1.0,
        help="Path to pretrained E2VID model",
    )
    parser.add_argument(
        "--output_channels",
        type=int,
        default=8,
        help="voxelvae need initialization"
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=64,
        help="voxelvae need initialization"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--teacher_lora_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        required=True,
        help="Dir containing the step-1 diffusion LoRA (transformer.safetensors).",
    )
    parser.add_argument(
        "--init_voxel_vae_path",
        type=str,
        default=None,
        required=False,
        help=(
            "Path to the step-2 voxel_vae.pth FILE used to initialize the voxel VAE. "
            "Decoupled from --lora_dir so the diffusion LoRA and the voxel VAE can live in "
            "different dirs (no need to assemble/copy both into one dir). "
            "If None, falls back to <lora_dir>/voxel_vae.pth for backward compatibility."
        ),
    )
    parser.add_argument(
        "--default_embedding_dir",
        type=str,
        default="dataset/default",
        help='path to prompt embeddings'
    )
    parser.add_argument(
        "--null_embedding_dir",
        type=str,
        default="dataset/null",
        help='path to prompt embeddings'
    )
    parser.add_argument(
        "--use_default_prompt",
        action="store_true",
        help="use default prompt",
    )
    parser.add_argument(
        "--use_dasm",
        action="store_true",
        help="use dasm",
    )
    parser.add_argument(
        "--use_teacher_lora",
        action="store_true",
        help="use fine-tune lora weights for teacher model",
    )
    parser.add_argument(
        "--use_random_bias",
        action="store_true",
        help="add ramdom to DASM step",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help=("The dimension of the SR model LoRA update matrices."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tsdsr-checkpoint",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="voxel_color",
        help="log_name",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--data_fraction", type=float, default=1.0,
        help="Fraction of REDS sequences to use (data-scaling ablation #4). 1.0 = all (no subsampling).")
    parser.add_argument("--data_subset_seed", type=int, default=0,
        help="Seed for the nested sequence subsampling (data-scaling ablation #4). "
             "Same seed across fractions => nested subsets (1% subset of 5% subset of ...).")
    parser.add_argument(
        "--log_code",
        action="store_true",
        help="log code",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=5000,
        help="Number of steps to log validation images.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_reg",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode"]
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' , `"wandb" (default)` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def collate_fn(batch_list, weight_dtype=torch.float16):
    """
    batch_list: 输入是一个列表，每个元素是一个序列(来自__getitem__的final_sequence)
    每个序列包含self.sequence_length个字典
    返回的数据结构: 每个键的值是 [sequence_length, batch_size, ...] 形状的张量
    """
    # 获取序列长度 (所有序列长度相同)
    sequence_length = len(batch_list[0])
    batch_size = len(batch_list)

    # 重组数据结构: 从 [batch_size, sequence_length] 转换为 [sequence_length, batch_size]
    # 这样每一时间步的数据可以并行处理
    transposed_batch = list(zip(*batch_list))

    # 初始化最终结果的字典
    batch = {
        "hr_img": [],
        "events": [],
        "path": []
    }

    # 处理每个时间步的数据
    for time_step in range(sequence_length):
        # 当前时间步的所有样本
        time_step_samples = transposed_batch[time_step]

        # 1. 处理图像数据
        hr_img = torch.stack([sample["hr_img"] for sample in time_step_samples])

        # 2. 处理事件数据
        events = torch.stack([sample["events"] for sample in time_step_samples])

        # 保存当前时间步的结果
        batch["hr_img"].append(hr_img)
        batch["events"].append(events)

    # 将列表转换为张量，维度为 (sequence_length, batch_size, ...)
    for key in ["hr_img", "events"]:
        batch[key] = torch.stack(batch[key]).to(dtype=weight_dtype)

    # 处理path (只需要第一个时间步的path，因为同一序列path相同)
    tmp_sample = transposed_batch[0]
    path = [sample["path"] for sample in tmp_sample]
    batch["path"] = path

    return batch

def collate_fn_val(examples, weight_dtype):
    """
    新的验证集collate函数，处理新格式的数据
    examples是list of dict，每个dict包含:
        - frames: [T, 1, H, W]
        - events: [T, num_bins, H, W]
        - timestamps: [T]
        - dts: [T]
        - sequence_name, sequence_idx, segment_start, segment_length
    """
    # 提取并stack所有数据
    frames = torch.stack([example['frames'] for example in examples])  # [B, T, 1, H, W]
    events = torch.stack([example['events'] for example in examples])  # [B, T, num_bins, H, W]
    timestamps = torch.stack([example['timestamps'] for example in examples])  # [B, T]
    dts = torch.stack([example['dts'] for example in examples])  # [B, T]

    # 可选：收集元数据
    sequence_names = [example['sequence_name'] for example in examples]
    sequence_indices = [example['sequence_idx'] for example in examples]
    segment_starts = [example['segment_start'] for example in examples]

    return {
        'hr_img': frames.to(dtype=weight_dtype),
        'events': events.to(dtype=weight_dtype),
        'timestamps': timestamps,
        'dts': dts,
        'sequence_names': sequence_names,
        'sequence_indices': sequence_indices,
        'segment_starts': segment_starts,
    }


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer"
    )
    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae"
    )# vae只用decoder
    voxel_vae = OptimizedEventVAEEncoderv5(input_channels=args.input_channels,
                                   latent_channels=args.output_channels,
                                   base_channels=args.base_channels,
                                   temporal_fusion=args.temporal_fusion)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    voxel_vae.requires_grad_(True) # 可能这个不能训练

    # Load voxel vae from its own path (decoupled from the diffusion LoRA dir): the step-2
    # voxel_vae.pth and the step-1 transformer.safetensors no longer need to be assembled
    # into a single --lora_dir. Falls back to <lora_dir>/voxel_vae.pth for old scripts.
    init_voxel_vae_path = args.init_voxel_vae_path or os.path.join(args.lora_dir, "voxel_vae.pth")
    voxel_vae.load_state_dict(torch.load(init_voxel_vae_path, map_location=accelerator.device, weights_only=False)['state_dict'], strict=True)

    # now we will add new LoRA weights to the DiT model
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_q_proj","add_k_proj","add_v_proj","proj","linear","proj_out"],
    )
    # add the LoRA weights to the SR model
    transformer.add_adapter(transformer_lora_config, adapter_name="default")
    stu_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.lora_dir ,weight_name="transformer.safetensors")
    load_lora_state_dict(stu_lora_state_dict, transformer)
    transformer.enable_adapters()

    # Set the model in the correct device and dtype.
    weight_dtype = torch.float16
    if accelerator.mixed_precision == "no":
        weight_dtype = torch.float32
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    voxel_vae.to(accelerator.device, dtype=weight_dtype)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Save and load model hooks
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            for index, model in enumerate(models):
                if isinstance(model, type(unwrap_model(transformer))) and index == 1:
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model, adapter_name="default")
                    StableDiffusion3Pipeline.save_lora_weights(
                output_dir, transformer_lora_layers=transformer_lora_layers_to_save,weight_name=f"transformer.safetensors"
            )
                elif isinstance(model, type(unwrap_model(voxel_vae))):
                    print("Saving voxel_vae model...")
                    voxel_vae_dir = os.path.join(output_dir)
                    os.makedirs(voxel_vae_dir, exist_ok=True)

                    # 保存完整模型状态（兼容不同检查点格式）
                    torch.save({
                        'state_dict': model.state_dict(),
                    }, os.path.join(voxel_vae_dir, 'voxel_vae.pth'))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        voxel_vae = models.pop()
        transformer = models.pop()
        vae = models.pop()

        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir,weight_name="transformer.safetensors")
        load_lora_state_dict(transformer_lora_state_dict, transformer)
        voxel_vae.load_state_dict(torch.load(os.path.join(input_dir, 'voxel_vae.pth'), map_location=accelerator.device, weights_only=False)['state_dict'], strict=True)

        if args.mixed_precision == "fp16":
            models = [transformer, vae, voxel_vae]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # trainable parameters
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters())) + \
                                    list(filter(lambda p: p.requires_grad, voxel_vae.parameters()))

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer, voxel_vae]
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    optimizer_class = torch.optim.AdamW
    optimizer_g = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = H5_ESIM_Dataset_color_merge(device=accelerator.device,
        sequence_length=args.E2VID_sequence_length,
        output_resolution=(464,464),
        data_fraction=args.data_fraction,
        data_subset_seed=args.data_subset_seed,)

    # Data-scaling (#4): 记录本次 run 实际用到的序列清单（论文可溯 / 复现）
    if args.output_dir:
        import json
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "data_scaling_subset.json"), "w") as _f:
            json.dump({
                "fraction": args.data_fraction,
                "subset_seed": args.data_subset_seed,
                "num_sequences": len(train_dataset.file_paths),
                "sequences": [os.path.basename(p) for p in train_dataset.file_paths],
            }, _f, indent=2)

    # 训练集 DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, weight_dtype),
        num_workers=args.dataloader_num_workers,
    )

    val_dataset = MemMapDatasetVal(
        data_path=os.environ.get("BS_ERGB_PATH", "data/bs_ergb"),
        segment_length=args.E2VID_sequence_length,
        num_samples=10,
        num_bins=5,
        voxel_method={'method': 'between_frames'},
        seed=42
    )

    # 验证集 DataLoader
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # 可以使用相同的batch_size或者单独定义
        shuffle=True,  # 验证集不需要shuffle
        collate_fn=lambda examples: collate_fn_val(examples, weight_dtype),
        num_workers=args.dataloader_num_workers,
    )
    batch_encoder = BatchEncoder(device=accelerator.device, sd2_path=args.pretrained_model_name_or_path)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_g = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_g,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if getattr(accelerator.state, "fsdp_plugin", None):
        fsdp_plugin = accelerator.state.fsdp_plugin
        fsdp_plugin.ignored_modules = [vae]

    # Prepare everything with our `accelerator`.
    vae, transformer, voxel_vae  = accelerator.prepare(vae, transformer, voxel_vae)

    val_dataloader, train_dataloader, optimizer_g, lr_scheduler_g  = accelerator.prepare(val_dataloader, train_dataloader, optimizer_g, lr_scheduler_g)
    val_iter = itertools.cycle(val_dataloader)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "color_vae"
        log_name = args.log_name
        time = datetime.datetime.now().strftime('%m-%d_%H:%M')
        accelerator.init_trackers(tracker_name, config=vars(args),
                                  init_kwargs={"wandb": {"name": f"{log_name}_lr{args.learning_rate}_{time}"}}
                                  )
        if args.log_code and is_wandb_available():
            wandb.run.log_code(".", log_name,
                           include_fn=lambda path: path.endswith(".py") or path.endswith(".sh"),
                           exclude_fn=lambda path, root: os.path.relpath(path, root).startswith(".history/"))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            if "latest-checkpoint" in dirs:
                path  = "latest-checkpoint"
            else:
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        elif path == "latest-checkpoint":
            accelerator.print(f"Resuming from checkpoint {path} and starting a new training run.")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = 0

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Define the function to get the sigmas
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = []
        for t in timesteps:
            # 找到 schedule_timesteps 中与 t 最近的值的下标
            idx = (schedule_timesteps - t).abs().argmin().item()
            step_indices.append(idx)

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # get_sigmas(torch.tensor([680.], device=accelerator.device))

    log_dict = {}
    lpips = pyiqa.create_metric('lpips', as_loss=True).cuda()
    # =============== 初始化光流模型（放在循环外） ===============
    flow_model = raft_large(pretrained=True).to(accelerator.device).eval()
    for p in flow_model.parameters():
        p.requires_grad = False
    autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)
    if args.use_default_prompt:
        # Do not delete prompt_default, although it is not used
        prompt_default = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extrememeticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
        prompt_embeds_default = torch.load(os.path.join(args.default_embedding_dir, "prompt_embeds.pt") , map_location=accelerator.device).repeat(args.train_batch_size, 1, 1)
        pooled_prompt_embeds_default = torch.load(os.path.join(args.default_embedding_dir, "pool_embeds.pt"), map_location=accelerator.device).repeat(args.train_batch_size, 1)
        print("using default prompt")
    else:
        print("using user prompt")
    vae.eval()
    for epoch in range(first_epoch, args.num_train_epochs):
        voxel_vae.train()
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [vae, transformer, voxel_vae]
            # 使用编码器处理批次数据
            with autocast_ctx:
                with torch.no_grad():
                    batch = batch_encoder.encode_batch(batch)
            with accelerator.accumulate(models_to_accumulate):
                voxel_vae.reset_states()
                # 获取序列长度
                sequence_length = batch['events'].shape[0]
                valid_steps = 0  # 记录有效时间步数

                # 预热阶段：不计算梯度，只更新RNN状态
                with autocast_ctx:
                    with torch.no_grad():
                        for t in range(args.E2VID_skip):
                            events_t = batch['events'][t]
                            _ = voxel_vae(events_t, sample=True)

                    image_prev, latent_prev = None, None  # 用于 temporal loss
                    # 处理序列中的每个时间步
                    for t in range(args.E2VID_skip, sequence_length):
                        # 提取当前时间步的数据
                        events_t = batch['events'][t]
                        model_input, stu_mean, stu_log_var = voxel_vae(events_t, sample=True)
                        valid_steps += 1  # 计数有效步数

                        # Sample the latent space
                        model_input_hr = batch["latent_hr"][t]
                        hr_values = batch["hr_img"][t]

                        timesteps = torch.tensor([680.], device=accelerator.device)

                        ##########################################################
                        # Predict the noise residual，这里不知道为什么有的时候会报错
                        model_pred = transformer(
                            hidden_states=model_input,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds_default,
                            pooled_projections=pooled_prompt_embeds_default,
                            return_dict=False,
                        )[0]
                        sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                        latent_stu = model_pred * (-sigmas) + model_input

                        image_stu = unwrap_model(vae).decode(latent_stu / unwrap_model(vae).config.scaling_factor, return_dict=False)[0].clamp(-1, 1)

                        image_stu = (image_stu * 0.5 + 0.5)
                        hr_values = (hr_values * 0.5 + 0.5)

                        # Remove mse loss when getting no regular spot but oversmoothed results
                        mse_loss = F.mse_loss(latent_stu.float(), model_input_hr.float().detach(), reduction='mean')
                        lpips_loss = lpips(image_stu, hr_values)
                        data_loss =  1 * lpips_loss.float() + 1 * mse_loss
                        data_loss = data_loss.mean()

                        # ========== 光流时序一致性损失 ==========
                        temporal_loss = torch.tensor(0., device=image_stu.device)
                        if t > args.E2VID_skip and image_prev is not None:
                            with torch.no_grad():
                                # 用输入帧（而不是上色结果）估计光流，稳定性更好
                                prev_frame_in = batch["hr_img"][t-1]  # [1,3,H,W]
                                curr_frame_in = batch["hr_img"][t]
                                flow = flow_model(prev_frame_in, curr_frame_in)[-1]  # RAFT 输出金字塔最后一层
                            warped_prev = warp_with_flow(image_prev, flow)
                            temporal_loss = F.l1_loss(image_stu, warped_prev.detach())

                        image_prev = image_stu.detach()
                        # Compute total loss
                        loss_g = data_loss + args.lambda_temp * temporal_loss



                # backward
                accelerator.backward(loss_g)
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer_g.step()
                    lr_scheduler_g.step()
                    optimizer_g.zero_grad()

                # 保存当前帧输出，供下一步用
                image_prev = image_stu.detach()
                # Clear the cache to avoid memory leak
                del latent_stu, image_stu, hr_values, loss_g, data_loss, model_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 500:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                    "lpips_loss": lpips_loss.detach().item(),
                    "temporal_loss": temporal_loss.detach().item(),
                    "mse_loss": mse_loss.detach().item(),
                    }
            progress_bar.set_postfix(**logs)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            if accelerator.is_main_process and step % args.validation_steps == 49:
                # 切换到评估模式
                transformer.eval()
                # 获取一个验证batch
                val_batch = next(val_iter)
                # 使用编码器处理批次数据
                with autocast_ctx:
                    with torch.no_grad():
                        val_batch = batch_encoder.encode_batch(val_batch)
                val_log_dict = {}

                voxel_vae.reset_states()
                # 获取序列长度
                sequence_length = val_batch['events'].shape[0]
                valid_steps = 0  # 记录有效时间步数

                with torch.no_grad():
                    for t in range(sequence_length-1):
                        events_t = val_batch['events'][:, t, :, :, :]
                        _ = voxel_vae(events_t, sample=True)

                    # 提取当前时间步的数据
                    events_t = val_batch['events'][:, -1, :, :, :]
                    val_model_input, _, _ = voxel_vae(events_t, sample=True)
                    valid_steps += 1  # 计数有效步数

                    # Sample the latent space
                    val_model_input_hr = val_batch["latent_hr"][:, -1]

                    timesteps = torch.tensor([680.], device=accelerator.device)

                    ##########################################################
                    # Predict the noise residual
                    val_model_pred = transformer(
                        hidden_states=val_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds_default,
                        pooled_projections=pooled_prompt_embeds_default,
                        return_dict=False,
                    )[0]
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                    val_latent_stu = val_model_pred * (-sigmas) + val_model_input

                    val_image_stu = unwrap_model(vae).decode(val_latent_stu / unwrap_model(vae).config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
                val_log_dict["image_stu"] = val_image_stu.float().cpu()
                val_log_dict["hq"] = val_batch["hr_img"][:, -1].float().cpu()
                val_log_dict["prompt"] = "empty"

                # 调用验证记录函数
                val_hr_values = val_batch["hr_img"][:, -1]
                val_image_stu = (val_image_stu * 0.5 + 0.5)
                val_hr_values = (val_hr_values * 0.5 + 0.5)

                # 计算 validation losses
                val_mse_loss = F.mse_loss(val_latent_stu.float().detach(), val_model_input_hr.float().detach(), reduction='mean')
                val_lpips_loss = lpips(val_image_stu, val_hr_values)

                # 整理日志
                val_logs = {
                    "val_lpips_loss": val_lpips_loss.detach().item(),
                    "val_mse_loss": val_mse_loss.detach().item(),
                }
                if accelerator.is_main_process:
                    accelerator.log(val_logs, step=global_step)
                    progress_bar.set_postfix(**val_logs)
                log_validation(val_log_dict, args, accelerator, len(unwrap_model(vae).config.block_out_channels))

                del val_latent_stu, val_hr_values, val_image_stu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 切换回训练模式
                transformer.train()
                voxel_vae.train()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if accelerator.sync_gradients:
            save_path = os.path.join(args.output_dir, f"checkpoint-latest")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

    accelerator.end_training()
