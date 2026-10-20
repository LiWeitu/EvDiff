#!/usr/bin/env python
# coding=utf-8
import argparse
import copy
import datetime
import logging
import math
import os
import sys
sys.path.append(os.getcwd())
import shutil
from contextlib import nullcontext
from pathlib import Path
import torch.nn.functional as F
import torch
import itertools
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
import pyiqa
import diffusers
from diffusers import (
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
from training.data.data_h5_gray import H5_ESIM_Dataset_gray_merge

from training.models.voxel_model import OptimizedEventVAEEncoderv5
if is_wandb_available():
    import wandb
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)
valid_yushu = 49 # 原来是49


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--vae_lora_dir",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--rank_vae",
        type=int,
        default=64,
        help="和直接训练的vaelora保持一致",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--temporal_fusion",
        type=str,
        default="gated",
        help="Path to pretrained E2VID model",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default=None,
        help="Path to the public E2VID+ checkpoint from EVREAL.",
    )
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
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--use_dasm",
        action="store_true",
        help="use dasm",
    )
    parser.add_argument(
        "--use_random_bias",
        action="store_true",
        help="add ramdom to DASM step",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="voxel_vae-checkpoint",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="voxel_vae_02",
        help="log_name",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        default=2500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
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
        "--learning_rate",
        type=float,
        default=1e-5,
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

def collate_fn(batch_list, weight_dtype=torch.float32):
    """
    batch_list: 输入是一个列表，每个元素是一个序列(来自__getitem__的final_sequence)
    每个序列包含self.sequence_length个字典
    """
    sequence_length = len(batch_list[0])
    # 重组数据结构: 从 [batch_size, sequence_length] 转换为 [sequence_length, batch_size]
    transposed_batch = list(zip(*batch_list))
    # 初始化最终结果的字典
    batch = {
        "gray_img": [],
        "events": []  # 新增的事件数据
    }

    # 处理每个时间步的数据
    for time_step in range(sequence_length):
        # 当前时间步的所有样本
        time_step_samples = transposed_batch[time_step]

        # 1. 处理图像数据
        gray_img = torch.stack([sample["gray_img"] for sample in time_step_samples])
        # 4. 处理事件数据
        events = torch.stack([sample["events"] for sample in time_step_samples])

        # 保存当前时间步的结果
        batch["gray_img"].append(gray_img)
        batch["events"].append(events)

    # 将列表转换为张量，维度为 (sequence_length, batch_size, ...)
    for key in ["gray_img", "events"]:
        batch[key] = torch.stack(batch[key]).to(dtype=weight_dtype)
    return batch

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
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # logger创建好了，打印一下参数
    logger.info(args)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    voxel_vae = OptimizedEventVAEEncoderv5(input_channels=args.input_channels,
                                   latent_channels=args.output_channels,
                                   base_channels=args.base_channels,
                                   temporal_fusion=args.temporal_fusion)

    from training.utils.teacher import load_teacher
    e2vid_model = load_teacher(args.teacher_checkpoint)
    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae",
        )

    voxel_vae.requires_grad_(True)
    e2vid_model.requires_grad_(False)
    vae.requires_grad_(False)

    # add the LoRA weights to the vae encoder
    vae_target_modules = ['encoder.conv_in', 'encoder.down_blocks.0.resnets.0.conv1', 'encoder.down_blocks.0.resnets.0.conv2', 'encoder.down_blocks.0.resnets.1.conv1',
                          'encoder.down_blocks.0.resnets.1.conv2', 'encoder.down_blocks.0.downsamplers.0.conv', 'encoder.down_blocks.1.resnets.0.conv1',
                          'encoder.down_blocks.1.resnets.0.conv2', 'encoder.down_blocks.1.resnets.0.conv_shortcut', 'encoder.down_blocks.1.resnets.1.conv1', 'encoder.down_blocks.1.resnets.1.conv2',
                          'encoder.down_blocks.1.downsamplers.0.conv', 'encoder.down_blocks.2.resnets.0.conv1', 'encoder.down_blocks.2.resnets.0.conv2',
                          'encoder.down_blocks.2.resnets.0.conv_shortcut', 'encoder.down_blocks.2.resnets.1.conv1', 'encoder.down_blocks.2.resnets.1.conv2', 'encoder.down_blocks.2.downsamplers.0.conv',
                          'encoder.down_blocks.3.resnets.0.conv1', 'encoder.down_blocks.3.resnets.0.conv2', 'encoder.down_blocks.3.resnets.1.conv1', 'encoder.down_blocks.3.resnets.1.conv2',
                          'encoder.mid_block.attentions.0.to_q', 'encoder.mid_block.attentions.0.to_k', 'encoder.mid_block.attentions.0.to_v', 'encoder.mid_block.attentions.0.to_out.0',
                          'encoder.mid_block.resnets.0.conv1', 'encoder.mid_block.resnets.0.conv2', 'encoder.mid_block.resnets.1.conv1', 'encoder.mid_block.resnets.1.conv2', 'encoder.conv_out', 'quant_conv']
    vae_lora_config = LoraConfig(
        r=args.rank_vae,
        lora_alpha=args.rank_vae,
        init_lora_weights="gaussian",
        target_modules=vae_target_modules
    )
    vae.add_adapter(vae_lora_config, adapter_name="default")
    vae_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.vae_lora_dir ,weight_name="vae.safetensors")
    load_lora_state_dict(vae_lora_state_dict, vae)
    vae.enable_adapters()

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        voxel_vae.load_state_dict(checkpoint["state_dict"])


    # Set the model in the correct device and dtype.
    weight_dtype = torch.float16
    if accelerator.mixed_precision == "no":
        weight_dtype = torch.float32
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    voxel_vae.to(accelerator.device, dtype=weight_dtype)
    # Keep the frozen E2VID+ teacher in fp32.
    e2vid_model.to(accelerator.device, dtype=torch.float32)
    # VAE微调的时候用的fp16
    vae.to(accelerator.device, dtype=torch.float16)
    print("voxel_vae dtype:", next(voxel_vae.parameters()).dtype)
    print("e2vid_model dtype:", next(e2vid_model.parameters()).dtype)
    print("vae dtype:", next(vae.parameters()).dtype)

    # Save and load model hooks
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for index, model in enumerate(models):
                if isinstance(model, OptimizedEventVAEEncoderv5):
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
        e2vid_tmp = models.pop()
        vae_tmp = models.pop()
        voxel_vae = models.pop()
        # 加载 E2VID 状态
        voxel_vae_path = os.path.join(input_dir, 'voxel_vae.pth')

        if os.path.exists(voxel_vae_path):
            print(f"Loading voxel_vae model with config from {voxel_vae_path}")
            checkpoint = torch.load(voxel_vae_path, map_location=accelerator.device, weights_only=False)

            state_dict = checkpoint['state_dict']
            voxel_vae.load_state_dict(state_dict, strict=True)

            voxel_vae.train()
            for param in voxel_vae.parameters():
                param.requires_grad = True

        else:
            print(f"Warning: No E2VID checkpoint found at {voxel_vae_path}")
            print("使用初始化的E2VID模型")


        if args.mixed_precision == "fp16":
            models = [voxel_vae]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # trainable parameters
    voxel_vae_parameters = list(filter(lambda p: p.requires_grad, voxel_vae.parameters()))

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [voxel_vae]
        cast_training_params(models, dtype=torch.float32)

    # Optimization parameters
    voxel_vae_with_lr = {"params": voxel_vae_parameters, "lr": args.learning_rate}
    params_to_optimize = [voxel_vae_with_lr]

    optimizer_class = torch.optim.AdamW
    optimizer_g = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 定义文件路径
    train_idx_path = os.path.join(args.output_dir, "train_indices.pt")
    val_idx_path = os.path.join(args.output_dir, "val_indices.pt")

    full_dataset = H5_ESIM_Dataset_gray_merge(device=accelerator.device,
                                        sequence_length=args.E2VID_sequence_length)

    # 检查是否已经存在保存好的索引
    if os.path.exists(train_idx_path) and os.path.exists(val_idx_path):
        logger.info(f"Loading existing train/validation indices...")
        train_indices = torch.load(train_idx_path)
        val_indices = torch.load(val_idx_path)
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    else:
        # 如果是第一次运行，进行随机分割并保存索引
        logger.info(f"Creating and saving new train/validation indices...")
        train_size = int(0.95 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        # 提取索引并保存
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices

        torch.save(train_indices, train_idx_path)
        torch.save(val_indices, val_idx_path)

    # 训练集 DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, weight_dtype),
        num_workers=args.dataloader_num_workers,
    )

    # 验证集 DataLoader
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,  # 可以使用相同的batch_size或者单独定义
        shuffle=False,  # 验证集不需要shuffle
        collate_fn=lambda examples: collate_fn(examples, weight_dtype),
        num_workers=args.dataloader_num_workers,
    )

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

    # Prepare everything with our `accelerator`.
    voxel_vae, e2vid_model, vae = accelerator.prepare(voxel_vae, e2vid_model, vae)
    # vae, transformer, transformer_reg  = accelerator.prepare(vae, transformer, transformer_reg)

    train_dataloader, val_dataloader, optimizer_g, lr_scheduler_g = accelerator.prepare(train_dataloader, val_dataloader, optimizer_g, lr_scheduler_g)
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
        tracker_name = "vae_e2vid_old"
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
            # print("path:", path)
            # print("os.path.join(args.output_dir, path)",os.path.join(args.output_dir, path))
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

    log_dict = {}
    autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)
    e2vid_model.eval()
    vae.eval()

    for epoch in range(first_epoch, args.num_train_epochs):
        voxel_vae.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [voxel_vae]
            with accelerator.accumulate(models_to_accumulate):
                # 获取序列长度
                sequence_length = batch['events'].shape[0]

                # 为整个序列创建存储空间
                loss_g_t = 0
                # 在时间步循环外初始化累积变量
                total_loss_g = torch.tensor(0.0, device=accelerator.device)
                valid_steps = 0  # 记录有效时间步数

                # 处理序列中的每个时间步
                voxel_vae.reset_states()
                e2vid_model.reset_states()
                for t in range(args.E2VID_skip):
                    # 提取当前时间步的数据
                    events_t = batch['events'][t]
                    with torch.no_grad():
                        _ = e2vid_model(events_t)

                for t in range(args.E2VID_skip, sequence_length):
                    # 提取当前时间步的数据
                    events_t = batch['events'][t]

                    with autocast_ctx:
                        stu_z, stu_mean, stu_log_var = voxel_vae(events_t, sample=True)
                        # 获取教师模型输出
                        with torch.no_grad():
                            valid_steps += 1
                            e2vid_gray = e2vid_model(events_t)['image']
                            e2vid_gray = e2vid_gray.repeat(1, 3, 1, 1)
                            e2vid_gray = e2vid_gray * 2 - 1

                            # 获取教师模型的分布参数和采样
                            tea_dist = unwrap_model(vae).encode(e2vid_gray).latent_dist
                            tea_z = tea_dist.sample() * unwrap_model(vae).config.scaling_factor
                            tea_mean = tea_dist.mean
                            tea_log_var = tea_dist.logvar
                        # 计算损失 TODO 这里的损失并没有把序列的都加起来，而是只保留了最后一个的loss。
                        loss_mse = F.mse_loss(stu_z, tea_z.detach(), reduction='mean')
                        # 2. KL散度损失（使用解析形式）
                        # KL(N(μ_s, σ_s^2) || N(μ_t, σ_t^2)) =
                        #   log(σ_t/σ_s) + (σ_s^2 + (μ_s - μ_t)^2)/(2σ_t^2) - 1/2
                        tea_var = torch.exp(tea_log_var)
                        stu_var = torch.exp(stu_log_var)

                        # 计算KL散度项
                        kl_loss = 0.5 * (
                            tea_log_var.detach() - stu_log_var - 1
                            + (stu_var + (stu_mean - tea_mean.detach())**2) / (tea_var.detach() + 1e-8)
                        )

                        # 对空间维度求和并取批次平均
                        kl_loss = kl_loss.sum(dim=[1, 2, 3]).mean()
                        # 组合损失（调整权重）
                        kl_weight = 1e-12  # 可调整的超参数
                        loss_g_t = loss_mse + kl_weight * kl_loss

                        # 累加各损失项，不要detach()
                        total_loss_g += loss_g_t

                # backward
                total_loss_g = total_loss_g / valid_steps  # 平均损失
                accelerator.backward(total_loss_g)
                if accelerator.sync_gradients:
                    params_to_clip = voxel_vae_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_g.step()
                lr_scheduler_g.step()
                optimizer_g.zero_grad()
                # # Clear the cache to avoid memory leak
                del stu_z, stu_mean, stu_log_var, tea_dist, tea_z, tea_mean, tea_log_var
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 15:
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
                    "total_loss": total_loss_g.detach().item(),
                    "loss_mse": loss_mse.detach().item(),
                    "kl_loss": kl_loss.detach().item(),
                    }
            progress_bar.set_postfix(**logs)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            if accelerator.is_main_process and step % args.validation_steps == 9:
                # 切换到评估模式
                voxel_vae.eval()
                voxel_vae.reset_states()
                e2vid_model.reset_states()
                # 获取一个验证batch
                val_batch = next(val_iter)
                val_valid_step = 0

                with torch.no_grad():
                    with autocast_ctx:
                        sequence_length = val_batch['events'].shape[0]
                        val_total_loss_g = torch.tensor(0.0, device=accelerator.device)
                        for t in range(sequence_length):
                            # 跳过E2VID_skip
                            # 提取当前时间步的数据
                            val_events_t = batch['events'][t]
                            # 使用RNN处理事件序列（连续处理）
                            with torch.no_grad():
                                val_stu_z, val_stu_mean, val_stu_log_var = voxel_vae(val_events_t, sample=True)
                            # 跳过E2VID_skip
                            if t < args.E2VID_skip:
                                continue

                            # 获取教师模型输出
                            with torch.no_grad():
                                val_valid_step += 1
                                val_e2vid_gray = e2vid_model(val_events_t)['image']
                                val_e2vid_gray = val_e2vid_gray.repeat(1, 3, 1, 1)
                                val_e2vid_gray = val_e2vid_gray * 2 - 1

                                # 获取教师模型的分布参数和采样
                                val_tea_dist = unwrap_model(vae).encode(val_e2vid_gray).latent_dist
                                val_tea_z = val_tea_dist.sample() * unwrap_model(vae).config.scaling_factor
                                val_tea_mean = val_tea_dist.mean
                                val_tea_log_var = val_tea_dist.logvar
                            # 计算损失 TODO 这里的损失并没有把序列的都加起来，而是只保留了最后一个的loss。
                            val_loss_mse = F.mse_loss(val_stu_z, val_tea_z.detach(), reduction='mean')
                            # 2. KL散度损失（使用解析形式）
                            # KL(N(μ_s, σ_s^2) || N(μ_t, σ_t^2)) =
                            #   log(σ_t/σ_s) + (σ_s^2 + (μ_s - μ_t)^2)/(2σ_t^2) - 1/2
                            val_tea_var = torch.exp(val_tea_log_var)
                            val_stu_var = torch.exp(val_stu_log_var)

                            # 计算KL散度项
                            val_kl_loss = 0.5 * (
                                val_tea_log_var - val_stu_log_var - 1
                                + (val_stu_var + (val_stu_mean - val_tea_mean)**2) / (val_tea_var + 1e-8)
                            )

                            # 对空间维度求和并取批次平均
                            val_kl_loss = val_kl_loss.sum(dim=[1, 2, 3]).mean()
                            # 组合损失（调整权重）
                            val_loss_g_t = val_loss_mse + kl_weight * val_kl_loss
                            val_total_loss_g += val_loss_g_t


                        val_total_loss_g /= val_valid_step
                        logs = {
                            "val_total_loss": val_total_loss_g.detach().item(),
                            "val_loss_mse": val_loss_mse.detach().item(),
                            "val_kl_loss": val_kl_loss.detach().item(),
                            }
                        accelerator.log(logs, step=global_step)

                del val_batch, val_tea_z, val_tea_mean, val_tea_log_var, val_stu_z, val_stu_mean, val_stu_log_var
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 切换回训练模式
                voxel_vae.train()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if accelerator.sync_gradients:
            save_path = os.path.join(args.output_dir, f"checkpoint-latest")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

    accelerator.end_training()
