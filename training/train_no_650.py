#!/usr/bin/env python
# coding=utf-8
import argparse
import copy
import datetime
import logging
import json
import math
import os
import sys
import itertools
import numpy as np
sys.path.append(os.getcwd())
import shutil
from contextlib import nullcontext
from pathlib import Path
import torch.nn.functional as F
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
from training.data.data import Real_ESRGAN_Dataset_strict

if is_wandb_available():
    import wandb
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)

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
        lq = torch.chunk(log_dict["lq"], args.train_batch_size, dim=0)[0]
        image_stu = torch.chunk(log_dict["image_stu"], args.train_batch_size, dim=0)[0]

        image_processor = VaeImageProcessor(vae_scale_factor=2 ** (vae_scale - 1))
        image = image_processor.postprocess(hq.detach())[0]
        images.append(image)

        image = image_processor.postprocess(lq.detach())[0]
        images.append(image)

        image = image_processor.postprocess(image_stu.detach())[0]
        images.append(image)

    labels = ["hq", "lq", "image_stu"]
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

class BatchEncoder():
    def __init__(self, device="cuda", sd3_path="checkpoints/sd3-medium"):
        super().__init__()
        self.weight_dtype = torch.float16

        # 初始化VAE
        self.vae = AutoencoderKL.from_pretrained(sd3_path, subfolder="vae").to(device)
        self.device = device

        # 初始化文本编码器
        self.tokenizers, self.text_encoders = self._load_text_encoders(sd3_path)
        self._cache_empty_prompts()

    def _load_text_encoders(self, sd3_path):
        """加载text encoders"""
        from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModelWithProjection, T5TokenizerFast

        tokenizer_one = CLIPTokenizer.from_pretrained(
            sd3_path,
            subfolder="tokenizer",
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            sd3_path,
            subfolder="tokenizer_2",
        )
        tokenizer_three = T5TokenizerFast.from_pretrained(
            sd3_path,
            subfolder="tokenizer_3",
        )

        text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
            sd3_path, subfolder="text_encoder"
        ).to(self.device, dtype=self.weight_dtype)
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            sd3_path, subfolder="text_encoder_2"
        ).to(self.device, dtype=self.weight_dtype)
        text_encoder_three = T5EncoderModel.from_pretrained(
            sd3_path, subfolder="text_encoder_3"
        ).to(self.device, dtype=self.weight_dtype)

        # 设置为eval模式
        text_encoder_one.eval()
        text_encoder_two.eval()
        text_encoder_three.eval()

        tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
        text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
        return tokenizers, text_encoders

    def _cache_empty_prompts(self):
        """缓存空prompt的embeddings"""
        with torch.no_grad():
            self.empty_variants_cache = {}
            empty_variants = ["", "empty prompt", " ", "none", "null"]

            for variant in empty_variants:
                embeds, pooled = self._encode_prompt([variant])
                self.empty_variants_cache[variant] = (embeds[0], pooled[0])

    def _encode_prompt_with_clip(self, text_encoder, tokenizer, prompt, num_images_per_prompt=1):
        """使用CLIP编码prompt"""
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(self.device), output_hidden_states=True)

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, pooled_prompt_embeds

    def _encode_prompt_with_t5(self, text_encoder, tokenizer, prompt, num_images_per_prompt=1):
        """使用T5编码prompt"""
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(self.device))[0]

        dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _encode_prompt(self, prompt_list, num_images_per_prompt=1):
        """完整的prompt编码流程"""
        # 处理空提示
        empty_embeds = []
        valid_prompts = []
        indices = []

        for i, prompt in enumerate(prompt_list):
            if prompt in self.empty_variants_cache:
                empty_embeds.append(self.empty_variants_cache[prompt])
            else:
                valid_prompts.append(prompt)
                indices.append(i)

        # 编码非空提示
        if valid_prompts:
            clip_tokenizers = self.tokenizers[:2]
            clip_text_encoders = self.text_encoders[:2]

            clip_prompt_embeds_list = []
            clip_pooled_prompt_embeds_list = []

            for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
                prompt_embeds, pooled_prompt_embeds = self._encode_prompt_with_clip(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    prompt=valid_prompts,
                    num_images_per_prompt=num_images_per_prompt,
                )
                clip_prompt_embeds_list.append(prompt_embeds)
                clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

            clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

            t5_prompt_embed = self._encode_prompt_with_t5(
                self.text_encoders[-1],
                self.tokenizers[-1],
                prompt=valid_prompts,
                num_images_per_prompt=num_images_per_prompt,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )
            valid_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            valid_pooled = pooled_prompt_embeds

        # 组合结果
        all_embeds = []
        all_pooled = []
        idx = 0
        valid_idx = 0

        for i in range(len(prompt_list)):
            if i in indices:
                # 非空提示
                all_embeds.append(valid_embeds[valid_idx])
                all_pooled.append(valid_pooled[valid_idx])
                valid_idx += 1
            else:
                # 空提示
                e, p = empty_embeds[idx]
                all_embeds.append(e)
                all_pooled.append(p)
                idx += 1

        return torch.stack(all_embeds), torch.stack(all_pooled)

    def encode_batch(self, batch):
        """
        批量编码函数
        输入:
            batch: DataLoader返回的批次数据，包含:
                "lr_img": [B, C, H, W] 张量
                "hr_img": [B, C, H, W] 张量
                "prompt_text": 字符串列表
                "hr_path": 字符串列表（可选）

        返回:
            添加了编码结果的相同批次数据
        """
        with torch.no_grad():
            # 编码HR图像
            hr_imgs = batch["hr_img"].to(self.device, dtype=self.weight_dtype)
            latent_hr = self.vae.encode(hr_imgs).latent_dist.sample()
            latent_hr = latent_hr * self.vae.config.scaling_factor

            # 编码文本提示
            prompt_embeds, pooled_embeds = self._encode_prompt(batch["prompts"])

            # 添加到批次数据
            batch["latent_hr"] = latent_hr
            batch["prompt_embeds"] = prompt_embeds
            batch["pooled_prompt_embeds"] = pooled_embeds

            return batch

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="Path to checkpoint",
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
        "--use_random_bias",
        action="store_true",
        help="add ramdom to DASM step",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help=("The dimension of the SR model LoRA update matrices."),
    )
    parser.add_argument(
        "--rank_lora",
        type=int,
        default=16,
        help=("The dimension of the auxiliary model LoRA update matrices."),
    )
    parser.add_argument(
        "--rank_vae",
        type=int,
        default=64,
        help=("The dimension of the vae LoRA update matrices."),
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
        default="tsdsr",
        help="log_name",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--data_fraction", type=float, default=1.0,
        help="Stage-1 data-scaling sweep: fraction of the Places365 TRAIN set to use (1.0 = full set). "
             "Subsampling is nested + fixed-seed (see --data_subset_seed); the val split is left intact.",
    )
    parser.add_argument(
        "--data_subset_seed", type=int, default=0,
        help="Seed for the nested train-index subsample. Independent of --seed (which governs the 95/5 split), "
             "so the split is identical across fractions and subsamples are nested (0.1% subset of ... subset of 100%).",
    )
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
        default="fp16",
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

def collate_fn(examples, weight_dtype=torch.float16):
    lr_img = [example["lr_img"] for example in examples]
    hr_img = [example["hr_img"] for example in examples]
    hr_path = [example["hr_path"] for example in examples]

    prompts = [example["prompt_text"] for example in examples]

    lr_img = torch.stack(lr_img)
    hr_img = torch.stack(hr_img)

    batch = {
        "lr_img": lr_img.to(dtype=weight_dtype),
        "hr_img": hr_img.to(dtype=weight_dtype),
        "prompts": prompts,
        "hr_path": hr_path}
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
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    # now we will add new LoRA weights to the DiT model
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_q_proj","add_k_proj","add_v_proj","proj","linear","proj_out"],
    )
    # add the LoRA weights to the SR model
    transformer.add_adapter(transformer_lora_config, adapter_name="default")
    if args.checkpoint_path is not None:
        stu_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.checkpoint_path ,weight_name="transformer.safetensors")
        load_lora_state_dict(stu_lora_state_dict, transformer)
    transformer.enable_adapters()

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
    if args.checkpoint_path is not None:
        vae_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(args.checkpoint_path ,weight_name="vae.safetensors")
        load_lora_state_dict(vae_lora_state_dict, vae)
    vae.enable_adapters()

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
                elif isinstance(model, type(unwrap_model(vae))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model, adapter_name="default")
                    StableDiffusion3Pipeline.save_lora_weights(
                output_dir, transformer_lora_layers=transformer_lora_layers_to_save,weight_name=f"vae.safetensors"
            )

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        transformer = models.pop()
        vae = models.pop()

        vae_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir,weight_name="vae.safetensors")
        transformer_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir,weight_name="transformer.safetensors")

        load_lora_state_dict(vae_lora_state_dict, vae)
        load_lora_state_dict(transformer_lora_state_dict, transformer)

        if args.mixed_precision == "fp16":
            models = [transformer, vae]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # trainable parameters
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters())) + list(filter(lambda p: p.requires_grad, vae.parameters()))

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer, vae]
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

    train_idx_path = os.path.join(args.output_dir, "train_indices.pt")
    val_idx_path = os.path.join(args.output_dir, "val_indices.pt")

    full_dataset = Real_ESRGAN_Dataset_strict(device=accelerator.device)

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

    # ===== Stage-1 data-scaling: nested, fixed-seed subsample of TRAIN indices =====
    # The val split above is governed by --seed (fixed at 43 across all runs) and is left
    # INTACT, so validation curves stay comparable across fractions. Only the train indices
    # are subsampled, using a SEPARATE rng (--data_subset_seed) so that subsamples are nested:
    # one permutation per seed, perm[:N_f] with N_f proportional to fraction, re-sorted by
    # position  =>  0.1% ⊂ 0.3% ⊂ 1% ⊂ 3% ⊂ 10% ⊂ 30% ⊂ 100%.
    if args.data_fraction < 1.0:
        n_total_train = len(train_indices)
        n_keep = max(1, int(round(n_total_train * args.data_fraction)))
        perm = np.random.default_rng(args.data_subset_seed).permutation(n_total_train)
        keep_positions = np.sort(perm[:n_keep])
        train_indices = [train_indices[int(p)] for p in keep_positions]
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "data_scaling_subset.json"), "w") as f:
            json.dump({
                "data_fraction": args.data_fraction,
                "data_subset_seed": args.data_subset_seed,
                "n_keep_train": n_keep,
                "n_total_train": n_total_train,
                "n_total_corpus": len(full_dataset),
                "kept_global_indices": list(train_indices),
            }, f)
        logger.info(
            f"[data-scaling] kept {n_keep}/{n_total_train} train images "
            f"(fraction={args.data_fraction}, subset_seed={args.data_subset_seed}); "
            f"full corpus={len(full_dataset)}"
        )

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

    batch_encoder = BatchEncoder(device=accelerator.device, sd3_path=args.pretrained_model_name_or_path)

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
    vae, transformer  = accelerator.prepare(vae, transformer)

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
        tracker_name = "tsdsr"
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
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            # Only consider numbered checkpoints; skip the end-of-training
            # "checkpoint-latest" folder, whose name has no step number and would
            # crash int() below. "checkpoint-latest" is only written after the
            # training loop finishes, so the highest numbered checkpoint holds an
            # equivalent (or earlier-but-valid) state and gives a recoverable step.
            dirs = [
                d for d in dirs
                if d.startswith("checkpoint-") and d.split("-")[1].isdigit()
            ]
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
    autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)
    if args.use_default_prompt:
        # Do not delete prompt_default, although it is not used
        prompt_default = "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extrememeticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
        prompt_embeds_default = torch.load(os.path.join(args.default_embedding_dir, "prompt_embeds.pt") , map_location=accelerator.device).repeat(args.train_batch_size, 1, 1)
        pooled_prompt_embeds_default = torch.load(os.path.join(args.default_embedding_dir, "pool_embeds.pt"), map_location=accelerator.device).repeat(args.train_batch_size, 1)

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        vae.train()
        for step, batch in enumerate(train_dataloader):
            # 使用编码器处理批次数据
            with autocast_ctx:
                with torch.no_grad():
                    batch = batch_encoder.encode_batch(batch)
            models_to_accumulate = [vae, transformer]
            with accelerator.accumulate(models_to_accumulate):
                lr_values = batch["lr_img"]
                hr_values = batch["hr_img"]
                prompts = batch["prompts"]
                prompt_embeds = batch["prompt_embeds"]
                pooled_prompt_embeds = batch["pooled_prompt_embeds"]
                log_dict["lq"] = lr_values.float().cpu()
                log_dict["hq"] = hr_values.float().cpu()
                log_dict["prompt"] = prompts
                with autocast_ctx:
                    # Sample the latent space
                    model_input = unwrap_model(vae).encode(lr_values).latent_dist.sample() * unwrap_model(vae).config.scaling_factor
                    model_input_hr = batch["latent_hr"]
                    if not args.use_default_prompt:
                        prompt_embeds_default = prompt_embeds
                        pooled_prompt_embeds_default = pooled_prompt_embeds

                    timesteps = torch.tensor([680.], device=accelerator.device)

                    ##########################################################
                    # Predict the noise residual，这里不知道为什么有的时候会报错
                    try:
                        model_pred = transformer(
                            hidden_states=model_input,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds_default,
                            pooled_projections=pooled_prompt_embeds_default,
                            return_dict=False,
                        )[0]
                    except:
                        continue
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                    latent_stu = model_pred * (-sigmas) + model_input

                    image_stu = unwrap_model(vae).decode(latent_stu / unwrap_model(vae).config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
                    if accelerator.is_main_process and step % args.validation_steps == 0 and step > 0:
                        log_dict["image_stu"] = image_stu.cpu()

                    image_stu = (image_stu * 0.5 + 0.5)
                    hr_values = (hr_values * 0.5 + 0.5)

                    # Remove mse loss when getting no regular spot but oversmoothed results
                    mse_loss = F.mse_loss(latent_stu.float(), model_input_hr.float().detach(), reduction='mean')
                    lpips_loss = lpips(image_stu, hr_values)
                    data_loss =  1 * lpips_loss.float() + 1 * mse_loss
                    data_loss = data_loss.mean()

                    # Compute total loss
                    loss_g = data_loss


                # backward
                accelerator.backward(loss_g)
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_g.step()
                lr_scheduler_g.step()
                optimizer_g.zero_grad()
                # Clear the cache to avoid memory leak
                del image_stu, lr_values, hr_values, prompt_embeds, pooled_prompt_embeds, loss_g, data_loss, model_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 and global_step > 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints
                                if d.startswith("checkpoint-") and d.split("-")[1].isdigit()
                            ]
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
                    "mse_loss": mse_loss.detach().item(),
                    }
            progress_bar.set_postfix(**logs)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
            if accelerator.is_main_process and step % args.validation_steps == 0 and step > 0:
                # log_validation(log_dict, args,accelerator, len(unwrap_model(vae).config.block_out_channels))
                # 切换到评估模式
                transformer.eval()
                vae.eval()

                # 获取一个验证batch
                val_batch = next(val_iter)
                val_log_dict = {}
                # 使用编码器处理批次数据
                with autocast_ctx:
                    with torch.no_grad():
                        val_batch = batch_encoder.encode_batch(val_batch)

                with torch.no_grad():
                    val_lr_values = val_batch["lr_img"]
                    val_hr_values = val_batch["hr_img"]
                    val_prompts = val_batch["prompts"]
                    val_prompt_embeds = val_batch["prompt_embeds"]
                    val_pooled_prompt_embeds = val_batch["pooled_prompt_embeds"]

                    val_log_dict["lq"] = val_lr_values.float().cpu()
                    val_log_dict["hq"] = val_hr_values.float().cpu()
                    val_log_dict["prompt"] = val_prompts

                    with autocast_ctx:
                        # Sample the latent space
                        model_input = unwrap_model(vae).encode(val_lr_values).latent_dist.sample() * unwrap_model(vae).config.scaling_factor
                        model_input_hr = val_batch["latent_hr"]
                        if not args.use_default_prompt:
                            prompt_embeds_default = val_prompt_embeds
                            pooled_prompt_embeds_default = val_pooled_prompt_embeds

                        timesteps = torch.tensor([680.], device=accelerator.device)

                        ##########################################################
                        # Predict the noise residual
                        model_pred = transformer(
                            hidden_states=model_input,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds_default,
                            pooled_projections=pooled_prompt_embeds_default,
                            return_dict=False,
                        )[0]
                        sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                        val_latent_stu = model_pred * (-sigmas) + model_input

                        val_image_stu = unwrap_model(vae).decode(val_latent_stu / unwrap_model(vae).config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
                        val_log_dict["image_stu"] = val_image_stu.cpu()

                # 调用验证记录函数
                val_image_stu = (val_image_stu * 0.5 + 0.5)
                val_hr_values = (val_hr_values * 0.5 + 0.5)

                # 计算 validation losses
                val_mse_loss = F.mse_loss(val_latent_stu.float(), model_input_hr.float().detach(), reduction='mean')
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

                del val_latent_stu, val_lr_values, val_hr_values, val_prompt_embeds, val_pooled_prompt_embeds, val_image_stu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 切换回训练模式
                transformer.train()
                vae.train()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if accelerator.sync_gradients:
            save_path = os.path.join(args.output_dir, f"checkpoint-latest")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

    accelerator.end_training()
