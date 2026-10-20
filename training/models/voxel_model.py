import torch
from diffusers import AutoencoderKL
from torchinfo import summary
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D, UNetMidBlock2D
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 - 减少计算量但保持表达能力"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class EfficientResBlock(nn.Module):
    """高效的残差块 - 更多参数但更少计算"""
    def __init__(self, channels, expansion_ratio=2):
        super().__init__()
        hidden_channels = channels * expansion_ratio

        # 使用1x1卷积扩展维度
        self.expand = nn.Conv2d(channels, hidden_channels, 1)

        # 深度可分离卷积处理空间信息
        self.dwconv = DepthwiseSeparableConv(
            hidden_channels, hidden_channels, 3, 1, 1
        )

        # 压缩回原始维度
        self.project = nn.Conv2d(hidden_channels, channels, 1)

        # Squeeze-and-Excitation注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        # 扩展 -> 深度卷积 -> 压缩
        x = self.expand(x)
        x = self.dwconv(x)
        x = self.project(x)

        # SE注意力
        se_weight = self.se(x)
        x = x * se_weight

        # 残差连接
        x = x + identity
        return self.norm(x)



# 辅助函数用于模型分析


class EfficientTemporalFusionv2(nn.Module):
    """
    增强时序融合模块 - 平衡简单性与有效性

    核心改进：
    1. 运动感知门控（Motion-Aware Gating）
    2. 双层记忆机制（短期+长期）
    3. 自适应遗忘率（基于局部相似度）
    4. 历史特征精炼（减少累积误差）

    相比v2的简化：
    - 移除全局可学习参数（易崩溃）
    - 移除场景切换检测（过于复杂）
    - 保持轻量级网络结构
    """

    def __init__(self, channels, fusion_type='gated'):
        super().__init__()
        self.channels = channels
        self.fusion_type = fusion_type

        # === 核心改进1: 运动检测网络 ===
        # 检测当前帧与历史的差异，输出运动强度图
        self.motion_detector = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels // 2, 3, padding=1, groups=channels // 4),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()  # [0, 1] 运动强度
        )

        # === 核心改进2: 相似度检测（用于自适应遗忘）===
        self.similarity_detector = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()  # [0, 1] 相似度
        )

        # === 核心改进3: 历史精炼网络 ===
        # 在使用历史前先过一层，减少累积误差
        self.history_refiner = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )

        # === 主融合门控（增强版）===
        if fusion_type == 'gated':
            # 输入: [当前特征, 精炼历史, 运动图, 相似度图]
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(channels * 2 + 2, channels, 1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid()
            )

        # === 记忆状态 ===
        self.short_memory = None  # 短期记忆（快速更新）
        self.long_memory = None   # 长期记忆（慢速更新）
        self.initialized = False

        # 固定超参数（不可学习，避免崩溃）
        self.short_momentum = 0.3   # 短期记忆更新速率
        self.long_momentum = 0.05   # 长期记忆更新速率
        self.min_history_weight = 0.3  # 最小历史权重（避免完全遗忘）

    def reset_states(self):
        """重置所有历史状态"""
        self.short_memory = None
        self.long_memory = None
        self.initialized = False

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 当前帧特征
        Returns:
            [B, C, H, W] 融合后的特征
        """
        # 初始化或形状变化时重置
        if not self.initialized or self.short_memory is None or self.short_memory.shape != x.shape:
            self.short_memory = x.detach()
            self.long_memory = x.detach()
            self.initialized = True
            return x

        if self.fusion_type == 'gated':
            return self._gated_fusion(x)
        else:
            raise ValueError("Not implement Temporal Model")

    def _gated_fusion(self, x):
        """门控融合（主推荐方案）"""

        # === Step 1: 检测运动和相似度 ===
        motion_input = torch.cat([x, self.short_memory], dim=1)
        motion_map = self.motion_detector(motion_input)  # [B, 1, H, W]

        similarity_input = torch.cat([x, self.short_memory], dim=1)
        similarity_map = self.similarity_detector(similarity_input)  # [B, 1, H, W]

        # === Step 2: 自适应遗忘率 ===
        # 核心思想: 相似度高 + 运动小 = 保留历史多
        #          相似度低 + 运动大 = 保留历史少
        forget_rate = similarity_map * (1.0 - motion_map * 0.7)
        # 限制遗忘率下限，避免完全丢弃历史
        forget_rate = torch.clamp(forget_rate, min=self.min_history_weight, max=0.95)

        # === Step 3: 精炼历史特征 ===
        # 混合短期和长期记忆，然后精炼
        blended_memory = 0.7 * self.short_memory + 0.3 * self.long_memory
        refined_history = self.history_refiner(blended_memory)
        # 残差连接，保留原始信息
        refined_history = refined_history * 0.5 + blended_memory * 0.5

        # === Step 4: 应用遗忘率 ===
        decayed_history = refined_history * forget_rate

        # === Step 5: 门控融合 ===
        fusion_input = torch.cat([x, decayed_history, motion_map, similarity_map], dim=1)
        gate = self.fusion_gate(fusion_input)

        # 限制gate范围，避免极端值
        gate = torch.clamp(gate, min=0.15, max=0.85)

        result = gate * x + (1 - gate) * decayed_history

        # === Step 6: 更新记忆（分离梯度）===
        with torch.no_grad():
            # 短期记忆快速更新
            self.short_memory = (1 - self.short_momentum) * self.short_memory + \
                               self.short_momentum * result.detach()

            # 长期记忆慢速更新（用于稳定性）
            self.long_memory = (1 - self.long_momentum) * self.long_memory + \
                              self.long_momentum * result.detach()

        return result


class OptimizedEventVAEEncoderv5(nn.Module):
    def __init__(self,
                 input_channels=5,
                 latent_channels=8,  # 改为潜在表示的通道数（不是分布参数）
                 base_channels=64,
                 expansion_ratios=[2, 3, 4],
                 temporal_fusion='gated',
                 use_checkpoint=True,
                 debug_mode=False):
        super().__init__()

        self.debug_mode = debug_mode
        self.use_checkpoint = use_checkpoint
        self.latent_channels = latent_channels  # 存储潜在通道数

        # 计算各层通道数
        self.channels = [
            base_channels,
            base_channels * 2,
            base_channels * 3,
            base_channels * 4
        ]

        if debug_mode:
            print(f"Channel configuration: {self.channels}")
            print(f"Temporal fusion type: {temporal_fusion}")

        # 输入适配层
        self.input_adapter = nn.Sequential(
            DepthwiseSeparableConv(input_channels, self.channels[0]),
            EfficientResBlock(self.channels[0], expansion_ratios[0])
        )

        # 多尺度特征提取
        self.feature_blocks = nn.ModuleList()
        self.temporal_fusion_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        for i in range(3):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]

            feature_block = nn.Sequential(
                EfficientResBlock(in_ch, expansion_ratios[i]),
                EfficientResBlock(in_ch, expansion_ratios[i]),
            )
            self.feature_blocks.append(feature_block)

            temporal_block = EfficientTemporalFusionv2(
                in_ch, fusion_type=temporal_fusion
            )
            self.temporal_fusion_blocks.append(temporal_block)

            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            )
            self.downsample_blocks.append(downsample)

        # 中间处理块
        final_channels = self.channels[-1]
        self.mid_block = nn.Sequential(
            EfficientResBlock(final_channels, expansion_ratios[-1]),
            EfficientTemporalFusionv2(final_channels, fusion_type=temporal_fusion),
            EfficientResBlock(final_channels, expansion_ratios[-1]),
            EfficientResBlock(final_channels, expansion_ratios[-1]),
        )

        # 关键修改1：输出分布参数而不是直接输出潜在表示
        # 输出通道数应为潜在通道数的2倍（μ和log_var）
        self.distribution_adapter = nn.Sequential(
            nn.GroupNorm(32, final_channels),
            nn.SiLU(),
            nn.Conv2d(final_channels, 2 * latent_channels, kernel_size=3, padding=1)
        )

        # 参数统计
        if debug_mode:
            self._print_model_info()

    def _print_model_info(self):
        """打印模型信息用于调试"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def reset_states(self):
        """重置所有时序状态"""
        for temporal_block in self.temporal_fusion_blocks:
            temporal_block.reset_states()
        if hasattr(self.mid_block[2], 'reset_states'):
            self.mid_block[2].reset_states()

    def _checkpoint_function(self, func, *args):
        """条件性使用梯度检查点"""
        """条件性使用梯度检查点 - 跳过时序融合模块"""
        # 检查函数是否包含时序融合
        if any(isinstance(m, EfficientTemporalFusionv2) for m in func.modules()):
            # 时序模块不使用检查点
            return func(*args)

        # 其他模块使用检查点
        if self.use_checkpoint and self.training:
            return checkpoint(func, *args, use_reentrant=False)
        else:
            return func(*args)

    def forward(self, x, sample=True):
        """
        处理单个时间步的事件数据
        Args:
            x: [B, 5, H, W] 事件voxel数据
            sample: 是否进行采样（训练时为True，某些推理场景可为False）
        Returns:
            如果sample=True: 采样后的潜在表示 [B, latent_channels, H/8, W/8]
            总是返回: (mean, log_var) 或 (z, mean, log_var)
        """
        if self.debug_mode:
            print(f"Input shape: {x.shape}")

        # 输入适配
        x = self.input_adapter(x)
        if self.debug_mode:
            print(f"After input adapter: {x.shape}")

        # 多尺度特征提取和时序融合
        for i, (feat_block, temp_block, down_block) in enumerate(
            zip(self.feature_blocks, self.temporal_fusion_blocks, self.downsample_blocks)
        ):
            x = self._checkpoint_function(feat_block, x)
            x = temp_block(x)
            x = down_block(x)

            if self.debug_mode:
                print(f"After block {i}: {x.shape}")

        # 中间处理
        x = self._checkpoint_function(self.mid_block, x)
        if self.debug_mode:
            print(f"After mid block: {x.shape}")

        # 得到形状为 [B, 2*latent_channels, H/8, W/8] 的输出
        dist_params = self.distribution_adapter(x)
        mean, log_var = torch.chunk(dist_params, 2, dim=1)

        # 重参数化采样
        if sample:
            # 重参数化技巧（可微分）
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mean + eps * std
            return z, mean, log_var
        else:
            # 不采样时直接返回均值
            return mean, mean, log_var




















# 使用示例
###############################################################################

    # # 设置设备
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"使用设备: {device}\n")

    # # 创建模型
    # print("创建模型...")
    # model = OptimizedEventVAEEncoderv7(
    #     in_channels=3,
    #     base_channels=64,
    #     num_blocks=4
    # )
    # model = model.to(device)
    # print("✓ 模型创建成功\n")

    # # 1. 测试前向传播
    # test_forward_pass(model, device)

    # # 2. 计算模型复杂度
    # calculate_model_complexity(model, input_size=(5, 512, 512))


    ##############################################################################
    # # 初始化VAE模型
    # vae = AutoencoderKL.from_pretrained(
    #     subfolder="vae"
    # )
    # encoder = vae.encoder.eval()  # 获取encoder并设为评估模式

    # # 创建512x512的模拟输入 (batch_size=1, channels=3)
    # dummy_input = torch.randn(1, 3, 512, 512)

    # # 打印结构信息
    # print("="*50)
    # print("VAE Encoder Structure:")
    # print(encoder)

    # # 使用torchinfo打印详细结构和计算量
    # print("="*50)
    # print("Detailed Architecture & FLOPs:")
    # summary(
    #     encoder,
    #     input_data=dummy_input,
    #     col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    #     depth=6,  # 显示到残差块级别
    #     verbose=2,  # 显示完整结构
    #     device="cuda"  # 在CPU上运行分析
    # )
