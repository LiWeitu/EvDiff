import torch
from diffusers import AutoencoderKL
from torchinfo import summary
import torch.nn as nn
from torchvision.ops import Conv2dNormActivation
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D, UNetMidBlock2D
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

class EfficientTemporalFusion(nn.Module):
    """高效的时序信息融合模块 - 替代LSTM"""
    def __init__(self, channels, fusion_type='gated'):
        super().__init__()
        self.channels = channels
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # 轻量级注意力机制
            self.query = nn.Conv2d(channels, channels // 4, 1)
            self.key = nn.Conv2d(channels, channels // 4, 1)
            self.value = nn.Conv2d(channels, channels, 1)
            self.out_proj = nn.Conv2d(channels, channels, 1)

            # 新增注意力分数计算层
            self.attn_conv = nn.Sequential(
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid()
            )
        elif fusion_type == 'gated':
            # 门控融合
            self.gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels // 2, 1),
                nn.SiLU(),
                nn.Conv2d(channels // 2, channels, 1),
                nn.Sigmoid()
            )
        
        # 历史状态存储
        self.history = None
        self.initialized = False


    def reset_states(self):
        """重置历史状态"""
        self.history = None
        self.initialized = False
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 当前帧特征
        Returns:
            [B, C, H, W] 融合历史信息后的特征
        """
        batch_size = x.size(0)
        
        # 动态调整history buffer大小
        if not self.initialized or self.history.shape != x.shape:
            self.history = torch.zeros_like(x)
            self.initialized = True
            
        if self.fusion_type == 'gated':
            # 门控融合 TODO: 增加一个对history的处理，因为需要遗忘，voxel开闭操作
            combined = torch.cat([x, self.history], dim=1)
            gate = self.gate(combined)
            result = gate * x + (1 - gate) * self.history
            self.history = result.detach()  # 更新历史但不传梯度
            # self.history = result
            return result
            
        elif self.fusion_type == 'attention':
            # 修复后的注意力机制
            q = self.query(x)
            k = self.key(self.history)
            v = self.value(self.history)
            
            # 计算空间注意力图 (单通道)
            attn = self.attn_conv(q * k)  # [B, 1, H, W]
            
            # 应用注意力权重到历史特征
            weighted_v = attn * v
            
            # 融合当前特征和加权历史特征
            result = x + self.out_proj(weighted_v)
            self.history = result.detach()
            # self.history = result
            return result
            
        return x

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

class OptimizedEventVAEEncoderv2(nn.Module):
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
            
            temporal_block = EfficientTemporalFusion(
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
            EfficientResBlock(final_channels, expansion_ratios[-1]),
            EfficientTemporalFusion(final_channels, fusion_type=temporal_fusion),
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
    #     """条件性使用梯度检查点"""
    #     if self.use_checkpoint and self.training:
    #         return checkpoint(func, *args, use_reentrant=False)
    #     else:
    #         return func(*args)
        """条件性使用梯度检查点 - 跳过时序融合模块"""
        # 检查函数是否包含时序融合
        if any(isinstance(m, EfficientTemporalFusion) for m in func.modules()):
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
            return mean, log_var

def analyze_encoder(encoder, input_shape=(1, 5, 512, 512)):
    from torchinfo import summary
    
    print("="*50)
    print("EventVAE Encoder Structure:")
    # print(encoder)
    
    print("="*50)
    print("Detailed Architecture & FLOPs:")
    summary(
        encoder,
        input_data=torch.randn(input_shape),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=1,
        verbose=None,
        device="cuda"
    )

# 辅助函数用于模型分析
def analyze_model_efficiency(model, input_shape=(1, 5, 512, 512)):
    """分析模型效率"""
    model.eval()
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")
    
    # 显存和速度测试
    device = next(model.parameters()).device
    x = torch.randn(*input_shape).to(device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # 速度测试
    import time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(100):
        with torch.no_grad():
            _ = model(x)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    avg_time = (time.time() - start_time) / 100
    
    print(f"平均推理时间: {avg_time*1000:.2f}ms")
    
    # 显存占用（如果使用CUDA）
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"峰值显存占用: {peak_memory:.2f}MB")


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

class OptimizedEventVAEEncoderv6(nn.Module):
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
        
        # 输入适配层
        self.input_adapter = nn.Sequential(
            DepthwiseSeparableConv(input_channels, self.channels[0]),
            EfficientResBlock(self.channels[0], expansion_ratios[0])
        )
        
        # 多尺度特征提取
        self.feature_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        for i in range(3):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            
            feature_block = nn.Sequential(
                EfficientResBlock(in_ch, expansion_ratios[i]),
                EfficientResBlock(in_ch, expansion_ratios[i]),
                EfficientResBlock(in_ch, expansion_ratios[i]),
            )
            self.feature_blocks.append(feature_block)
            
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
            EfficientResBlock(final_channels, expansion_ratios[-1]),
            EfficientResBlock(final_channels, expansion_ratios[-1]),
            EfficientResBlock(final_channels, expansion_ratios[-1]),
        )
        
        # 输出分布参数
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
        # 删除与时序相关的重置方法，因为我们已移除时序结构
        pass
    
    def _checkpoint_function(self, func, *args):
        """条件性使用梯度检查点"""
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
        
        # 多尺度特征提取
        for i, feat_block in enumerate(self.feature_blocks):
            x = self._checkpoint_function(feat_block, x)
            x = self.downsample_blocks[i](x)
            
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

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # 单个卷积层处理所有门控
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # i, f, o, g
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for name, param in self.conv.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # 设置遗忘门的偏置为1，帮助梯度流动
                param.data.fill_(0)
                # 遗忘门偏置在第二个四分之一
                param.data[self.hidden_dim:2*self.hidden_dim] = 1.0

    def forward(self, x, h_prev=None, c_prev=None):
        batch_size, _, height, width = x.size()
        
        # 初始化隐藏状态
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, height, width, 
                               device=x.device, dtype=x.dtype)
        if c_prev is None:
            c_prev = torch.zeros(batch_size, self.hidden_dim, height, width,
                               device=x.device, dtype=x.dtype)
        
        # 合并输入和先前隐藏状态
        combined = torch.cat([x, h_prev], dim=1)
        
        # 通过门控卷积
        combined_conv = self.conv(combined)
        
        # 分割为输入门、遗忘门、输出门和候选细胞状态
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # 门控激活
        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = torch.tanh(cc_g)     # 候选值
        
        # 更新细胞状态和隐藏状态
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def reset_states(self):
        """重置状态（实际在前向传播中处理）"""
        pass

class MultiScaleConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_layers=2, kernel_size=3):
        super(MultiScaleConvLSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims] * num_layers
        
        # 创建多层ConvLSTM
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else self.hidden_dims[i-1]
            self.layers.append(ConvLSTMCell(layer_input_dim, self.hidden_dims[i], kernel_size))
        
        # 状态存储
        self.hidden_states = [None] * num_layers
        self.cell_states = [None] * num_layers

    def forward(self, x):
        """
        前向传播 - 处理单个时间步
        Args:
            x: 输入张量 [B, C, H, W]
        Returns:
            输出张量 [B, hidden_dims[-1], H, W]
        """
        current_input = x
        
        for layer_idx in range(self.num_layers):
            h_prev, c_prev = self.hidden_states[layer_idx], self.cell_states[layer_idx]
            h_next, c_next = self.layers[layer_idx](current_input, h_prev, c_prev)
            
            # 更新状态
            self.hidden_states[layer_idx] = h_next
            self.cell_states[layer_idx] = c_next
            
            current_input = h_next
        
        return current_input

    def reset_states(self):
        """重置所有层的状态"""
        self.hidden_states = [None] * self.num_layers
        self.cell_states = [None] * self.num_layers

class OptimizedEventVAEEncoderv7(nn.Module):
    def __init__(self, 
                 input_channels=5, 
                 latent_channels=8,
                 base_channels=64,
                 expansion_ratios=[2, 3, 4],
                 temporal_fusion="gated",
                 convlstm_layers=1,
                 use_checkpoint=True,
                 debug_mode=False):
        super().__init__()
        
        self.debug_mode = debug_mode
        self.use_checkpoint = use_checkpoint
        self.latent_channels = latent_channels
        
        # 计算各层通道数
        self.channels = [
            base_channels,
            base_channels * 2,
            base_channels * 3,
            base_channels * 4
        ]
        
        if debug_mode:
            print(f"Channel configuration: {self.channels}")
            print(f"ConvLSTM layers: {convlstm_layers}")
        
        # 输入适配层
        self.input_adapter = nn.Sequential(
            DepthwiseSeparableConv(input_channels, self.channels[0]),
            EfficientResBlock(self.channels[0], expansion_ratios[0])
        )
        
        # 多尺度特征提取（使用ConvLSTM替换时序融合）
        self.feature_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()  # 现在使用ConvLSTM
        self.downsample_blocks = nn.ModuleList()
        
        for i in range(3):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            
            # 特征提取块
            feature_block = nn.Sequential(
                EfficientResBlock(in_ch, expansion_ratios[i]),
                EfficientResBlock(in_ch, expansion_ratios[i]),
            )
            self.feature_blocks.append(feature_block)
            
            # 替换：使用ConvLSTM代替原来的时序融合
            temporal_block = MultiScaleConvLSTM(
                input_dim=in_ch,
                hidden_dims=in_ch,  # 保持相同维度
                num_layers=convlstm_layers,
                kernel_size=3
            )
            self.temporal_blocks.append(temporal_block)
            
            # 下采样
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            )
            self.downsample_blocks.append(downsample)
        
        # 中间处理块（也使用ConvLSTM）
        final_channels = self.channels[-1]
        
        # 重新设计中间块以包含ConvLSTM
        self.mid_res_blocks1 = nn.Sequential(
            EfficientResBlock(final_channels, expansion_ratios[-1]),
            EfficientResBlock(final_channels, expansion_ratios[-1]),
        )
        
        self.mid_temporal = MultiScaleConvLSTM(
            input_dim=final_channels,
            hidden_dims=final_channels,
            num_layers=convlstm_layers,
            kernel_size=3
        )
        
        self.mid_res_blocks2 = nn.Sequential(
            EfficientResBlock(final_channels, expansion_ratios[-1]),
            EfficientResBlock(final_channels, expansion_ratios[-1]),
        )
        
        # 输出分布参数
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
        """重置所有ConvLSTM的状态"""
        for temporal_block in self.temporal_blocks:
            temporal_block.reset_states()
        self.mid_temporal.reset_states()
    
    def _checkpoint_function(self, func, *args):
        """条件性使用梯度检查点 - 跳过时序模块"""
        # ConvLSTM模块不使用检查点（因为有内部状态）
        if any(isinstance(m, (MultiScaleConvLSTM, ConvLSTMCell)) for m in func.modules()):
            return func(*args)
        
        # 其他模块使用检查点
        if self.use_checkpoint and self.training:
            return checkpoint(func, *args, use_reentrant=False)
        else:
            return func(*args)
    
    def forward(self, x, sample=True):
        """
        处理单个时间步的事件数据（使用ConvLSTM进行时序建模）
        Args:
            x: [B, 5, H, W] 事件voxel数据
            sample: 是否进行采样
        Returns:
            如果sample=True: (z, mean, log_var)
            如果sample=False: (mean, mean, log_var)
        """
        if self.debug_mode:
            print(f"Input shape: {x.shape}")
            
        # 输入适配
        x = self.input_adapter(x)
        if self.debug_mode:
            print(f"After input adapter: {x.shape}")
        
        # 多尺度特征提取和ConvLSTM时序建模
        for i, (feat_block, temp_block, down_block) in enumerate(
            zip(self.feature_blocks, self.temporal_blocks, self.downsample_blocks)
        ):
            # 特征提取
            x = self._checkpoint_function(feat_block, x)
            
            # ConvLSTM时序处理（不使用检查点）
            x = temp_block(x)
            
            # 下采样
            x = down_block(x)
            
            if self.debug_mode:
                print(f"After block {i}: {x.shape}")
        
        # 中间处理
        x = self._checkpoint_function(self.mid_res_blocks1, x)
        x = self.mid_temporal(x)  # ConvLSTM处理
        x = self._checkpoint_function(self.mid_res_blocks2, x)
        
        if self.debug_mode:
            print(f"After mid blocks: {x.shape}")
        
        # 得到分布参数
        dist_params = self.distribution_adapter(x)
        mean, log_var = torch.chunk(dist_params, 2, dim=1)
        
        # 重参数化采样
        if sample:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mean + eps * std
            return z, mean, log_var
        else:
            return mean, mean, log_var

class ConvGRUCell(nn.Module):
    """卷积GRU单元"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvGRUCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # 单个卷积层处理更新门和重置门
        self.conv_gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=2 * hidden_dim,  # r, z (reset, update)
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        # 候选隐藏状态的卷积
        self.conv_candidate = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=hidden_dim,  # h_tilde
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.data.fill_(0)
                # 可以设置更新门的偏置为1，有助于训练初期保留信息
                if 'conv_gates' in name:
                    # 更新门偏置在第二个半部分
                    param.data[self.hidden_dim:] = 1.0

    def forward(self, x, h_prev=None):
        batch_size, _, height, width = x.size()
        
        # 初始化隐藏状态
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, height, width, 
                               device=x.device, dtype=x.dtype)
        
        # 合并输入和先前隐藏状态
        combined = torch.cat([x, h_prev], dim=1)
        
        # 计算重置门和更新门
        gates = self.conv_gates(combined)
        r, z = torch.split(gates, self.hidden_dim, dim=1)
        
        r = torch.sigmoid(r)  # 重置门
        z = torch.sigmoid(z)  # 更新门
        
        # 计算候选隐藏状态
        combined_candidate = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_candidate(combined_candidate))
        
        # 更新隐藏状态
        h_next = (1 - z) * h_prev + z * h_tilde
        
        return h_next

    def reset_states(self):
        """重置状态（实际在前向传播中处理）"""
        pass


class MultiScaleConvGRU(nn.Module):
    """多层ConvGRU模块"""
    def __init__(self, input_dim, hidden_dims, num_layers=2, kernel_size=3):
        super(MultiScaleConvGRU, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims if isinstance(hidden_dims, list) else [hidden_dims] * num_layers
        
        # 创建多层ConvGRU
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else self.hidden_dims[i-1]
            self.layers.append(ConvGRUCell(layer_input_dim, self.hidden_dims[i], kernel_size))
        
        # 状态存储
        self.hidden_states = [None] * num_layers

    def forward(self, x):
        """
        前向传播 - 处理单个时间步
        Args:
            x: 输入张量 [B, C, H, W]
        Returns:
            输出张量 [B, hidden_dims[-1], H, W]
        """
        current_input = x
        
        for layer_idx in range(self.num_layers):
            h_prev = self.hidden_states[layer_idx]
            h_next = self.layers[layer_idx](current_input, h_prev)
            
            # 更新状态
            self.hidden_states[layer_idx] = h_next
            
            current_input = h_next
        
        return current_input

    def reset_states(self):
        """重置所有层的状态"""
        self.hidden_states = [None] * self.num_layers

class OptimizedEventVAEEncoderv9(nn.Module):
    """使用ConvGRU替代ConvLSTM的编码器"""
    def __init__(self, 
                 input_channels=5, 
                 latent_channels=8,
                 base_channels=64,
                 expansion_ratios=[2, 3, 4],
                 temporal_fusion="gated",
                 convgru_layers=1,
                 use_checkpoint=True,
                 debug_mode=False):
        super().__init__()
        
        self.debug_mode = debug_mode
        self.use_checkpoint = use_checkpoint
        self.latent_channels = latent_channels
        
        # 计算各层通道数
        self.channels = [
            base_channels,
            base_channels * 2,
            base_channels * 3,
            base_channels * 4
        ]
        
        if debug_mode:
            print(f"Channel configuration: {self.channels}")
            print(f"ConvGRU layers: {convgru_layers}")
        
        # 输入适配层
        self.input_adapter = nn.Sequential(
            DepthwiseSeparableConv(input_channels, self.channels[0]),
            EfficientResBlock(self.channels[0], expansion_ratios[0])
        )
        
        # 多尺度特征提取（使用ConvGRU替换时序融合）
        self.feature_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()  # 现在使用ConvGRU
        self.downsample_blocks = nn.ModuleList()
        
        for i in range(3):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            
            # 特征提取块
            feature_block = nn.Sequential(
                EfficientResBlock(in_ch, expansion_ratios[i]),
                EfficientResBlock(in_ch, expansion_ratios[i]),
            )
            self.feature_blocks.append(feature_block)
            
            # 使用ConvGRU代替ConvLSTM
            temporal_block = MultiScaleConvGRU(
                input_dim=in_ch,
                hidden_dims=in_ch,  # 保持相同维度
                num_layers=convgru_layers,
                kernel_size=3
            )
            self.temporal_blocks.append(temporal_block)
            
            # 下采样
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            )
            self.downsample_blocks.append(downsample)
        
        # 中间处理块（也使用ConvGRU）
        final_channels = self.channels[-1]
        
        # 重新设计中间块以包含ConvGRU
        self.mid_res_blocks1 = nn.Sequential(
            EfficientResBlock(final_channels, expansion_ratios[-1]),
            EfficientResBlock(final_channels, expansion_ratios[-1]),
        )
        
        self.mid_temporal = MultiScaleConvGRU(
            input_dim=final_channels,
            hidden_dims=final_channels,
            num_layers=convgru_layers,
            kernel_size=3
        )
        
        self.mid_res_blocks2 = nn.Sequential(
            EfficientResBlock(final_channels, expansion_ratios[-1]),
            EfficientResBlock(final_channels, expansion_ratios[-1]),
        )
        
        # 输出分布参数
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
        """重置所有ConvGRU的状态"""
        for temporal_block in self.temporal_blocks:
            temporal_block.reset_states()
        self.mid_temporal.reset_states()
    
    def _checkpoint_function(self, func, *args):
        """条件性使用梯度检查点 - 跳过时序模块"""
        # ConvGRU模块不使用检查点（因为有内部状态）
        if any(isinstance(m, (MultiScaleConvGRU, ConvGRUCell)) for m in func.modules()):
            return func(*args)
        
        # 其他模块使用检查点
        if self.use_checkpoint and self.training:
            return checkpoint(func, *args, use_reentrant=False)
        else:
            return func(*args)
    
    def forward(self, x, sample=True):
        """
        处理单个时间步的事件数据（使用ConvGRU进行时序建模）
        Args:
            x: [B, 5, H, W] 事件voxel数据
            sample: 是否进行采样
        Returns:
            如果sample=True: (z, mean, log_var)
            如果sample=False: (mean, mean, log_var)
        """
        if self.debug_mode:
            print(f"Input shape: {x.shape}")
            
        # 输入适配
        x = self.input_adapter(x)
        if self.debug_mode:
            print(f"After input adapter: {x.shape}")
        
        # 多尺度特征提取和ConvGRU时序建模
        for i, (feat_block, temp_block, down_block) in enumerate(
            zip(self.feature_blocks, self.temporal_blocks, self.downsample_blocks)
        ):
            # 特征提取
            x = self._checkpoint_function(feat_block, x)
            
            # ConvGRU时序处理（不使用检查点）
            x = temp_block(x)
            
            # 下采样
            x = down_block(x)
            
            if self.debug_mode:
                print(f"After block {i}: {x.shape}")
        
        # 中间处理
        x = self._checkpoint_function(self.mid_res_blocks1, x)
        x = self.mid_temporal(x)  # ConvGRU处理
        x = self._checkpoint_function(self.mid_res_blocks2, x)
        
        if self.debug_mode:
            print(f"After mid blocks: {x.shape}")
        
        # 得到分布参数
        dist_params = self.distribution_adapter(x)
        mean, log_var = torch.chunk(dist_params, 2, dim=1)
        
        # 重参数化采样
        if sample:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mean + eps * std
            return z, mean, log_var
        else:
            return mean, mean, log_var


class WindowAttention(nn.Module):
    """简化的窗口注意力，固定窗口数量"""
    def __init__(self, dim, num_windows=16, shift_size=0, num_heads=1):
        super().__init__()
        self.num_windows = num_windows
        self.shift_size = shift_size
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        num_win = self.num_windows
        
        # 计算窗口大小
        win_h = (H + num_win - 1) // num_win
        win_w = (W + num_win - 1) // num_win
        
        # 自动padding
        pad_h = (num_win * win_h - H) % (num_win * win_h)
        pad_w = (num_win * win_w - W) % (num_win * win_w)
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[2], x.shape[3]

        # shift计算
        if self.shift_size > 0:
            shift_h = min(self.shift_size, Hp)
            shift_w = min(self.shift_size, Wp)
            x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(2, 3))

        # 窗口划分
        x = x.view(B, C, Hp // win_h, win_h, Wp // win_w, win_w)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(-1, win_h * win_w, C)

        # 注意力
        out, _ = self.attn(x, x, x)

        # 恢复形状
        out = out.view(B, Hp // win_h, Wp // win_w, win_h, win_w, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(B, C, Hp, Wp)

        # shift back
        if self.shift_size > 0:
            out = torch.roll(out, shifts=(shift_h, shift_w), dims=(2, 3))

        # 去padding
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out


class SimpleTemporalFusion(nn.Module):
    def __init__(self, channels, num_frames=3, num_heads=1, num_windows=16):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        
        # 时序注意力：在时间维度上融合多帧信息
        self.temporal_attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        
        # 窗口注意力（类似Swin SR，两层shift）
        self.window_attn1 = WindowAttention(channels, num_windows=num_windows, 
                                           shift_size=0, num_heads=num_heads)
        self.window_attn2 = WindowAttention(channels, num_windows=num_windows,
                                           shift_size=num_windows//2, num_heads=num_heads)
        
        # 简单的输出投影
        self.output_proj = nn.Conv2d(channels, channels, 1)
        
        # 状态管理
        self.frame_buffer = []

    def reset_states(self):
        self.frame_buffer = []

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 更新帧缓冲区
        self.frame_buffer.append(x.detach())
        if len(self.frame_buffer) > self.num_frames:
            self.frame_buffer.pop(0)
        
        # 如果没有足够的历史帧，直接返回
        if len(self.frame_buffer) < 2:
            return x
        
        # 时序注意力：在时间维度融合
        frames = torch.stack(self.frame_buffer)  # [T, B, C, H, W]
        T = frames.shape[0]
        
        # 重塑为时序序列 [B*H*W, T, C]
        temporal_seq = frames.permute(1, 3, 4, 0, 2).contiguous()  # [B, H, W, T, C]
        temporal_seq = temporal_seq.view(B * H * W, T, C)
        
        # 时序注意力（关注时间维度关系）
        temporal_out, _ = self.temporal_attn(temporal_seq, temporal_seq, temporal_seq)
        
        # 取最后一帧（当前帧增强后的结果）
        temporal_out = temporal_out[:, -1, :]  # [B*H*W, C]
        temporal_out = temporal_out.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 窗口注意力处理（类似Swin SR的两层结构）
        # 第一层：无shift
        window_out = self.window_attn1(temporal_out)
        # 第二层：有shift
        window_out = self.window_attn2(window_out)
        
        # 最终输出投影
        out = self.output_proj(window_out)
        
        return out


class OptimizedEventVAEEncoderv8(nn.Module):
    def __init__(
        self,
        input_channels=5,
        latent_channels=8,
        base_channels=64,
        expansion_ratios=[2, 3, 4],
        temporal_fusion="gated",
        temporal_layers=[0, 1],
        stage_depth=[2, 2, 2],
        mid_depth=4,
        use_checkpoint=True,
        debug_mode=False,
        # 新增参数：mid block中的注意力配置
        use_mid_attention=True,
        mid_attention_heads=4
    ):
        super().__init__()

        self.debug_mode = debug_mode
        self.use_checkpoint = use_checkpoint
        self.latent_channels = latent_channels
        self.use_mid_attention = use_mid_attention

        # 解析 temporal_layers
        self.temporal_layers = self._parse_temporal_layers(temporal_layers)

        # 通道配置
        self.channels = [
            base_channels,
            base_channels * 2,
            base_channels * 3,
            base_channels * 4
        ]

        # 输入适配
        self.input_adapter = nn.Sequential(
            DepthwiseSeparableConv(input_channels, self.channels[0]),
            EfficientResBlock(self.channels[0], expansion_ratios[0])
        )

        # 特征提取 + 时序融合
        self.feature_blocks = nn.ModuleList()
        self.temporal_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        for i in range(3):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]

            blocks = []
            for _ in range(stage_depth[i]):
                blocks.append(EfficientResBlock(in_ch, expansion_ratios[i]))

            self.feature_blocks.append(nn.Sequential(*blocks))
            # 时序模块
            if self.temporal_layers[i]:
                self.temporal_blocks.append(
                    SimpleTemporalFusion(in_ch)
                )
            else:
                self.temporal_blocks.append(None)


            # 下采样
            self.downsample_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.SiLU()
                )
            )

        # ⭐ 改进的中间层：可选的注意力机制
        final_channels = self.channels[-1]
        mid_blocks = []
        mid_blocks.append(EfficientResBlock(final_channels, expansion_ratios[-1]))
        mid_blocks.append(EfficientResBlock(final_channels, expansion_ratios[-1]))
        
        for i in range(mid_depth):            
            # 在特定位置插入注意力（比如每2个block插入一个）
            if self.use_mid_attention and i % 2 == 1:  # 在第2、4...个block前插入
                # 窗口注意力：由于特征图小，可以使用较小的窗口数量
                mid_blocks.append(
                    WindowAttention(final_channels, num_windows=4, 
                                    shift_size=0, num_heads=mid_attention_heads)
                )
                # 添加第二个shift的注意力
                mid_blocks.append(
                    WindowAttention(final_channels, num_windows=4,
                                    shift_size=2, num_heads=mid_attention_heads)
                )
            mid_blocks.append(EfficientResBlock(final_channels, expansion_ratios[-1]))

        self.mid_block = nn.Sequential(*mid_blocks)

        # 输出分布适配器
        self.distribution_adapter = nn.Sequential(
            nn.GroupNorm(32, final_channels),
            nn.SiLU(),
            nn.Conv2d(final_channels, 2 * latent_channels, 3, padding=1)
        )

        if debug_mode:
            self._print_model_info()

    # --------------- helper: 解析用户传入的 temporal_layers ----------
    def _parse_temporal_layers(self, layers):
        if isinstance(layers, list) and len(layers) == 3 and all(isinstance(x, bool) for x in layers):
            return layers
        out = [False, False, False]
        for idx in layers:
            if 0 <= idx < 3:
                out[idx] = True
        return out

    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}, Trainable: {trainable:,}")
        print(f"Temporal layers enabled: {self.temporal_layers}")

    # ------------ reset temporal states -------------
    def reset_states(self):
        for tb in self.temporal_blocks:
            if tb is not None:
                tb.reset_states()

    # 检查点逻辑
    def _checkpoint(self, func, *args):
        if any(isinstance(m, SimpleTemporalFusion) for m in func.modules()):
            return func(*args)
        if self.use_checkpoint and self.training:
            return checkpoint(func, *args, use_reentrant=False)
        return func(*args)

    # ---------------- forward -----------------
    def forward(self, x, sample=True):
        x = self.input_adapter(x)

        # 3 stages
        for feat_block, temp_block, down_block in zip(
                self.feature_blocks, self.temporal_blocks, self.downsample_blocks):

            x = self._checkpoint(feat_block, x)
            if temp_block is not None:
                x = temp_block(x)
            x = down_block(x)

        # 中间层
        x = self._checkpoint(self.mid_block, x)

        # VAE 分布
        dist = self.distribution_adapter(x)
        mean, log_var = torch.chunk(dist, 2, dim=1)

        if sample:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + eps * std, mean, log_var
        else:
            return mean, mean, log_var


def tv_loss_fn(x, weight=0.00001):
    # 计算水平梯度（沿着宽度方向）
    dx = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])  # 去除边界的差异
    dy = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])  # 去除边界的差异

    # 为了保证大小一致，对dx和dy做切片操作以匹配维度
    dx = dx[:, :, :, :-1]  # 去除dx的最后一列，保证宽度一致
    dy = dy[:, :, :-1, :]  # 去除dy的最后一行，保证高度一致

    # 计算水平和垂直梯度的L2范数
    loss = torch.sum(dx + dy)
    
    # 加权（可以通过调节权重来控制平滑的强度）
    return weight * loss


def test_forward_pass(model, device='cuda'):
    """测试模型前向传播"""
    print("\n" + "="*60)
    print("测试模型前向传播")
    print("="*60)
    
    model = model.to(device)
    model.eval()
    
    # 测试不同尺寸
    test_sizes = [
        (1, 3, 128, 128),
        (1, 3, 256, 256),
        (1, 3, 512, 512),
        (2, 3, 512, 512),  # batch size = 2
    ]
    
    for size in test_sizes:
        try:
            dummy_input = torch.randn(*size).to(device)
            with torch.no_grad():
                output, residual = model(dummy_input)
            
            print(f"✓ 输入: {size} -> 输出: {output.shape}, 残差: {residual.shape}")
            
            # 检查输出范围
            print(f"  输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
            print(f"  残差值范围: [{residual.min().item():.4f}, {residual.max().item():.4f}]")
            
        except Exception as e:
            print(f"✗ 输入: {size} 失败: {str(e)}")
    
    print("="*60 + "\n")

def calculate_model_complexity(model, input_size=(3, 512, 512)):
    """
    计算模型的参数量和计算量
    需要安装: pip install thop
    如果没有thop，会使用简化版本计算
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("模型复杂度分析")
    print("="*60)
    print(f"输入尺寸: {input_size}")
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 尝试使用thop计算FLOPs
    try:
        from thop import profile, clever_format
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"FLOPs: {flops}")
        print(f"Params (thop): {params}")
    except ImportError:
        print("提示: 安装 thop 可以获得更详细的FLOPs计算")
        print("      pip install thop")
        # 简化计算
        with torch.no_grad():
            _ = model(dummy_input)
    
    # 测试推理速度
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(dummy_input)
        
        # 计时
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        import time
        start_time = time.time()
        num_iterations = 100
        
        for _ in range(num_iterations):
            _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        fps = 1.0 / avg_time
        
        print(f"平均推理时间: {avg_time*1000:.2f} ms")
        print(f"FPS: {fps:.2f}")
    
    print("="*60 + "\n")
    
    return total_params, trainable_params

# 使用示例
if __name__ == "__main__":
    # encoder = OptimizedEventVAEEncoderv2(use_checkpoint=False, base_channels=96, temporal_fusion="gated")
    # encoder = OptimizedEventVAEEncoderv5(use_checkpoint=False, latent_channels=16, base_channels=96, temporal_fusion="gated")
    # encoder = OptimizedEventVAEEncoderv7(use_checkpoint=True, latent_channels=16, convlstm_layers=1, base_channels=96)
    encoder = OptimizedEventVAEEncoderv9(use_checkpoint=True, latent_channels=16, convgru_layers=1, base_channels=96)
    # encoder = OptimizedEventVAEEncoderv8(use_checkpoint=True, latent_channels=16, base_channels=64, temporal_layers=[1,2], 
    #                                      stage_depth=[2,2,6], mid_depth=12, use_mid_attention=False)
    encoder = encoder.to('cuda')
    analyze_encoder(encoder)
    analyze_model_efficiency(encoder)
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
    #     "/path/to/sd3-medium",  # 示例预训练模型
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