import os
import sys
sys.path.append(os.getcwd())
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import glob
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
from collections import defaultdict
import cv2
import numpy as np
import random
from tqdm import tqdm
from skimage.util import random_noise
import argparse
from diffusers import AutoencoderKL

'''
train
    -gt
        -a
            -airfield
                -00000001.jpg
                -00000002.jpg
                -...
    -lr_bicubic
        -a
            -airfield
                -00000001.jpg
                -00000002.jpg
                -...
    -prompt_txt
        -a
            -airfield
                -00000001.txt
                -00000002.txt
                -...
    -prompt_embeds
        -a
            -airfield
                -00000001.pt
                -00000002.pt
                -...
    -pool_embeds
        -a
            -airfield
                -00000001.pt
                -00000002.pt
                -...
    -latent_hr
        -a
            -airfield
                -00000001.pt
                -00000002.pt
                -...
'''
# Paths to the datasets
# NOTE: new Places365 layout only has images under gt/ (no prompt_txt/).
# Step-1 runs with --use_default_prompt, so per-image prompts are unused;
# _validate_file_batch below treats a missing prompt file as non-fatal.
# Dataset layout on server: <PLACES365_PATH>/gt/<a>/<class>/<id>.jpg
#   当前结构：data/places365/train/gt/<a>/<class>/<id>.jpg
#   所以 PLACES365_PATH 指向 places365/train，拼接 hr_dir_name="gt" 得到 HR 根目录
PLACES365_PATH = os.environ.get("PLACES365_PATH", "data/places365/train")

data_path = [PLACES365_PATH]
hr_dir_name = "gt"
prompt_dir_name = "prompt_txt"

class Real_ESRGAN_Dataset_strict(Dataset):
    def __init__(self,
                root_dir_path=data_path,
                process_size=512,
                max_sample=None,
                device="cpu",
                max_workers=12,  # 新增：并行工作线程数
                enable_vae_cache=False,
                ):
        self.device = device
        self.max_sample = max_sample
        self.process_size = process_size
        self.hr_dir_name = hr_dir_name
        self.prompt_dir_name = prompt_dir_name
        self.trans = transforms.ToTensor()
        self.max_workers = max_workers

        # 存储所有文件路径的列表
        self.hr_img_name = []
        self.prompt_name = []

        # 用于调试的计数器
        self.missing_count = {
            'prompt': 0,
            'hr': 0
        }

        self.degrader = AdvancedGrayscaleDegrader()  # 创建退化类实例
        # 使用线程锁保护共享数据
        self.lock = threading.Lock()

        self._initialize_dataset_parallel(root_dir_path)

        self.img_nums = len(self.hr_img_name)
        if self.img_nums == 0:
            raise ValueError("No valid samples found! Please check your dataset paths and structure.")

    def _get_all_image_files_fast(self, directory):
        """快速获取所有图像文件，使用glob模式匹配"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []

        for ext in image_extensions:
            # 使用 ** 进行递归搜索
            pattern = os.path.join(directory, '**', ext)
            image_files.extend(glob.glob(pattern, recursive=True))
            # 同时搜索大写扩展名
            pattern = os.path.join(directory, '**', ext.upper())
            image_files.extend(glob.glob(pattern, recursive=True))

        return image_files

    def _validate_file_batch(self, hr_files_batch, dataset_paths):
        """批量验证文件存在性"""
        valid_samples = []
        local_missing_count = defaultdict(int)

        hr_data_path, prompt_path = dataset_paths

        for hr_file in hr_files_batch:
            rel_path = os.path.relpath(hr_file, hr_data_path)
            base_name = os.path.splitext(rel_path)[0]

            # 构建对应的所有文件路径
            prompt_file = os.path.join(prompt_path, base_name + ".txt")

            # prompt 在新数据集上可选：缺失时记一笔但不丢弃样本
            # （step-1 使用 --use_default_prompt，逐图 prompt 本就不参与训练）
            if os.path.exists(prompt_file):
                prompt_for_sample = prompt_file
            else:
                local_missing_count['prompt'] += 1
                prompt_for_sample = None

            valid_samples.append({
                'hr': hr_file,
                'prompt': prompt_for_sample,
            })

        return valid_samples, local_missing_count

    def _initialize_dataset_parallel(self, root_dir_path):
        """并行初始化数据集"""
        for root_dir in root_dir_path:
            if not os.path.exists(root_dir):
                print(f"Warning: Dataset path {root_dir} does not exist, skipping...")
                continue

            print(f"Processing dataset: {root_dir}")

            # 构建各个子目录的路径
            hr_data_path = os.path.join(root_dir, self.hr_dir_name)
            prompt_path = os.path.join(root_dir, self.prompt_dir_name)

            # 检查所有必要目录的存在性
            # prompt_txt 目录在新数据集上不存在：只要求 HR 目录，
            # prompt 缺失时降级为空 prompt（见 _validate_file_batch）
            required_dirs = [
                (hr_data_path, 'HR images'),
            ]
            if not os.path.exists(prompt_path):
                print(f"Warning: no prompt directory {prompt_path} — prompts will be empty")

            dirs_ok = True
            for path, name in required_dirs:
                if not os.path.exists(path):
                    print(f"Error: Missing required directory {name}: {path}")
                    dirs_ok = False

            if not dirs_ok:
                print(f"Skipping dataset {root_dir} due to missing directories")
                continue

            # 快速获取所有LR图像文件
            print("hr_data_path",hr_data_path)
            hr_files = self._get_all_image_files_fast(hr_data_path)
            print(f"Found {len(hr_files)} HR images in {hr_data_path}")
            print(f"Performing parallel validation with {self.max_workers} workers...")

            # 将文件列表分批处理
            batch_size = max(100, len(hr_files) // (self.max_workers * 4))  # 动态调整批大小
            hr_file_batches = [hr_files[i:i + batch_size] for i in range(0, len(hr_files), batch_size)]

            dataset_paths = (hr_data_path, prompt_path)

            # 并行处理文件验证
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有批次任务
                future_to_batch = {
                    executor.submit(self._validate_file_batch, batch, dataset_paths): batch
                    for batch in hr_file_batches
                }

                # 收集结果
                for future in as_completed(future_to_batch):
                    try:
                        valid_samples, local_missing_count = future.result()

                        # 线程安全地更新共享数据
                        with self.lock:
                            for sample in valid_samples:
                                self.prompt_name.append(sample['prompt'])
                                self.hr_img_name.append(sample['hr'])

                            # 更新缺失文件统计
                            for key, count in local_missing_count.items():
                                self.missing_count[key] += count

                    except Exception as exc:
                        print(f'Batch processing generated an exception: {exc}')

            print(f"Validation completed for {root_dir}")

    def _get_all_image_files(self, root_dir):
        """递归获取所有图像文件路径"""
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        all_image_paths = []

        for ext in image_extensions:
            # 使用 ** 进行递归搜索
            pattern = os.path.join(root_dir, '**', ext)
            paths = glob.glob(pattern, recursive=True)
            all_image_paths.extend(paths)

        return sorted(all_image_paths)

    def __getitem__(self, idx):
        try:
            # 加载LR和HR图像
            hr_pil = Image.open(self.hr_img_name[idx]).convert("RGB")
            hr_numpy = np.array(hr_pil)
            lr_degraded_numpy = self.degrader.random_degradation(hr_numpy)
            if len(lr_degraded_numpy.shape) == 2:  # 灰度图
                lr_degraded_numpy = np.stack([lr_degraded_numpy]*3, axis=-1)
            lr_pil = Image.fromarray(lr_degraded_numpy.astype(np.uint8))

            lr_img = self.trans(lr_pil) * 2 - 1
            hr_img = self.trans(hr_pil).squeeze() * 2 - 1


            # 加载prompt文本
            if self.prompt_name[idx] and os.path.exists(self.prompt_name[idx]):
                with open(self.prompt_name[idx], "r", encoding='utf-8') as f:
                    prompt = f.read().strip()
            else:
                prompt = "empty prompt"

            return {
                "lr_img": lr_img,
                "hr_img": hr_img,
                "prompt_text": prompt,
                "hr_path": self.hr_img_name[idx],
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            print(f"LR path: {self.lr_img_name[idx]}")
            print(f"HR path: {self.hr_img_name[idx]}")
            # 打印缓存统计信息用于调试
            if self.prompt_stats["total_count"] > 0:
                empty_ratio = self.prompt_stats["empty_count"] / self.prompt_stats["total_count"]
                print(f"Empty prompt ratio: {empty_ratio:.2%}")

    def __len__(self):
        if self.max_sample:
            return min(self.max_sample, self.img_nums)
        return self.img_nums

    def get_sample_info(self, idx):
        """获取样本的详细信息，用于调试"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        return {
            "hr_img_path": self.hr_img_name[idx],
            "prompt_path": self.prompt_name[idx],
        }

    def print_dataset_stats(self):
        """打印数据集统计信息"""
        print(f"\n=== Strict Dataset Statistics ===")
        print(f"Total valid samples: {len(self)}")
        print(f"Max samples (if limited): {self.max_sample}")
        print(f"Process size: {self.process_size}")
        print(f"Device: {self.device}")

        if len(self) > 0:
            print(f"\nSample file paths:")
            sample_info = self.get_sample_info(0)
            for key, path in sample_info.items():
                if path:
                    rel_path = os.path.relpath(path, os.path.dirname(path).split(os.sep)[0])
                    print(f"  {key}: {rel_path}")
                else:
                    print(f"  {key}: Not available")

        # 打印缺失文件统计
        print(f"\nMissing file statistics:")
        for key, count in self.missing_count.items():
            print(f"  Missing {key}: {count}")

        # 统计每个数据集的样本数量
        print(f"\nSamples per dataset:")
        for i, root_dir in enumerate(data_path):
            if not os.path.exists(root_dir):
                continue
            count = sum(1 for path in self.lr_img_name if path.startswith(root_dir))
            print(f"  {os.path.basename(root_dir)}: {count}")


# 泊松噪声
class AdvancedGrayscaleDegrader:
    def __init__(self):
        # 初始化各种退化参数的范围
        self.noise_intensity = (0.01, 0.1)         # 高斯噪声强度（扩大范围）
        self.sensor_range = (0.05, 0.3)            # 传感器伪影大小范围
        self.sensor_intensity = (0.2, 0.6)         # 伪影可见度（扩大范围）
        self.edge_blur_range = (1.0, 4.0)          # 边缘模糊（扩大范围）
        self.edge_discontinuity_prob = 0.4          # 边缘不连续概率（提高）
        self.block_range = (15, 50)                # 块状伪影大小范围（扩大）
        self.block_intensity = (0.1, 0.7)           # 块状伪影强度（扩大范围）
        self.low_freq_noise_range = (0.01, 0.08)    # 低频噪声（扩大范围）
        self.line_drop_prob = 0.05                  # 行丢失概率
        self.low_freq_prob = 0.4                     # 应用运动模糊的概率

        # 优化运动模糊参数
        self.motion_blur_angle_range = (-30, 30)       # 更小的角度范围（自然抖动）
        self.motion_blur_length_range = (3, 20)         # 更短的模糊范围
        self.global_blur_prob = 0.3                     # 全局模糊的概率（局部优先）
        self.motion_strength_range = (0.5, 1.0)         # 模糊强度控制
        self.ghost_prob = 0.25                          # 重影效果的概率
        self.object_blur_min_size = 0.15                # 模糊对象的最小尺寸（相对于图像）
        self.object_blur_max_size = 0.45                # 模糊对象的最大尺寸
        self.object_shape_probabilities = [0.6, 0.3, 0.1] # 形状概率：椭圆 > 多边形 > 轮廓
        self.shape_irregularity = (0.1, 0.5)            # 形状不规则程度


    def add_motion_blur(self, image):
        """添加更自然的运动模糊效果"""
        # 优先使用对象运动模糊，更自然
        if random.random() > self.global_blur_prob:
            return self._add_object_motion_blur(image)
        else:
            blurred = self._add_global_motion_blur(image)

            # 有概率在全局模糊基础上叠加对象模糊
            if random.random() < 0.4:
                blurred = self._add_object_motion_blur(blurred)

            return blurred

    def _add_global_motion_blur(self, image):
        """添加更自然的全局运动模糊（模拟相机抖动）"""
        h, w = image.shape

        # 随机生成模糊角度和长度（更保守的参数）
        angle = random.uniform(*self.motion_blur_angle_range)
        base_length = random.uniform(*self.motion_blur_length_range)

        # 根据图像大小调整模糊强度
        scaling_factor = min(h, w) / 512  # 以512px为基准
        length = max(2, int(base_length * scaling_factor * 0.8))  # 缩小强度

        # 转换为弧度
        angle_rad = np.deg2rad(angle)

        # 创建运动模糊核
        kernel_size = length * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2

        # 使用非直线模糊路径（轻微波浪形更真实）
        points = []
        for i in range(-length, length + 1):
            offset = np.sin(i / max(1, length) * np.pi) * 0.3  # 轻微波动
            dx = int(i * np.cos(angle_rad) + offset * np.sin(angle_rad))
            dy = int(i * np.sin(angle_rad) + offset * np.cos(angle_rad))
            points.append((center + dx, center + dy))

        # 在核上绘制曲线路径
        for i in range(1, len(points)):
            cv2.line(kernel, points[i-1], points[i], 1, thickness=1)

        # 归一化核（添加轻微衰减效果）
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            # 防止除零错误
            return image

        # 应用模糊
        blurred = cv2.filter2D(image.astype(np.float32), -1, kernel)

        # 添加重影效果（概率降低）
        if random.random() < self.ghost_prob and length > 7:
            ghost_intensity = random.uniform(0.02, 0.08)  # 更轻微的重影
            ghost_offset = random.randint(2, 5)

            # 创建偏移图像作为重影
            ghost = np.zeros_like(blurred)
            dx = int(ghost_offset * np.cos(angle_rad))
            dy = int(ghost_offset * np.sin(angle_rad))
            src_y1, src_y2 = max(0, -dy), min(h, h - dy)
            src_x1, src_x2 = max(0, -dx), min(w, w - dx)
            dst_y1, dst_y2 = max(0, dy), min(h, h + dy)
            dst_x1, dst_x2 = max(0, dx), min(w, w + dx)

            ghost[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

            # 混合重影（部分混合）
            blended = cv2.addWeighted(blurred, 1 - ghost_intensity, ghost, ghost_intensity, 0)
            # 只在高强度区域使用重影
            mask = cv2.threshold(ghost, 20, 255, cv2.THRESH_BINARY)[1].astype(np.float32) / 255
            result = blended * mask + blurred * (1 - mask)
            return np.clip(result, 0, 255).astype(np.uint8)

        return np.clip(blurred, 0, 255).astype(np.uint8)

    def _generate_irregular_shape(self, size_x, size_y, irregularity):
        """生成不规则多边形形状"""
        points = []
        num_points = random.randint(5, 8)  # 顶点数量

        # 生成基础椭圆点
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            rx = size_x / 2 * (1 + random.uniform(-irregularity, irregularity))
            ry = size_y / 2 * (1 + random.uniform(-irregularity, irregularity))
            points.append((
                int(size_x / 2 + rx * np.cos(angle)),
                int(size_y / 2 + ry * np.sin(angle))
            ))

        # 随机扰动点位置
        for i in range(num_points):
            points[i] = (
                points[i][0] + random.randint(-int(size_x*0.1), int(size_x*0.1)),
                points[i][1] + random.randint(-int(size_y*0.1), int(size_y*0.1))
            )

        return np.array(points)

    def _add_object_motion_blur(self, image):
        """模拟运动物体的模糊效果"""
        h, w = image.shape

        # 随机确定模糊区域尺寸
        object_w = random.randint(
            int(w * self.object_blur_min_size),
            int(w * self.object_blur_max_size)
        )
        object_h = random.randint(
            int(h * self.object_blur_min_size),
            int(h * self.object_blur_max_size)
        )

        # 随机确定中心点
        center_x = random.randint(object_w // 2, w - object_w // 2)
        center_y = random.randint(object_h // 2, h - object_h // 2)

        # 随机选择模糊方向
        direction_angle = random.uniform(0, 360)
        dx = np.cos(np.deg2rad(direction_angle))
        dy = np.sin(np.deg2rad(direction_angle))

        # 模糊长度随物体大小变化
        max_length = min(object_w, object_h) // 2
        blur_length = random.randint(max(2, max_length // 4), max_length // 2)

        # 创建运动模糊核（小尺寸）
        kernel_size = max(5, blur_length * 2 - 1)
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        cv2.line(kernel,
                 (center - int(dx * blur_length), center - int(dy * blur_length)),
                 (center + int(dx * blur_length), center + int(dy * blur_length)),
                 1, thickness=1)

        # 归一化核
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        else:
            kernel = None

        # 创建掩模
        mask = np.zeros((h, w), dtype=np.float32)

        # 选择形状类型：椭圆、多边形或轮廓（不同概率）
        shape_type = random.choices(
            ["ellipse", "polygon", "contour"],
            self.object_shape_probabilities
        )[0]

        irregularity = random.uniform(*self.shape_irregularity)

        if shape_type == "ellipse":
            # 椭圆形
            cv2.ellipse(mask, (center_x, center_y),
                       (object_w // 2, object_h // 2),
                       random.uniform(0, 360), 0, 360, 1, -1)
        elif shape_type == "polygon":
            # 多边形
            points = self._generate_irregular_shape(object_w, object_h, irregularity)
            # 偏移到正确位置
            points += (center_x - object_w // 2, center_y - object_h // 2)
            cv2.fillConvexPoly(mask, points.astype(int), 1)
        else:  # contour
            # 从图像中提取真实轮廓
            canny_edges = cv2.Canny(image, 100, 200)
            contours, _ = cv2.findContours(
                canny_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 选择包含中心点的轮廓
                valid_contours = [
                    cnt for cnt in contours
                    if cv2.pointPolygonTest(cnt, (center_x, center_y), False) >= 0
                ]

                if valid_contours:
                    cnt = max(valid_contours, key=cv2.contourArea)
                    cv2.drawContours(mask, [cnt], -1, 1, -1)

            # 如果没有找到合适轮廓，回退到椭圆
            if np.max(mask) < 0.5:
                cv2.ellipse(mask, (center_x, center_y),
                           (object_w // 2, object_h // 2),
                           random.uniform(0, 360), 0, 360, 1, -1)

        # 羽化边缘（重要）
        feather_radius = min(object_w, object_h) // 4
        mask_blurred = cv2.GaussianBlur(mask,
                                       (feather_radius * 2 + 1, feather_radius * 2 + 1),
                                       feather_radius)

        # 应用模糊（如果创建了有效的模糊核）
        if kernel is not None:
            region_blurred = cv2.filter2D(
                image.astype(np.float32), -1, kernel)
        else:
            region_blurred = image.astype(np.float32)

        # 随机强度混合
        strength = random.uniform(*self.motion_strength_range)
        result = image.copy().astype(np.float32)

        # 应用模糊区域
        result = (1 - mask_blurred * strength) * image + mask_blurred * strength * region_blurred

        return np.clip(result, 0, 255).astype(np.uint8)

    def add_adaptive_gaussian_noise(self, image):
        """添加自适应高斯噪声，在平坦区域增加更多噪声"""
        # 计算图像的局部方差
        mean, stddev = cv2.meanStdDev(image)
        local_variance = cv2.Laplacian(image, cv2.CV_64F).var()

        # 基于图像复杂度调整噪声强度
        min_intensity, max_intensity = self.noise_intensity
        if local_variance < 1000:  # 低纹理区域
            intensity = random.uniform(min_intensity, max_intensity * 2)
        else:
            intensity = random.uniform(min_intensity, max_intensity)

        # 添加噪声
        noisy = random_noise(image, mode='gaussian', var=intensity**2)
        return (noisy * 255).astype(np.uint8)

    def create_block_artifacts(self, image, grain_strength=0.2):
        """在低细节区域创建自然、颗粒感强烈的伪影（灰度专用）

        参数:
            image: 输入灰度图像
            grain_strength: 颗粒感强度系数 (0.0-1.0)，默认为1.0
        """
        h, w = image.shape

        # 1. 低细节区域检测 - 更宽松的参数
        blur1 = cv2.GaussianBlur(image.astype(np.float32), (9, 9), 2.0)

        # 计算局部方差
        kernel = np.ones((30, 30), np.float32) / 900
        mean = cv2.filter2D(blur1, -1, kernel)
        sqr_mean = cv2.filter2D(blur1**2, -1, kernel)
        variance = sqr_mean - mean**2

        # 结合梯度信息
        grad_x = cv2.Sobel(blur1, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur1, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # 归一化并组合纹理特征
        variance_norm = variance / (np.max(variance) + 1e-8)
        grad_norm = grad_mag / (np.max(grad_mag) + 1e-8)
        texture_score = variance_norm * 0.5 + grad_norm * 0.5

        # 识别低细节区域
        low_detail_threshold = np.percentile(texture_score, 25)
        low_detail_mask = texture_score < low_detail_threshold

        # 形态学操作连接邻近区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
        low_detail_mask = cv2.morphologyEx(low_detail_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        low_detail_mask = cv2.morphologyEx(low_detail_mask, cv2.MORPH_OPEN, kernel)

        # 2. 寻找合适的伪影位置
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(low_detail_mask, connectivity=8)

        # 筛选足够大的区域
        min_area = max(1000, h * w * 0.002)
        valid_regions = []

        for label in range(1, n_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > min_area:
                valid_regions.append({
                    'label': label,
                    'area': area,
                    'bbox': {
                        'x': stats[label, cv2.CC_STAT_LEFT],
                        'y': stats[label, cv2.CC_STAT_TOP],
                        'w': stats[label, cv2.CC_STAT_WIDTH],
                        'h': stats[label, cv2.CC_STAT_HEIGHT]
                    },
                    'center': (int(centroids[label][0]), int(centroids[label][1]))
                })

        if not valid_regions:
            return image

        # 3. 生成伪影 - 减少数量但增大面积
        num_artifacts = min(3, len(valid_regions))
        selected_regions = random.sample(valid_regions, num_artifacts)

        for region in selected_regions:
            bbox = region['bbox']
            center_x, center_y = region['center']

            # 确定伪影尺寸 - 减小尺寸范围
            region_size = max(bbox['w'], bbox['h'])
            min_size = max(100, int(region_size * 0.5))
            max_size = max(min_size + 50, min(300, int(region_size * 1.0)))
            artifact_size = random.randint(min_size, max_size)

            # 计算伪影边界 - 保持偏移范围
            half_size = artifact_size // 2
            max_offset = int(artifact_size * 0.4)
            offset_x = random.randint(-max_offset, max_offset)
            offset_y = random.randint(-max_offset, max_offset)

            center_x = max(half_size, min(w - half_size, center_x + offset_x))
            center_y = max(half_size, min(h - half_size, center_y + offset_y))

            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(w, center_x + half_size)
            y2 = min(h, center_y + half_size)

            # 确保区域有效 - 降低跳过阈值
            if x2 - x1 < 40 or y2 - y1 < 40:
                continue

            # 获取扩展区域 - 解决边界问题
            expand_size = int(artifact_size * 0.2)
            exp_x1 = max(0, x1 - expand_size)
            exp_y1 = max(0, y1 - expand_size)
            exp_x2 = min(w, x2 + expand_size)
            exp_y2 = min(h, y2 + expand_size)

            # 获取扩展后的区域图像
            exp_img = image[exp_y1:exp_y2, exp_x1:exp_x2].copy().astype(np.float32)
            exp_h, exp_w = exp_img.shape

            # 计算原始区域在扩展区域内的位置
            orig_x1 = x1 - exp_x1
            orig_y1 = y1 - exp_y1
            orig_x2 = orig_x1 + (x2 - x1)
            orig_y2 = orig_y1 + (y2 - y1)

            # 4. 生成不规则伪影形状
            y_coords, x_coords = np.mgrid[0:exp_h, 0:exp_w]
            center_x_local = exp_w / 2 + random.uniform(-0.15, 0.15) * exp_w
            center_y_local = exp_h / 2 + random.uniform(-0.15, 0.15) * exp_h

            # 使用更稳定的形状生成方法
            shape_mask = np.zeros((exp_h, exp_w), dtype=np.float32)

            # 基础形状 - 椭圆
            ellipse_ratio = random.uniform(0.7, 1.3)
            ellipse_angle = random.uniform(0, np.pi)
            ellipse_w = exp_w * random.uniform(0.8, 1.2)
            ellipse_h = exp_h * random.uniform(0.8, 1.2)

            # 椭圆距离计算
            cos_angle = np.cos(ellipse_angle)
            sin_angle = np.sin(ellipse_angle)
            dx = (x_coords - center_x_local) * cos_angle - (y_coords - center_y_local) * sin_angle
            dy = (x_coords - center_x_local) * sin_angle + (y_coords - center_y_local) * cos_angle
            ellipse_dist = (dx / (ellipse_w/2))**2 + (dy / (ellipse_h/2))**2

            base_shape = np.clip(1.0 - ellipse_dist, 0, 1)

            # 添加多边形扰动
            num_sides = random.randint(3, 7)
            poly_radius = min(exp_w, exp_h) * random.uniform(0.4, 0.8)
            poly_angle_offset = random.uniform(0, 2*np.pi)

            for i in range(num_sides):
                angle = 2*np.pi * i / num_sides + poly_angle_offset
                next_angle = 2*np.pi * (i+1) / num_sides + poly_angle_offset

                # 计算多边形边的法向量
                edge_angle = (angle + next_angle) / 2
                edge_nx = np.cos(edge_angle)
                edge_ny = np.sin(edge_angle)

                # 计算点到边的距离
                px = x_coords - center_x_local - poly_radius * np.cos(angle)
                py = y_coords - center_y_local - poly_radius * np.sin(angle)
                dist_to_edge = px * edge_nx + py * edge_ny

                # 添加多边形影响
                poly_shape = np.clip(1.0 - dist_to_edge / (poly_radius*0.2), 0, 1)
                base_shape = np.maximum(base_shape, poly_shape * 0.7)

            shape_mask = base_shape.copy()

            # 添加多层噪声扰动
            for octave in range(4):
                frequency = 0.03 * (2 ** octave)
                amplitude = 0.25 / (1.5 ** octave)

                # 生成带旋转的噪声
                angle = random.random() * np.pi
                noise_x = (x_coords * np.cos(angle) + y_coords * np.sin(angle)) * frequency
                noise_y = (-x_coords * np.sin(angle) + y_coords * np.cos(angle)) * frequency

                # 使用稳定的噪声函数
                noise_layer = np.sin(noise_x) * np.cos(noise_y) + 0.5 * np.sin(2*noise_x + random.random())
                shape_mask += noise_layer * amplitude

            # 确保shape_mask没有无效值
            shape_mask = np.nan_to_num(shape_mask)
            shape_mask = np.clip(shape_mask, 0, None)

            # 归一化并应用非线性变换
            min_val = np.min(shape_mask)
            max_val = np.max(shape_mask)
            if max_val - min_val > 1e-8:
                shape_mask = (shape_mask - min_val) / (max_val - min_val + 1e-8)
            else:
                shape_mask = np.zeros_like(shape_mask)

            # 安全地应用幂函数
            exponent = random.uniform(0.6, 1.2)
            shape_mask = np.power(np.clip(shape_mask, 0.01, 1.0), exponent)
            shape_mask = np.clip(shape_mask, 0, 1)

            # 5. 创建颗粒纹理 - 添加强度控制
            base_grain_intensity = 45 + random.randint(-10, 20)
            grain_intensity = base_grain_intensity * grain_strength

            # 多尺度颗粒
            grain_texture = np.zeros((exp_h, exp_w))
            for scale in [1.0, 0.6, 0.3, 0.15]:
                scale_h = max(1, int(exp_h * scale))
                scale_w = max(1, int(exp_w * scale))

                noise_layer = np.random.uniform(-1, 1, (scale_h, scale_w))
                if scale < 1.0:
                    noise_layer = cv2.resize(noise_layer, (exp_w, exp_h), interpolation=cv2.INTER_LINEAR)

                grain_texture += noise_layer * (scale * grain_intensity)

            # 限制颗粒值范围 - 根据强度系数调整上限
            max_grain = 100 * grain_strength
            grain_texture = np.clip(grain_texture, -max_grain, max_grain)

            # 6. 创建深色区域
            local_mean = np.mean(exp_img[orig_y1:orig_y2, orig_x1:orig_x2])

            # 限制变暗强度以避免纯黑
            darkening_amount = min(local_mean * 0.35 + 40, 100)

            # 应用伪影效果
            artifact_effect = shape_mask

            # 变暗处理
            darkened_img = exp_img - (artifact_effect * darkening_amount)
            darkened_img = np.maximum(darkened_img, 0)

            # 添加颗粒纹理
            darkened_img += grain_texture * artifact_effect

            # 7. 边缘羽化和混合
            blend_mask = cv2.GaussianBlur(artifact_effect, (0, 0), sigmaX=8, sigmaY=8)
            blend_mask = np.clip(blend_mask, 0, 1)

            # === 修复边界渐变掩模 ===
            # 创建正确的距离变换掩模，确保内部区域保持完整
            border_mask = np.ones((exp_h, exp_w), dtype=np.float32)
            border_width = min(20, expand_size)

            if border_width > 0:
                # 创建距离图（内部为1，边界渐变到0）
                dist_map = np.ones((exp_h, exp_w))

                # 上边界渐变
                dist_map[:border_width, :] = np.linspace(0, 1, border_width)[:, np.newaxis]
                # 下边界渐变
                dist_map[-border_width:, :] = np.linspace(1, 0, border_width)[:, np.newaxis]
                # 左边界渐变
                left_grad = np.linspace(0, 1, border_width)
                dist_map[:, :border_width] = np.minimum(dist_map[:, :border_width], left_grad)
                # 右边界渐变
                right_grad = np.linspace(1, 0, border_width)
                dist_map[:, -border_width:] = np.minimum(dist_map[:, -border_width:], right_grad)

                border_mask = dist_map

            # 应用边界渐变
            blend_mask *= border_mask

            # 最终混合
            result_region = exp_img * (1 - blend_mask) + darkened_img * blend_mask
            result_region = np.clip(result_region, 0, 255)

            # 安全地转换为uint8
            result_region = result_region.astype(np.uint8)

            # 应用效果到扩展区域
            image[exp_y1:exp_y2, exp_x1:exp_x2] = result_region

        return image

    def disrupt_edges(self, image):
        """破坏物体边缘的连续性（灰度专用）"""
        # 检测边缘
        edges = cv2.Canny(image, 50, 150)

        # 随机选择一些边缘进行破坏
        edge_points = np.argwhere(edges > 0)
        if len(edge_points) > 100:  # 确保有足够的边缘
            # 边缘模糊（有时强度较大）
            blur_amount = random.uniform(*self.edge_blur_range)

            # 使用更大的模糊内核进行更强的边缘破坏
            blurred = cv2.GaussianBlur(image, (0, 0), blur_amount)

            # 创建边缘区域掩模
            edge_mask = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)
            edge_mask = edge_mask.astype(np.float32) / 255

            # 应用边缘模糊
            result = image.copy().astype(np.float32)
            result = result * (1 - edge_mask) + blurred.astype(np.float32) * edge_mask

            # 随机添加边缘不连续
            if random.random() < self.edge_discontinuity_prob:
                h, w = image.shape
                num_disruptions = random.randint(5, 20)
                for _ in range(num_disruptions):
                    # 选择边缘点
                    pt_idx = random.randint(0, len(edge_points)-1)
                    y, x = edge_points[pt_idx]

                    # 创建局部中断（1-3像素的位移）
                    size = random.randint(1, 3)
                    displacement = random.randint(1, 5)
                    direction = random.choice([(-1,0), (1,0), (0,-1), (0,1)])

                    # 移动局部区域
                    for dy in range(-size, size+1):
                        for dx in range(-size, size+1):
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < h and 0 <= nx < w:
                                target_y = min(max(0, ny + direction[0]*displacement), h-1)
                                target_x = min(max(0, nx + direction[1]*displacement), w-1)
                                result[ny, nx] = image[target_y, target_x]

            return np.clip(result, 0, 255).astype(np.uint8)
        return image

    def super_resolution_simulation(self, image,
                              scale_factor: float = 0.5,
                              random_method: bool = True,
                              method: str = None):
        """
        模拟超分辨率场景：先降采样再上采样回原分辨率

        Args:
            image: 输入图像，支持 numpy.ndarray, torch.Tensor, PIL.Image
            scale_factor: 降采样因子，默认0.5（降到1/2分辨率）
            random_method: 是否随机选择上采样方法
            method: 指定上采样方法，当random_method=False时生效
                    可选: 'bilinear', 'bicubic', 'lanczos', 'nearest', 'area'

        Returns:
            与输入相同类型和分辨率的图像
        """

        # 记录输入类型和原始尺寸
        input_type = type(image)
        original_shape = None

        # 转换为numpy数组进行处理
        if isinstance(image, torch.Tensor):
            # 处理torch tensor
            if image.dim() == 4:  # (B, C, H, W)
                batch_size = image.shape[0]
                image_np = image.cpu().numpy()
                original_shape = image.shape
            elif image.dim() == 3:  # (C, H, W) 或 (H, W, C)
                if image.shape[0] <= 4:  # 假设通道在第一维
                    image_np = image.permute(1, 2, 0).cpu().numpy()
                else:  # 通道在最后一维
                    image_np = image.cpu().numpy()
                original_shape = image.shape
            else:  # (H, W)
                image_np = image.cpu().numpy()
                if len(image_np.shape) == 2:
                    image_np = np.expand_dims(image_np, axis=-1)
                original_shape = image.shape

        elif isinstance(image, Image.Image):
            image_np = np.array(image)
            if len(image_np.shape) == 2:
                image_np = np.expand_dims(image_np, axis=-1)

        elif isinstance(image, np.ndarray):
            image_np = image.copy()
            if len(image_np.shape) == 2:
                image_np = np.expand_dims(image_np, axis=-1)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # 确保图像是uint8或float类型
        if image_np.dtype == np.uint8:
            is_uint8 = True
            image_np = image_np.astype(np.float32) / 255.0
        else:
            is_uint8 = False
            if image_np.max() > 1.0:
                image_np = image_np / 255.0

        # 处理批量数据
        if len(image_np.shape) == 4:  # (B, H, W, C) or (B, C, H, W)
            results = []
            for i in range(image_np.shape[0]):
                single_img = image_np[i]
                processed = self._process_single_image(single_img, scale_factor, random_method, method)
                results.append(processed)
            result_np = np.stack(results, axis=0)
        else:
            result_np = self._process_single_image(image_np, scale_factor, random_method, method)

        # 转换回原始数据类型
        if is_uint8:
            result_np = np.clip(result_np * 255.0, 0, 255).astype(np.uint8)
        else:
            result_np = np.clip(result_np, 0, 1).astype(np.float32)

        # 转换回原始类型
        if input_type == torch.Tensor:
            if original_shape is not None:
                if len(original_shape) == 4:  # (B, C, H, W)
                    result = torch.from_numpy(result_np.transpose(0, 3, 1, 2))
                elif len(original_shape) == 3:
                    if original_shape[0] <= 4:  # (C, H, W)
                        if result_np.shape[-1] == 1:
                            result = torch.from_numpy(result_np.squeeze(-1))
                        else:
                            result = torch.from_numpy(result_np.transpose(2, 0, 1))
                    else:  # (H, W, C)
                        result = torch.from_numpy(result_np)
                else:  # (H, W)
                    result = torch.from_numpy(result_np.squeeze(-1))
            else:
                result = torch.from_numpy(result_np)

        elif input_type == Image.Image:
            if result_np.shape[-1] == 1:
                result = Image.fromarray(result_np.squeeze(-1))
            elif result_np.shape[-1] == 3:
                result = Image.fromarray(result_np)
            else:
                result = Image.fromarray(result_np)

        else:  # numpy.ndarray
            if len(image.shape) == 2 and result_np.shape[-1] == 1:
                result = result_np.squeeze(-1)
            else:
                result = result_np

        return result

    def _process_single_image(self, image: np.ndarray,
                            scale_factor: float,
                            random_method: bool,
                            method: str) -> np.ndarray:
        """处理单张图像的内部函数"""

        h, w = image.shape[:2]

        # 计算降采样后的尺寸
        new_h = max(1, int(h * scale_factor))
        new_w = max(1, int(w * scale_factor))

        # 降采样 - 使用area插值避免混叠
        downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 选择上采样方法
        upsampling_methods = {
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
            'nearest': cv2.INTER_NEAREST,
            'area': cv2.INTER_AREA,  # 对于上采样效果类似线性
        }

        if random_method:
            # 根据不同方法的实用性设置权重
            methods = ['bilinear', 'bicubic', 'lanczos', 'nearest']
            weights = [0.3, 0.4, 0.25, 0.05]  # bicubic和lanczos更常用
            selected_method = random.choices(methods, weights=weights)[0]
        else:
            selected_method = method if method is not None else 'bicubic'

        # 上采样回原分辨率
        if selected_method in upsampling_methods:
            upsampled = cv2.resize(downsampled, (w, h),
                                interpolation=upsampling_methods[selected_method])
        else:
            # 默认使用双三次插值
            upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)

        # 添加轻微的噪声模拟真实超分场景（可选）
        if random_method and random.random() < 0.1:  # 10%概率添加轻微噪声
            noise_std = random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_std, upsampled.shape).astype(np.float32)
            upsampled = np.clip(upsampled + noise, 0, 1)

        return upsampled


    def random_degradation(self, image):
        """应用随机组合的退化效果（最终输出为灰度）"""
        # 如果输入是彩色，转换为灰度
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        # 创建退化序列
        augmentations = [
            self.add_adaptive_gaussian_noise,
            self.create_block_artifacts,
            self.disrupt_edges,
            self.add_motion_blur,
            self.super_resolution_simulation,
        ]

        # 随机打乱并应用退化效果
        random.shuffle(augmentations)

        # 应用所有退化效果
        for aug in augmentations:
            # 每个退化有30%的概率被跳过（以增加多样性）
            if random.random() > 0.3:
                image = aug(image)

        return image



# 定义预期的形状
EXPECTED_SHAPES = {
    'lr_img': (3, 512, 512),
    'hr_img': (3, 512, 512),
    'latent_hr': (16, 64, 64),
    'prompt_embeds_input': (333, 4096),
    'pooled_prompt_embeds_input': (2048,)
}
