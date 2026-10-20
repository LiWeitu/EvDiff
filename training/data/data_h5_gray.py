import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import warnings
import h5py
import random

from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from bisect import bisect_left
from diffusers import AutoencoderKL

from training.utils.event_utils import events_to_voxel_torch
from training.utils.util import read_json
from training.utils.data_augmentation import add_hot_pixels_to_sequence_, add_noise_to_voxel


# Paths to the datasets
'''
H5_FOLDER_ROOT:
    - embed
        - temp000000000_out_0.h5
            - latent_hr
                - image000000000
                ...
            - pool_embeds
                - image000000000
                ...
            - prompt_embeds
                - image000000000
                ...
    - event
        - temp000000000_out_0.h5
            - events
                - ps
                - ts
                - xs
                - ys
            - images:
                attrs['timestamp'])
                - image000000000
                ...
            - rgb_images:
                attrs['timestamp'])
                - image000000000
            属性: duration, 值: 5.856733567
            属性: num_events, 值: 8583428
            属性: num_imgs, 值: 223
            属性: sensor_resolution, 值: [256 256]
            属性: t0, 值: 0.0
            属性: tk, 值: 5.856733567


'''
SD3_PATH = 'checkpoints/sd3-medium'
H5_FOLDER_ROOT = os.environ.get("REDS_SMALL1", "data/REDS/small1")
H5_FOLDER_ROOT2 = os.environ.get("REDS_SMALL2", "data/REDS/small2")
RANDOM_SKIP_EVENT_RATE = [0.05, 0.95]
DEFECT_INTENSITY = 0.5
MERGE_PREV_PROB = 0.2
# data_path = H5_FOLDER_ROOT
# folder_path = [H5_FOLDER_ROOT]
folder_path = [H5_FOLDER_ROOT,H5_FOLDER_ROOT2]
# folder_path = [H5_FOLDER_ROOT, H5_FOLDER_ROOT2, EVK4_FOLDER_ROOT_1]
# folder_path = [H5_FOLDER_ROOT, H5_KIN_A10, H5_KIN_A10_1, H5_KIN_A10_2, H5_KIN_3090]


class H5_ESIM_Dataset_gray_merge(Dataset):
    '''
    不返回latent，只返回event和图片
    '''
    def __init__(self, folder_path=folder_path, num_bins=5,
                 voxel_method=None, max_length=None, keep_ratio=1, sequence_length=30, device="cuda",
                 noise_std=0.1, noise_fraction=1.0, hot_pixel_std=0.1, max_hot_pixel_fraction=0.001, skip_head=5):
        # TODO:记得加入hot mask，在EvREC的utils里面
        self.trans = transforms.Compose([
                transforms.ToTensor(),  # 将图片转换为Tensor，并自动归一化到0-1
                # 你可以在这里添加其他转换操作，如调整大小、裁剪等
            ])
        self.num_bins = num_bins
        # 检查data_path是否存在
        for folder in folder_path:
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Data path {folder} does not exist.")
        self.all_folder_path = folder_path
        self.keep_ratio = keep_ratio
        self.has_images = True
        self.channels = self.num_bins
        self.sequence_length = sequence_length
        self.skip_head = skip_head
        self.merge_prev_prob = MERGE_PREV_PROB

        self.noise_std = noise_std
        self.noise_fraction = noise_fraction
        self.hot_pixel_std = hot_pixel_std
        self.max_hot_pixel_fraction = max_hot_pixel_fraction
        # 获取排序后的文件名列表（确保三个文件夹的文件一一对应）
        folders = ['event',]
        self.file_paths = []


        for data_path in self.all_folder_path:
            # 遍历每个文件夹，找到所有以.h5结尾的文件
            for fname in sorted(os.listdir(os.path.join(data_path, folders[0]))):
                if not fname.endswith('.h5'):
                    continue
                # 构建两个文件的完整路径
                paths = [os.path.join(data_path, folder, fname) for folder in folders]
                # 验证所有文件都存在
                if all(os.path.exists(p) for p in paths):
                    self.file_paths.append(tuple(paths))

        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No valid h5 file pairs found in {data_path}")

        # 获得每个h5中的image的名字，以及每个frame对应的event的索引，二维列表
        self.all_image_names = []
        self.all_event_idxs = []
        self.sensor_resolutions = [] # 不同的h5文件可能分辨率不同，所以需要记录
        remove_list = [] # 记录需要删除的文件
        # 创建一个txt记录排除了哪些文件
        with open("excluded_files.txt", "w") as f:
            print("computing frame indices...")# 打印在.log里面的内容不适合进度条
            for event_h5_path in tqdm(self.file_paths, desc="computing frame indices"):
            # for event_h5_path in self.file_paths:
                tmp_image_names, single_event_idxs, single_sensor_resolution = self.compute_frame_indices(event_h5_path[0])
                # print("single_sensor_resolution:", single_sensor_resolution)
                # 检测事件的长度，如果长度小于sequence_length，则跳过该文件，并且将该名字移出self.file_paths
                if len(single_event_idxs) < sequence_length + self.skip_head:
                    print(f"Warning: Skipping file {event_h5_path} with less than sequence_length events")
                    # self.file_paths.remove(event_h5_path)
                    remove_list.append(event_h5_path)
                    f.write(event_h5_path[0] + '\n')
                    continue
                self.all_image_names.append(tmp_image_names)
                self.all_event_idxs.append(single_event_idxs)
                self.sensor_resolutions.append(single_sensor_resolution)

        # 从file_paths删除需要删除的文件
        for item in remove_list:
            self.file_paths.remove(item)

        self.length = len(self.file_paths)
        print("number of img names list:", len(self.all_image_names))
        print("number of event idxs list:", len(self.all_event_idxs))
        print("self.length:",self.length)
        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.voxel_method = voxel_method
        self.set_voxel_method()

        if max_length is not None:
            self.length = min(self.length, max_length + 1)

        # 跳过部分事件
        self.random_skip_events_list = RANDOM_SKIP_EVENT_RATE
        self.defect_intensity = DEFECT_INTENSITY

    def __getitem__(self, index):
        '''
        返回一个sequence
        [{"gray_img": gray_img,
            "latent_hr": latent_hr,
            'events': voxel,
            }
        ]
        '''
        # 选择第index的序列，在这个序列中随机选取一个1到总长度-序列长度之间的帧作为起始点，连续返回一个列表
        if self.voxel_method['method'] == 'between_frames':
            assert 0 <= index < self.__len__(), f"index {index} out of bounds (0 <= x < {self.__len__()})"
        else:
            # TODO: 还没写
            assert 0 <= index < self.__len__(), f"index {index} out of bounds (0 <= x < {self.__len__()})"
            idx0, idx1 = self.get_event_indices(index)

        # 获取起始帧索引
        now_image_name_list = self.all_image_names[index]
        now_H5_name_list = self.file_paths[index]
        now_frame_length = len(now_image_name_list)

        idx0 = random.randint(self.skip_head, now_frame_length - self.sequence_length)
        indices = [idx0 + i for i in range(self.sequence_length)]

        ## ============ 修改：仅在H5路径包含"REDS"时启用合并逻辑 ============
        # 检查当前H5路径是否包含"REDS"
        enable_merge = "REDS" in now_H5_name_list[0]

        if enable_merge:
            # 确保我们有足够的空间进行合并（最多可能需要额外的帧）
            max_possible_idx = max(indices) + self.sequence_length  # 最坏情况：所有帧都需要合并
            if max_possible_idx >= now_frame_length:
                # 如果空间不够，调整起始位置
                idx0 = random.randint(self.skip_head, now_frame_length - 2 * self.sequence_length)
                indices = [idx0 + i for i in range(self.sequence_length)]

            # 记录哪些位置需要合并
            merge_flags = []
            for i in range(self.sequence_length):
                # 检查是否触发合并，且后面还有足够的帧
                should_merge = (random.random() < self.merge_prev_prob and
                            indices[i] + 1 < now_frame_length)
                merge_flags.append(should_merge)

            # 根据合并标志调整indices
            adjusted_indices = []
            offset = 0
            for i in range(self.sequence_length):
                current_idx = indices[i] + offset
                # 边界检查
                if current_idx >= now_frame_length:
                    current_idx = now_frame_length - 1
                adjusted_indices.append(current_idx)

                if merge_flags[i]:
                    offset += 1  # 下一个索引需要向后偏移
        else:
            # 不启用合并时直接使用原始索引
            adjusted_indices = indices
            merge_flags = [False] * self.sequence_length  # 没有合并
        # ============ 结束修改部分 ============

        gray_image_list, voxel_list = self.get_event_frame(
            now_H5_name_list[0], adjusted_indices, index, merge_flags=merge_flags
        )

        # 构建sequence，在组装sequence的时候转换voxel
        final_sequence = []
        # 构建sequence，在组装sequence的时候转换voxel
        for i in range(self.sequence_length):
            # idx0一定不为0，由最前面的randint决定
            tmp_sequence = {
                "gray_img": gray_image_list[i],
                'events': voxel_list[i],
                "path": now_H5_name_list[0],
            }
            final_sequence.append(tmp_sequence)

        # 加入hot pixel
        if self.hot_pixel_std is not None:
            add_hot_pixels_to_sequence_(final_sequence, hot_pixel_std=self.hot_pixel_std, max_hot_pixel_fraction=self.max_hot_pixel_fraction)

        return final_sequence

    def compute_timeblock_indices(self):
        """
        For each block of time (using t_events), find the start and
        end indices of the corresponding events
        """
        timeblock_indices = []
        start_idx = 0
        for i in range(len(self)):
            start_time = ((self.voxel_method['t'] - self.voxel_method['sliding_window_t']) * i) + self.t0
            end_time = start_time + self.voxel_method['t']
            end_idx = self.find_ts_index(end_time)
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return timeblock_indices

    def compute_frame_indices(self, single_event_name):
        '''
        处理输入的单个event中的h5，得到每个image对应的event的索引，每个image都有标签timestamp
        返回两个列表：image的所有名字的列表，以及每个image对应的event的索引
        获得sensor_resolution

        修改说明：
        - 以读写模式(r+)打开文件，以便修改标签
        - 当计算出的event index和标签值不一致时更新标签
        - 保留原始打印信息用于调试
        '''
        single_event_idxs = []
        # with h5py.File(single_event_name, 'r+') as f:  # 改为读写模式
        with h5py.File(single_event_name, 'r') as f:
            event_ts = f['events']['ts']
            all_images = f['images']
            tmp_image_names = list(f['images'].keys())

            single_sensor_resolution = [all_images[tmp_image_names[0]].shape[0], all_images[tmp_image_names[0]].shape[1]]

            for image_name in tmp_image_names:
                stored_idx = all_images[image_name].attrs['event_idx']
                single_event_idxs.append(stored_idx)

        return tmp_image_names, single_event_idxs, single_sensor_resolution

    def get_min_max_t(self):
        if self.has_images:
            min_t = min(self.frame_ts[0], self.t0)
            max_t = max(self.frame_ts[-1], self.tk)
        else:
            min_t = self.t0
            max_t = self.tk
        return min_t, max_t

    def set_voxel_method(self):
        """
        Given the desired method of computing voxels,
        compute the event_indices lookup table and dataset length
        """
        # TODO: 把其他的也改了，目前只改了between frames
        if self.voxel_method['method'] == 'between_frames':
            assert self.has_images, "Cannot use between_frames voxel method without images"
            # self.length = self.num_frames - 1# init里面有
            # self.event_indices = self.compute_frame_indices()# init里面弄过了
            # self.choose_frames_to_use()# 目前被废弃，里面什么都没有
        else:
            raise ValueError("Invalid voxel forming method chosen ({})".format(self.voxel_method))

    def __len__(self):
        return self.length

    def get_empty_voxel_grid(self, index):
        """Return an empty voxel grid filled with zeros"""
        size = (self.num_bins, *self.sensor_resolutions[index])
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps, single_sensor_resolution):
        """
        Given events, return voxel grid
        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        create voxel grid merging positive and negative events (resulting in NUM_BINS x H x W tensor).
        """
        # generate voxel grid which has size self.num_bins x H x W
        voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins, sensor_size=single_sensor_resolution)
        return voxel_grid

    def get_event_frame(self, event_h5_name, indices, index, merge_flags=None):
        """
        根据提供的索引列表获取数据，而不是起止索引
        indices: 要获取的帧索引列表
        """
        if merge_flags is None:
            merge_flags = [False] * len(indices)
        gray_images = []
        voxel_list = []

        with h5py.File(event_h5_name, 'r') as f:
            # 获取图像和标签 - 直接遍历提供的索引列表
            for img_idx in indices:
                now_image_name = self.all_image_names[index][img_idx]

                # 从HDF5读取numpy数组
                gray_numpy = f['images'][now_image_name][:, :]
                gray_pil = Image.fromarray(gray_numpy).convert('RGB')
                gray_img = self.trans(gray_pil).squeeze() * 2 - 1
                gray_images.append(gray_img)

            # 处理事件 - 确保数据类型一致
            xs = f['events']['xs'][:].astype(np.float32)  # 确保为float32
            ys = f['events']['ys'][:].astype(np.float32)  # 确保为float32
            ts = f['events']['ts'][:].astype(np.float32)  # 确保为float32
            ps = f['events']['ps'][:].astype(np.float32)  # 确保为float32

            rng = np.random.default_rng()
            u = rng.random()
            # 为每一帧获取事件段
            # ============ 修改：事件处理循环，支持合并 ============
            for i in range(len(indices)):
                if i == 0:
                    prev_idx = max(0, indices[0] - 1)
                else:
                    prev_idx = indices[i-1]

                curr_idx = indices[i]

                # 如果当前位置需要合并，且后面还有帧
                if merge_flags[i] and curr_idx + 1 < len(self.all_event_idxs[index]):
                    # 事件范围扩展到下一帧
                    event_idx_prev = self.all_event_idxs[index][prev_idx]
                    event_idx_end = self.all_event_idxs[index][curr_idx + 1]  # 使用下一帧的结束位置
                else:
                    # 正常情况
                    event_idx_prev = self.all_event_idxs[index][prev_idx]
                    event_idx_end = self.all_event_idxs[index][curr_idx]

                event_xs = xs[event_idx_prev:event_idx_end]
                event_ys = ys[event_idx_prev:event_idx_end]
                event_ts = ts[event_idx_prev:event_idx_end]
                event_ps = ps[event_idx_prev:event_idx_end]
                # ============ 结束修改部分 ============

                if self.random_skip_events_list:
                    # 独立的numpy随机数生成，似乎只有这里在用，保证同一个序列都丢弃同样比例的，seed同一个序列里是一样的，不同序列会变化

                    # 使用指数变换使高丢弃率更可能
                    low, high = self.random_skip_events_list
                    # 指数变换 - 使用指数函数增加高值区域的概率密度，因为HQF，ECD这些真实数据集事件数量相比完美的模拟器产生的事件数量更少
                    # 指数因子 > 1 时，随机值更偏向高值区域
                    exponent_factor = 1.4  # 值越大，越偏向高丢弃率  1.4可能有点小，但是可以正常工作
                    # 应用指数变换并映射到原始范围
                    skewed_u = u ** (1/exponent_factor)  # 变换为偏向高值分布
                    random_rate = low + (high - low) * skewed_u
                    # random_rate = rng.uniform(low=self.random_skip_events_list[0], high=self.random_skip_events_list[1])
                    event_xs, event_ys, event_ts, event_ps = self.random_skip_events(event_xs, event_ys, event_ts, event_ps, random_rate=random_rate)

                if self.defect_intensity:
                    # 新加的随机丢弃正或者负的事件
                    event_xs, event_ys, event_ts, event_ps = self.simulate_sensor_defect(event_xs, event_ys, event_ts, event_ps, width=self.sensor_resolutions[index][1], height=self.sensor_resolutions[index][0], defect_intensity=self.defect_intensity)

                event_xs = torch.tensor(event_xs, dtype=torch.float32)
                # 生成voxel
                if event_xs.nelement() == 0:
                    tmp_voxel = self.get_empty_voxel_grid(index=index)
                else:
                    ts_0 = event_ts[0]
                    # 获取事件数据并确保数据类型一致
                    # event_xs = torch.tensor(event_xs, dtype=torch.float32)
                    event_ys = torch.tensor(event_ys, dtype=torch.float32)
                    event_ts = torch.tensor(event_ts, dtype=torch.float32) - ts_0
                    event_ps = torch.tensor(event_ps, dtype=torch.float32) * 2.0 - 1.0
                    tmp_voxel = self.get_voxel_grid(
                        event_xs, event_ys, event_ts, event_ps,
                        single_sensor_resolution=self.sensor_resolutions[index]
                    )

                # 添加噪声
                if self.noise_std is not None:
                    tmp_voxel = add_noise_to_voxel(
                        tmp_voxel,
                        noise_std=self.noise_std,
                        noise_fraction=self.noise_fraction
                    )

                voxel_list.append(tmp_voxel)

        return gray_images, voxel_list

    def find_ts_index(self, timestamp, event_ts):
        index = np.searchsorted(event_ts, timestamp)
        return index

    def random_skip_events(self, xs, ys, ts, ps, random_rate, perturbation_intensity=0.1):
        """
        根据输入的比例范围随机跳过事件，增加概率扰动模拟真实带宽限制

        参数:
        xs, ys, ts, ps: 输入的对应数据数组
        random_rate: 基础跳过比例 (0-1)
        perturbation_intensity: 扰动强度 (0-1)，控制扰动范围 (default=0.1)
        temporal_variation: 是否随时间变化扰动 (default=False)

        返回:
        保留的数据元组 (xs_kept, ys_kept, ts_kept, ps_kept)
        """
        # 如果random_rate为None或0，直接返回原数据
        if random_rate is None or random_rate <= 0:
            return xs, ys, ts, ps

        # 确保所有输入数组长度相同
        assert len(xs) == len(ys) == len(ts) == len(ps), "所有输入数组长度必须相同"

        total_points = len(xs)

        # 1. 生成基础跳过概率（考虑扰动）
        actual_skip_rate = random_rate

        # 添加扰动：在基础值附近波动
        if perturbation_intensity > 0:
            # 计算扰动范围 (确保在[0,1]范围内)
            low_bound = max(0, random_rate - perturbation_intensity)
            high_bound = min(1, random_rate + perturbation_intensity)

            # 在扰动范围内随机选择实际跳过率
            actual_skip_rate = np.random.uniform(low_bound, high_bound)

        # 3. 静态扰动处理（无时间变化）
        # 计算要保留的数据点数量
        keep_count = int(total_points * (1 - actual_skip_rate))
        keep_count = max(1, keep_count)  # 至少保留1个点

        # 随机选择要保留的索引
        indices = np.arange(total_points)
        np.random.shuffle(indices)
        keep_indices = np.sort(indices[:keep_count])  # 保持原始顺序

        # 按照索引保留对应的数据点
        xs_kept = xs[keep_indices]
        ys_kept = ys[keep_indices]
        ts_kept = ts[keep_indices]
        ps_kept = ps[keep_indices]

        return xs_kept, ys_kept, ts_kept, ps_kept

    def simulate_sensor_defect(self, xs, ys, ts, ps, width, height,
                          defect_intensity=0.5, grid_variation=3):
        """
        模拟传感器缺陷，在特定区域随机丢弃正/负事件

        参数:
            xs, ys, ts, ps: 事件数据 (numpy数组)
            width, height: 传感器分辨率
            defect_intensity: 最大丢弃概率 (0-1)，在缺陷最严重的区域会应用此概率
            grid_variation: 网格大小变化幅度 (数值越大变化越大)

        返回:
            处理后的数据元组 (xs_kept, ys_kept, ts_kept, ps_kept)
        """
        # 参数校验
        assert len(xs) == len(ys) == len(ts) == len(ps), "输入数组长度必须相同"
        if defect_intensity <= 0 or len(xs) == 0:
            return xs, ys, ts, ps

        width = int(width)
        height = int(height)

        # 1. 随机生成缺陷模式 (0-无缺陷, 1-仅正事件缺陷, 2-仅负事件缺陷, 3-两者都有)
        defect_mode = np.random.choice([0, 1, 2, 3], p=[0.5, 0.24, 0.24, 0.02])

        # 如果没有启用任何缺陷
        if defect_mode == 0:
            return xs, ys, ts, ps

        # 2. 随机生成网格大小
        base_grid_size = int(max(4, min(width // 20, height // 20)))
        # base_grid_size = max(4, min(width//10, height//10))  # 基础网格大小

        # 确保生成的是整数而不是元组
        grid_size_x = int(max(2, base_grid_size + np.random.randint(-grid_variation, grid_variation+1)))
        grid_size_y = int(max(2, base_grid_size + np.random.randint(-grid_variation, grid_variation+1)))

        # 3. 创建网格划分
        x_edges = np.linspace(0, width, grid_size_x + 1).astype(int)
        y_edges = np.linspace(0, height, grid_size_y + 1).astype(int)

        # 4. 创建选择掩码 (初始全部保留)
        keep_mask = np.ones(len(xs), dtype=bool)

        # 5. 计算每个事件的网格索引
        x_idx = np.clip(np.digitize(xs, x_edges) - 1, 0, grid_size_x-1)
        y_idx = np.clip(np.digitize(ys, y_edges) - 1, 0, grid_size_y-1)

        # 6. 为每个网格单元生成随机丢弃因子
        # 使用简单的随机数而不是高斯随机场
        if defect_mode in [1, 3]:  # 正事件缺陷
            # pos_defects = np.random.random((grid_size_y, grid_size_x))
            pos_defects = self._create_correlated_random((grid_size_y, grid_size_x))
            pos_scaler = np.random.uniform(0.5, 1.0)  # 随机强度调节器

        if defect_mode in [2, 3]:  # 负事件缺陷
            # neg_defects = np.random.random((grid_size_y, grid_size_x))
            neg_defects = self._create_correlated_random((grid_size_y, grid_size_x))
            neg_scaler = np.random.uniform(0.5, 1.0)  # 随机强度调节器

        # 7. 处理正事件
        if defect_mode in [1, 3]:
            pos_mask = (ps == 1)
            if np.any(pos_mask):
                # 为每个正事件获取对应的网格缺陷值
                indices = (y_idx[pos_mask], x_idx[pos_mask])
                pos_values = pos_defects[indices]

                # 应用丢弃概率 (乘以强度调节器)
                drop_probs = defect_intensity * pos_scaler * pos_values

                # 应用丢弃
                rand_vals = np.random.rand(np.sum(pos_mask))
                drop_positions = rand_vals < drop_probs

                # 获取原始索引
                pos_indices = np.where(pos_mask)[0]
                drop_indices = pos_indices[drop_positions]
                keep_mask[drop_indices] = False

        # 8. 处理负事件
        if defect_mode in [2, 3]:
            neg_mask = (ps == -1) | (ps == 0)
            if np.any(neg_mask):
                # 为每个负事件获取对应的网格缺陷值
                indices = (y_idx[neg_mask], x_idx[neg_mask])
                neg_values = neg_defects[indices]

                # 应用丢弃概率 (乘以强度调节器)
                drop_probs = defect_intensity * neg_scaler * neg_values

                # 应用丢弃
                rand_vals = np.random.rand(np.sum(neg_mask))
                drop_positions = rand_vals < drop_probs

                # 获取原始索引
                neg_indices = np.where(neg_mask)[0]
                drop_indices = neg_indices[drop_positions]
                keep_mask[drop_indices] = False

        # 9. 应用最终掩码
        kept = np.where(keep_mask)
        return xs[kept], ys[kept], ts[kept], ps[kept]

    def _create_correlated_random(self, shape, correlation=0.7):
        """
        创建空间相关的随机图（高斯随机场）。

        参数:
            shape: 网格形状 (height, width)
            correlation: 空间相关性强度 (0-1)

        返回:
            空间相关的随机图 (0-1范围)
        """
        # 生成白噪声
        noise = np.random.randn(shape[0], shape[1])

        # 应用高斯滤波创建空间相关性
        kernel_size = int(max(shape) * (1 - correlation))
        kernel_size = max(3, kernel_size)

        if kernel_size > 1:
            # 创建高斯滤波器
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(noise, sigma=kernel_size/3)
        else:
            smoothed = noise

        # 归一化到0-1范围
        min_val = smoothed.min()
        max_val = smoothed.max()
        if max_val - min_val > 1e-6:
            return (smoothed - min_val) / (max_val - min_val)
        else:
            return np.zeros(shape)
