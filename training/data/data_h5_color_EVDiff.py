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
import glob

from training.utils.event_utils import events_to_voxel_torch
from training.utils.data_augmentation import add_hot_pixels_to_sequence_, add_noise_to_voxel


# Paths to the datasets
'''
H5_FOLDER_ROOT:
    - event
        - temp000000000_out_0.h5
            - events
                - ps
                - ts
                - xs
                - ys
            - images:
                - image000000000
                    attrs['timestamp'])
                ...
            - rgb_images:
                - image000000000
                    attrs['timestamp'])
            属性: duration, 值: 5.856733567
            属性: num_events, 值: 8583428
            属性: num_imgs, 值: 223
            属性: sensor_resolution, 值: [256 256]
            属性: t0, 值: 0.0
            属性: tk, 值: 5.856733567

MEM
    bike_bay_hdr
        events_p.npy
        events_ts.npy
        events_xy.npy
        image_event_indices.npy
        images_ts.npy
        images.npy
    ...

'''

H5_FOLDER_ROOT_1 = os.environ.get("REDS_SMALL1", "data/REDS/small1")
H5_FOLDER_ROOT_2 = os.environ.get("REDS_SMALL2", "data/REDS/small2")
folder_path = [H5_FOLDER_ROOT_1, H5_FOLDER_ROOT_2]
# folder_path = [H5_FOLDER_ROOT_1]

# folder_path = [H5_FOLDER_ROOT_1, H5_FOLDER_ROOT_2, H5_REAL_FOLDER_ROOT]
RANDOM_SKIP_EVENT_RATE = [0.05, 0.95]
DEFECT_INTENSITY = 0.4
MERGE_PREV_PROB = 0.2
# folder_path = [H5_FOLDER_ROOT, H5_KIN_A10, H5_KIN_A10_1, H5_KIN_A10_2, H5_KIN_3090]

class H5_ESIM_Dataset_color_merge(Dataset):
    '''
    每个H5都包含280左右长度的序列，每次截取一部分就行
    支持自定义输出分辨率，小于原始分辨率时进行裁剪
    '''
    # ============ 修改1: __init__ 方法 ============
    def __init__(self, folder_path=folder_path, num_bins=5,
                voxel_method=None, max_length=None, keep_ratio=1, sequence_length=30, device="cpu",
                noise_std=0.1, noise_fraction=1.0, hot_pixel_std=0.1, max_hot_pixel_fraction=0.001,
                skip_head=1, output_resolution=None, return_events=False,
                data_fraction=1.0, data_subset_seed=0):
        """
        新增参数:
            output_resolution: tuple (height, width) 或 None
                - None: 使用原始分辨率
                - (H, W): 指定输出分辨率，小于原始分辨率时随机裁剪
            return_events: bool, 默认False
                - False: 返回voxel (原有行为)
                - True: 返回原始events，不进行voxel化
        """
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

        # 新增：event返回模式
        self.return_events = return_events

        # 新增：输出分辨率设置
        self.output_resolution = output_resolution
        if output_resolution is not None:
            assert len(output_resolution) == 2, "output_resolution must be (height, width)"
            assert output_resolution[0] > 0 and output_resolution[1] > 0, "resolution must be positive"

        # 获取排序后的文件名列表
        folders = ['event']  # 只需要event文件夹
        self.file_paths = []


        for data_path in self.all_folder_path:
            # 遍历每个文件夹，找到所有以.h5结尾的文件
            for fname in sorted(os.listdir(os.path.join(data_path, folders[0]))):
                if not fname.endswith('.h5'):
                    continue
                # 构建文件的完整路径
                path = os.path.join(data_path, folders[0], fname)
                # 验证文件存在
                if os.path.exists(path):
                    self.file_paths.append(path)

        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No valid h5 files found in {data_path}")

        # 获得每个h5中的image的名字，以及每个frame对应的event的索引，二维列表
        self.all_image_names = []
        self.all_event_idxs = []
        self.sensor_resolutions = [] # 不同的h5文件可能分辨率不同，所以需要记录
        remove_list = [] # 记录需要删除的文件
        # 创建一个txt记录排除了哪些文件
        with open("excluded_files.txt", "w") as f:
            for event_h5_path in tqdm(self.file_paths, desc="computing frame indices"):
                tmp_image_names, single_event_idxs, single_sensor_resolution = self.compute_frame_indices(event_h5_path)

                # 新增：检查分辨率是否满足输出要求
                if self.output_resolution is not None:
                    if (single_sensor_resolution[0] < self.output_resolution[0] or
                        single_sensor_resolution[1] < self.output_resolution[1]):
                        print(f"Warning: Skipping file {event_h5_path} - resolution {single_sensor_resolution} "
                            f"is smaller than required {self.output_resolution}")
                        remove_list.append(event_h5_path)
                        f.write(f"{event_h5_path} (resolution too small)\n")
                        continue

                # 检测事件的长度，如果长度小于sequence_length，则跳过该文件，并且将该名字移出self.file_paths
                if len(single_event_idxs) < sequence_length + self.skip_head:
                    print(f"Warning: Skipping file {event_h5_path} with less than sequence_length events")
                    remove_list.append(event_h5_path)
                    f.write(event_h5_path + '\n')
                    continue
                self.all_image_names.append(tmp_image_names)
                self.all_event_idxs.append(single_event_idxs)
                self.sensor_resolutions.append(single_sensor_resolution)

        # 从file_paths删除需要删除的文件
        for item in remove_list:
            self.file_paths.remove(item)

        # ===== Data-scaling subset (EVDiff exp #4): 固定种子嵌套子采样 =====
        # 在完整文件列表建好后、self.length 计算前，按比例保留前 N 条。
        # 同一个 subset_seed 生成同一 perm，故 perm[:N_f] 满足
        # 1% ⊂ 5% ⊂ 10% ⊂ 30% ⊂ 50% ⊂ 100%（嵌套）。按原始下标重排以保持稳定顺序。
        total = len(self.file_paths)
        if data_fraction is not None and data_fraction < 1.0:
            n_keep = max(1, int(data_fraction * total + 0.5))   # 四舍五入
            rng = np.random.default_rng(data_subset_seed)
            perm = rng.permutation(total)
            keep_idx = sorted(perm[:n_keep].tolist())
            for _lst in ("file_paths", "all_image_names", "all_event_idxs", "sensor_resolutions"):
                setattr(self, _lst, [getattr(self, _lst)[i] for i in keep_idx])
        else:
            n_keep = total
        total_frames = sum(len(_n) for _n in self.all_image_names)
        print(f"[data-scaling] fraction={data_fraction} subset_seed={data_subset_seed} "
              f"kept {n_keep}/{total} sequences, {total_frames} event-frame pairs", flush=True)

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



    def _get_crop_params(self, original_resolution, index):
        """
        计算裁剪参数，在序列中保持一致的裁剪位置

        参数:
            original_resolution: (height, width) 原始分辨率
            index: 当前H5文件的索引，用于生成一致的随机种子

        返回:
            crop_params: dict with keys 'top', 'left', 'height', 'width'
        """
        if self.output_resolution is None:
            # 不裁剪，返回全图参数
            return {
                'top': 0,
                'left': 0,
                'height': original_resolution[0],
                'width': original_resolution[1]
            }

        out_h, out_w = self.output_resolution
        orig_h, orig_w = original_resolution

        # 如果输出分辨率大于等于原始分辨率，不裁剪
        if out_h >= orig_h and out_w >= orig_w:
            return {
                'top': 0,
                'left': 0,
                'height': orig_h,
                'width': orig_w
            }

        # 计算裁剪区域（随机裁剪，但在同一个__getitem__调用中保持一致）
        # 使用index作为种子的一部分，确保同一序列使用相同的裁剪位置
        max_top = max(0, orig_h - out_h)
        max_left = max(0, orig_w - out_w)

        # 每次getitem调用时随机选择裁剪位置
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0

        return {
            'top': top,
            'left': left,
            'height': min(out_h, orig_h),
            'width': min(out_w, orig_w)
        }

    # ============ 修改2: __getitem__ 方法 ============
    def __getitem__(self, index):
        '''
        返回一个sequence
        当return_events=False时（默认）:
        [{"lr_img": lr_img,
            "hr_img": hr_img,
            "prompt_text": prompt,
            'events': voxel,  # voxel tensor
            }
        ]

        当return_events=True时:
        [{"lr_img": lr_img,
            "hr_img": hr_img,
            "prompt_text": prompt,
            'events': {'xs': xs, 'ys': ys, 'ts': ts, 'ps': ps},  # 原始events
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
        now_H5_name = self.file_paths[index]
        now_frame_length = len(now_image_name_list)

        # 新增：为这个序列生成统一的裁剪参数
        crop_params = self._get_crop_params(self.sensor_resolutions[index], index)

        idx0 = random.randint(self.skip_head, now_frame_length - self.sequence_length)
        indices = [idx0 + i for i in range(self.sequence_length)]


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

        # 使用adjusted_indices替代原来的indices，传入裁剪参数和return_events标志
        rgb_image_list, gray_image_list, tag_list, events_or_voxel_list = self.get_event_tags_frame(
            now_H5_name, adjusted_indices, index, merge_flags=merge_flags,
            crop_params=crop_params, return_events=self.return_events
        )

        # 构建sequence
        final_sequence = []
        for i in range(self.sequence_length):
            tmp_sequence = {
                "gray_img": gray_image_list[i],
                "hr_img": rgb_image_list[i],
                # "prompt_text": tag_list[i],
                'events': events_or_voxel_list[i],  # 根据模式，这里是voxel或events dict
                "path": now_H5_name,
            }
            final_sequence.append(tmp_sequence)

        # 加入hot pixel (仅在voxel模式下)
        if not self.return_events and self.hot_pixel_std is not None:
            add_hot_pixels_to_sequence_(final_sequence, hot_pixel_std=self.hot_pixel_std,
                                        max_hot_pixel_fraction=self.max_hot_pixel_fraction)

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
        '''
        single_event_idxs = []
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

    def get_empty_voxel_grid(self, index, crop_params=None):
        """
        Return an empty voxel grid filled with zeros
        新增：支持裁剪后的分辨率
        """
        if crop_params is not None:
            size = (self.num_bins, crop_params['height'], crop_params['width'])
        else:
            size = (self.num_bins, *self.sensor_resolutions[index])
        return torch.zeros(size, dtype=torch.float32)

    def get_voxel_grid(self, xs, ys, ts, ps, single_sensor_resolution, crop_params=None):
        """
        Given events, return voxel grid
        新增：支持事件裁剪

        :param xs: tensor containg x coords of events
        :param ys: tensor containg y coords of events
        :param ts: tensor containg t coords of events
        :param ps: tensor containg p coords of events
        :param crop_params: 裁剪参数 dict
        create voxel grid merging positive and negative events (resulting in NUM_BINS x H x W tensor).
        """
        if crop_params is not None:
            # 裁剪事件：只保留在裁剪区域内的事件
            left, top = crop_params['left'], crop_params['top']
            right = left + crop_params['width']
            bottom = top + crop_params['height']

            # 创建掩码
            mask = (xs >= left) & (xs < right) & (ys >= top) & (ys < bottom)

            # 过滤事件
            xs_crop = xs[mask] - left  # 调整坐标到裁剪区域
            ys_crop = ys[mask] - top
            ts_crop = ts[mask]
            ps_crop = ps[mask]

            # 使用裁剪后的分辨率
            crop_resolution = (crop_params['height'], crop_params['width'])
            voxel_grid = events_to_voxel_torch(xs_crop, ys_crop, ts_crop, ps_crop,
                                              self.num_bins, sensor_size=crop_resolution)
        else:
            # 不裁剪
            voxel_grid = events_to_voxel_torch(xs, ys, ts, ps, self.num_bins,
                                              sensor_size=single_sensor_resolution)

        return voxel_grid

    # ============ 修改3: get_event_tags_frame 方法 ============
    def get_event_tags_frame(self, event_h5_name, indices, index, merge_flags=None,
                            crop_params=None, return_events=False):
        """
        根据提供的索引列表获取数据，支持事件合并和裁剪

        参数:
            event_h5_name: H5文件路径
            indices: 要获取的帧索引列表
            index: 当前H5文件在dataset中的索引
            merge_flags: 布尔列表，标记哪些位置需要与下一帧合并
            crop_params: 裁剪参数 dict，包含 'top', 'left', 'height', 'width'
            return_events: bool, 如果为True则返回原始events而非voxel

        返回:
            rgb_images, gray_images, tags, events_or_voxel_list
            - events_or_voxel_list: 当return_events=True时为events dict列表，否则为voxel列表
        """
        if merge_flags is None:
            merge_flags = [False] * len(indices)

        rgb_images = []
        gray_images = []
        tags = []
        events_or_voxel_list = []

        with h5py.File(event_h5_name, 'r') as f:
            # 获取图像和标签
            for img_idx in indices:
                now_image_name = self.all_image_names[index][img_idx]

                gray_numpy = f['images'][now_image_name][:, :]
                rgb_numpy = f['rgb_images'][now_image_name][:, :, :]

                # 应用裁剪到图像
                if crop_params is not None:
                    top = crop_params['top']
                    left = crop_params['left']
                    height = crop_params['height']
                    width = crop_params['width']

                    gray_numpy = gray_numpy[top:top+height, left:left+width]
                    rgb_numpy = rgb_numpy[top:top+height, left:left+width, :]
                    # gray_numpy = gray_numpy[top:top-height, left:left-width]
                    # rgb_numpy = rgb_numpy[top:top-height, left:left-width, :]

                gray_pil = Image.fromarray(gray_numpy).convert('RGB')
                rgb_pil = Image.fromarray(rgb_numpy).convert('RGB')

                gray_img = self.trans(gray_pil).squeeze() * 2 - 1
                rgb_img = self.trans(rgb_pil).squeeze() * 2 - 1

                gray_images.append(gray_img)
                rgb_images.append(rgb_img)
                tags.append(f['tags'][now_image_name][()])

            # 处理事件
            xs = f['events']['xs'][:].astype(np.float32)
            ys = f['events']['ys'][:].astype(np.float32)
            ts = f['events']['ts'][:].astype(np.float32)
            ps = f['events']['ps'][:].astype(np.float32)

            rng = np.random.default_rng()
            u = rng.random()

            # 事件处理循环，支持合并
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

                # 跳过事件处理
                low, high = self.random_skip_events_list
                exponent_factor = 1.4
                skewed_u = u ** (1/exponent_factor)
                random_rate = low + (high - low) * skewed_u
                event_xs, event_ys, event_ts, event_ps = self.random_skip_events(
                    event_xs, event_ys, event_ts, event_ps, random_rate=random_rate
                )

                # 传感器缺陷模拟
                event_xs, event_ys, event_ts, event_ps = self.simulate_sensor_defect(
                    event_xs, event_ys, event_ts, event_ps,
                    width=self.sensor_resolutions[index][1],
                    height=self.sensor_resolutions[index][0],
                    defect_intensity=self.defect_intensity
                )

                # 应用裁剪到事件坐标（如果需要）
                if crop_params is not None:
                    left, top = crop_params['left'], crop_params['top']
                    right = left + crop_params['width']
                    bottom = top + crop_params['height']

                    # 创建掩码：只保留裁剪区域内的事件
                    mask = (event_xs >= left) & (event_xs < right) & (event_ys >= top) & (event_ys < bottom)

                    event_xs = event_xs[mask] - left  # 调整坐标
                    event_ys = event_ys[mask] - top
                    event_ts = event_ts[mask]
                    event_ps = event_ps[mask]

                # 根据return_events模式返回不同的数据
                if return_events:
                    # 返回原始events（归一化时间戳）
                    if len(event_xs) > 0:
                        ts_0 = event_ts[0]
                        event_ts_normalized = event_ts - ts_0
                        event_ps_normalized = event_ps * 2.0 - 1.0
                    else:
                        ts_0 = 0
                        event_ts_normalized = event_ts
                        event_ps_normalized = event_ps * 2.0 - 1.0

                    events_dict = {
                        'xs': torch.tensor(event_xs, dtype=torch.float32),
                        'ys': torch.tensor(event_ys, dtype=torch.float32),
                        'ts': torch.tensor(event_ts_normalized, dtype=torch.float32),
                        'ps': torch.tensor(event_ps_normalized, dtype=torch.float32),
                        'ts_offset': ts_0  # 保存时间偏移量，方便恢复
                    }
                    events_or_voxel_list.append(events_dict)
                else:
                    if crop_params is not None:
                        sensor_size = (crop_params['height'], crop_params['width'])
                    else:
                        sensor_size = self.sensor_resolutions[index]
                    # 生成voxel（原有逻辑）
                    event_xs_tensor = torch.tensor(event_xs, dtype=torch.float32)
                    if event_xs_tensor.nelement() == 0:
                        tmp_voxel = self.get_empty_voxel_grid(index=index, crop_params=crop_params)
                    else:
                        ts_0 = event_ts[0]
                        event_ys_tensor = torch.tensor(event_ys, dtype=torch.float32)
                        event_ts_tensor = torch.tensor(event_ts, dtype=torch.float32) - ts_0
                        event_ps_tensor = torch.tensor(event_ps, dtype=torch.float32) * 2.0 - 1.0
                        tmp_voxel = self.get_voxel_grid(
                            event_xs_tensor, event_ys_tensor, event_ts_tensor, event_ps_tensor,
                            # single_sensor_resolution=self.sensor_resolutions[index],
                            single_sensor_resolution=sensor_size,
                            # crop_params=crop_params
                            crop_params=None, # 前面已经裁剪过一次了
                        )

                    if self.noise_std is not None:
                        tmp_voxel = add_noise_to_voxel(
                            tmp_voxel,
                            noise_std=self.noise_std,
                            noise_fraction=self.noise_fraction
                        )

                    events_or_voxel_list.append(tmp_voxel)

        return rgb_images, gray_images, tags, events_or_voxel_list



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
        if defect_mode in [1, 3]:  # 正事件缺陷
            pos_defects = self._create_correlated_random((grid_size_y, grid_size_x))
            pos_scaler = np.random.uniform(0.5, 1.0)  # 随机强度调节器

        if defect_mode in [2, 3]:  # 负事件缺陷
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





# ============ 测试代码 ============
