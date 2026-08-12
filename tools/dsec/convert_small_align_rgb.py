import os
import numpy as np
import h5py
import hdf5plugin
import cv2
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as Rot
import json
import tempfile
import shutil


class Transform:
    def __init__(self, translation: np.ndarray, rotation: Rot):
        if translation.ndim > 1:
            self._translation = translation.flatten()
        else:
            self._translation = translation
        assert self._translation.size == 3
        self._rotation = rotation

    @staticmethod
    def from_transform_matrix(transform_matrix: np.ndarray):
        translation = transform_matrix[:3, 3]
        rotation = Rot.from_matrix(transform_matrix[:3, :3])
        return Transform(translation, rotation)

    @staticmethod
    def from_rotation(rotation: Rot):
        return Transform(np.zeros(3), rotation)

    def R_matrix(self):
        return self._rotation.as_matrix()

    def R(self):
        return self._rotation

    def t(self):
        return self._translation

    def T_matrix(self) -> np.ndarray:
        return self._T_matrix_from_tR(self._translation, self._rotation.as_matrix())

    def q(self):
        return self._rotation.as_quat()

    def euler(self):
        return self._rotation.as_euler('xyz', degrees=True)

    def __matmul__(self, other):
        rotation = self._rotation * other._rotation
        translation = self._rotation.apply(other._translation) + self._translation
        return Transform(translation, rotation)

    def inverse(self):
        rotation = self._rotation.inv()
        translation = - rotation.apply(self._translation)
        return Transform(translation, rotation)


class ChunkedArrayWriter:
    """高效的分块数组写入器，避免重复加载"""
    
    def __init__(self, output_path, array_name, dtype=None):
        self.output_path = Path(output_path)
        self.array_name = array_name
        self.temp_dir = self.output_path / f"_temp_{array_name}"
        self.temp_dir.mkdir(exist_ok=True)
        self.chunk_files = []
        self.dtype = dtype
        self.total_size = 0
    
    def append_chunk(self, data):
        """保存一个数据块"""
        chunk_idx = len(self.chunk_files)
        chunk_file = self.temp_dir / f"chunk_{chunk_idx:06d}.npy"
        np.save(chunk_file, data)
        self.chunk_files.append(chunk_file)
        self.total_size += len(data)
    
    def finalize(self):
        """合并所有块并删除临时文件"""
        print(f"\n合并 {self.array_name} 的 {len(self.chunk_files)} 个块...")
        
        final_file = self.output_path / f"{self.array_name}.npy"
        
        if len(self.chunk_files) == 0:
            print(f"警告: {self.array_name} 没有数据块")
            return
        
        if self.total_size < 100_000_000:
            all_chunks = [np.load(f) for f in tqdm(self.chunk_files, desc="加载块")]
            combined = np.concatenate(all_chunks, axis=0)
            np.save(final_file, combined)
            del all_chunks, combined
        else:
            print("使用内存映射模式合并大文件...")
            self._merge_with_memmap(final_file)
        
        shutil.rmtree(self.temp_dir)
        print(f"✓ {self.array_name}.npy 已保存，共 {self.total_size:,} 个元素")
    
    def _merge_with_memmap(self, output_file):
        first_chunk = np.load(self.chunk_files[0])
        shape = list(first_chunk.shape)
        shape[0] = self.total_size
        dtype = first_chunk.dtype
        
        combined = np.lib.format.open_memmap(
            output_file, mode='w+', dtype=dtype, shape=tuple(shape)
        )
        
        offset = 0
        for chunk_file in tqdm(self.chunk_files, desc="合并块"):
            chunk = np.load(chunk_file)
            chunk_size = len(chunk)
            combined[offset:offset + chunk_size] = chunk
            offset += chunk_size
            del chunk
        
        del combined


class DSECToHQFConverter:
    def __init__(self, dsec_path, output_path, 
                 event_chunk_size=10_000_000,
                 image_batch_size=50):
        self.dsec_path = Path(dsec_path)
        self.output_path = Path(output_path)
        self.event_chunk_size = event_chunk_size
        self.image_batch_size = image_batch_size
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.output_path / "conversion_state.json"
        
        print(f"DSEC路径: {self.dsec_path}")
        print(f"输出路径: {self.output_path}")
        print(f"事件分块大小: {event_chunk_size:,}")
        print(f"图像批处理大小: {image_batch_size}")
        
        self.crop_params = None
        self.mapping = None
    
    def save_state(self, state):
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return None
    
    def compute_spatial_mapping(self):
        print("\n计算空间映射矩阵...")
        
        confpath = self.dsec_path / 'calibration' / 'cam_to_cam.yaml'
        assert confpath.exists(), f"校准文件不存在: {confpath}"
        
        conf = OmegaConf.load(confpath)
        
        K_r0 = np.eye(3)
        K_r0[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect0']['camera_matrix']
        K_r1 = np.eye(3)
        K_r1[[0, 1, 0, 1], [0, 1, 2, 2]] = conf['intrinsics']['camRect1']['camera_matrix']
        
        R_r0_0 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect0']))
        R_r1_1 = Rot.from_matrix(np.array(conf['extrinsics']['R_rect1']))
        
        T_r0_0 = Transform.from_rotation(R_r0_0)
        T_r1_1 = Transform.from_rotation(R_r1_1)
        T_1_0 = Transform.from_transform_matrix(np.array(conf['extrinsics']['T_10']))
        
        T_r1_r0 = T_r1_1 @ T_1_0 @ T_r0_0.inverse()
        R_r1_r0_matrix = T_r1_r0.R().as_matrix()
        P_r1_r0 = K_r1 @ R_r1_r0_matrix @ np.linalg.inv(K_r0)
        
        ht = 480
        wd = 640
        coords = np.stack(np.meshgrid(np.arange(wd), np.arange(ht)), axis=-1)
        coords_hom = np.concatenate((coords, np.ones((ht, wd, 1))), axis=-1)
        
        mapping = (P_r1_r0 @ coords_hom[..., None]).squeeze()
        mapping = (mapping / mapping[..., -1][..., None])[..., :2]
        mapping = mapping.astype('float32')
        
        print(f"映射矩阵尺寸: {mapping.shape}")
        
        np.save(self.output_path / "spatial_mapping.npy", mapping)
        
        return mapping
    
    def detect_and_compute_crop(self, image_files, num_samples=10):
        print("\n检测黑边并计算裁剪参数...")
        
        indices = np.linspace(0, len(image_files) - 1, num_samples, dtype=int)
        
        all_valid_rows = []
        all_valid_cols = []
        
        for idx in tqdm(indices, desc="采样图像"):
            img = cv2.imread(str(image_files[idx]))
            img_mapped = cv2.remap(img, self.mapping, None, interpolation=cv2.INTER_CUBIC)
            
            gray = cv2.cvtColor(img_mapped, cv2.COLOR_BGR2GRAY)
            valid_mask = gray > 10
            
            valid_rows = np.where(valid_mask.any(axis=1))[0]
            valid_cols = np.where(valid_mask.any(axis=0))[0]
            
            if len(valid_rows) > 0:
                all_valid_rows.append((valid_rows.min(), valid_rows.max()))
            if len(valid_cols) > 0:
                all_valid_cols.append((valid_cols.min(), valid_cols.max()))
            
            del img, img_mapped, gray, valid_mask
        
        row_min = max([r[0] for r in all_valid_rows])
        row_max = min([r[1] for r in all_valid_rows])
        col_min = max([c[0] for c in all_valid_cols])
        col_max = min([c[1] for c in all_valid_cols])
        
        crop_height = row_max - row_min + 1
        crop_width = col_max - col_min + 1
        
        final_height = (crop_height // 16) * 16
        final_width = (crop_width // 16) * 16
        
        height_diff = crop_height - final_height
        width_diff = crop_width - final_width
        
        row_min += height_diff // 2
        row_max = row_min + final_height - 1
        col_min += width_diff // 2
        col_max = col_min + final_width - 1
        
        self.crop_params = {
            'row_min': int(row_min),
            'row_max': int(row_max),
            'col_min': int(col_min),
            'col_max': int(col_max),
            'height': int(final_height),
            'width': int(final_width)
        }
        
        print(f"裁剪范围: 行[{row_min}, {row_max}], 列[{col_min}, {col_max}]")
        print(f"输出尺寸: {final_height} x {final_width}")
        
        with open(self.output_path / "crop_params.json", 'w') as f:
            json.dump(self.crop_params, f, indent=2)
        
        return self.crop_params
    
    def get_event_info(self):
        print("\n获取事件数据信息...")
        events_file = self.dsec_path / "events/left/events.h5"
        
        with h5py.File(events_file, 'r') as f:
            total_events = len(f['events/x'])
            t_offset = f['t_offset'][()]
            
            t_min = f['events/t'][0]
            t_max = f['events/t'][-1]
            
            print(f"事件总数: {total_events:,}")
            print(f"时间偏移量: {t_offset} us")
            print(f"事件时间范围: {t_min} ~ {t_max}")
        
        return total_events, t_offset, t_min, t_max
    
    def process_event_chunk(self, start_idx, end_idx, t_offset):
        events_file = self.dsec_path / "events/left/events.h5"
        
        with h5py.File(events_file, 'r') as f:
            x = f['events/x'][start_idx:end_idx]
            y = f['events/y'][start_idx:end_idx]
            t = f['events/t'][start_idx:end_idx]
            p = f['events/p'][start_idx:end_idx]
        
        row_min = self.crop_params['row_min']
        row_max = self.crop_params['row_max']
        col_min = self.crop_params['col_min']
        col_max = self.crop_params['col_max']
        
        valid_mask = (x >= col_min) & (x <= col_max) & (y >= row_min) & (y <= row_max)
        
        x_cropped = x[valid_mask] - col_min
        y_cropped = y[valid_mask] - row_min
        t_cropped = t[valid_mask]
        p_cropped = p[valid_mask]
        
        events_xy = np.stack([x_cropped, y_cropped], axis=1)
        events_p = p_cropped.astype(np.int64)
        
        event_ts_abs = t_cropped + t_offset
        events_ts = event_ts_abs / 1e6
        
        return events_xy, events_p, events_ts, valid_mask.sum()
    
    def process_events_in_chunks(self, total_events, t_offset):
        print("\n分块处理事件数据...")
        
        writer_xy = ChunkedArrayWriter(self.output_path, "events_xy")
        writer_p = ChunkedArrayWriter(self.output_path, "events_p")
        writer_ts = ChunkedArrayWriter(self.output_path, "events_ts")
        
        total_processed = 0
        total_valid = 0
        min_ts = None
        
        num_chunks = (total_events + self.event_chunk_size - 1) // self.event_chunk_size
        
        for chunk_idx in tqdm(range(num_chunks), desc="处理事件块"):
            start_idx = chunk_idx * self.event_chunk_size
            end_idx = min(start_idx + self.event_chunk_size, total_events)
            
            events_xy, events_p, events_ts, num_valid = self.process_event_chunk(
                start_idx, end_idx, t_offset
            )
            
            if min_ts is None and len(events_ts) > 0:
                min_ts = events_ts.min()
            
            writer_xy.append_chunk(events_xy)
            writer_p.append_chunk(events_p)
            writer_ts.append_chunk(events_ts)
            
            total_processed += (end_idx - start_idx)
            total_valid += num_valid
            
            del events_xy, events_p, events_ts
        
        print("\n合并事件数据块...")
        writer_xy.finalize()
        writer_p.finalize()
        writer_ts.finalize()
        
        print(f"\n事件处理完成:")
        print(f"  原始事件数: {total_processed:,}")
        print(f"  有效事件数: {total_valid:,}")
        print(f"  保留比例: {total_valid / total_processed * 100:.2f}%")
        
        self.normalize_event_timestamps(min_ts)
        
        return total_valid, min_ts
    
    def normalize_event_timestamps(self, min_ts):
        print("\n归一化事件时间戳...")
        ts_file = self.output_path / "events_ts.npy"
        
        events_ts = np.load(ts_file, mmap_mode='r+')
        
        chunk_size = 50_000_000
        for i in tqdm(range(0, len(events_ts), chunk_size), desc="归一化"):
            end = min(i + chunk_size, len(events_ts))
            events_ts[i:end] = events_ts[i:end] - min_ts
        
        max_ts = events_ts[-1]
        del events_ts
        
        print(f"时间戳归一化完成，范围: [0, {max_ts:.6f}]秒")
    
    def load_images_and_timestamps(self):
        print("\n加载图像信息...")
        
        timestamp_file = self.dsec_path / "images/left/exposure_timestamps.txt"
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f:
                if not line.startswith('#') and line.strip():
                    start, end = line.strip().split(',')
                    timestamps.append([int(start.strip()), int(end.strip())])
        timestamps = np.array(timestamps)
        
        image_dir = self.dsec_path / "images/left/rectified"
        image_files = sorted([f for f in image_dir.glob("*.png")])
        
        print(f"图像数量: {len(image_files)}")
        assert len(image_files) == len(timestamps), "图像数量与时间戳数量不匹配"
        
        return image_files, timestamps
    
    def process_images_in_batches(self, image_files, timestamps, min_event_ts):
        print("\n分批处理图像...")
        
        writer_images = ChunkedArrayWriter(self.output_path, "images")
        writer_ts = ChunkedArrayWriter(self.output_path, "images_ts")
        
        num_images = len(image_files)
        num_batches = (num_images + self.image_batch_size - 1) // self.image_batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="处理图像批次"):
            start_idx = batch_idx * self.image_batch_size
            end_idx = min(start_idx + self.image_batch_size, num_images)
            
            batch_images = []
            batch_timestamps = []
            
            for idx in range(start_idx, end_idx):
                # 读取图像
                img = cv2.imread(str(image_files[idx]))
                
                # 应用映射
                img_mapped = cv2.remap(img, self.mapping, None, interpolation=cv2.INTER_CUBIC)
                
                # 裁剪
                img_cropped = img_mapped[
                    self.crop_params['row_min']:self.crop_params['row_max']+1,
                    self.crop_params['col_min']:self.crop_params['col_max']+1
                ]
                
                # 关键修正：BGR 转 RGB
                img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
                
                batch_images.append(img_cropped)
                
                # 计算中心时间戳并归一化
                ts_center = timestamps[idx].mean() / 1e6 - min_event_ts
                batch_timestamps.append(ts_center)
                
                del img, img_mapped, img_cropped
            
            batch_images = np.array(batch_images)
            batch_timestamps = np.array(batch_timestamps)[:, np.newaxis]
            
            writer_images.append_chunk(batch_images)
            writer_ts.append_chunk(batch_timestamps)
            
            del batch_images, batch_timestamps
        
        print("\n合并图像数据块...")
        writer_images.finalize()
        writer_ts.finalize()
        
        print(f"图像处理完成: {num_images} 张")
    
    def compute_image_event_indices(self, timestamps, t_offset, min_event_ts, total_valid_events):
        print("\n计算图像-事件对齐索引...")
        
        events_ts = np.load(self.output_path / "events_ts.npy", mmap_mode='r')
        
        image_ts_center = timestamps.mean(axis=1) / 1e6 - min_event_ts
        
        image_event_indices = []
        
        for i in tqdm(range(len(image_ts_center)), desc="对齐"):
            end_time = image_ts_center[i]
            end_idx = np.searchsorted(events_ts, end_time, side='right')
            image_event_indices.append(end_idx)
        
        image_event_indices = np.array(image_event_indices)[:, np.newaxis]
        
        np.save(self.output_path / "image_event_indices.npy", image_event_indices)
        print(f"✓ image_event_indices.npy: {image_event_indices.shape}")
        
        del events_ts
    
    def convert(self):
        print("="*80)
        print("开始转换 DSEC -> HQF (RGB修正版)")
        print("="*80)
        
        state = self.load_state()
        if state and state.get('completed'):
            print("\n发现已完成的转换，跳过")
            return
        
        mapping_file = self.output_path / "spatial_mapping.npy"
        if mapping_file.exists():
            print("\n加载已有的空间映射矩阵...")
            self.mapping = np.load(mapping_file)
        else:
            self.mapping = self.compute_spatial_mapping()
        
        image_files, timestamps = self.load_images_and_timestamps()
        
        crop_file = self.output_path / "crop_params.json"
        if crop_file.exists():
            print("\n加载已有的裁剪参数...")
            with open(crop_file, 'r') as f:
                self.crop_params = json.load(f)
        else:
            self.crop_params = self.detect_and_compute_crop(image_files)
        
        total_events, t_offset, t_min, t_max = self.get_event_info()
        
        total_valid_events, min_event_ts = self.process_events_in_chunks(total_events, t_offset)
        
        self.process_images_in_batches(image_files, timestamps, min_event_ts)
        
        self.compute_image_event_indices(timestamps, t_offset, min_event_ts, total_valid_events)
        
        state = {
            'completed': True,
            'total_events': int(total_valid_events),
            'total_images': len(image_files),
            'crop_params': self.crop_params
        }
        self.save_state(state)
        
        print("\n" + "="*80)
        print("转换完成!")
        print("="*80)
        
        events_ts = np.load(self.output_path / "events_ts.npy", mmap_mode='r')
        images_ts = np.load(self.output_path / "images_ts.npy", mmap_mode='r')
        images = np.load(self.output_path / "images.npy", mmap_mode='r')
        
        print(f"\n数据统计:")
        print(f"  事件总数: {len(events_ts):,}")
        print(f"  图像总数: {len(images_ts)}")
        print(f"  事件时间范围: {events_ts[0]:.6f}s ~ {events_ts[-1]:.6f}s")
        print(f"  图像时间范围: {images_ts[0][0]:.6f}s ~ {images_ts[-1][0]:.6f}s")
        print(f"  图像尺寸: {images.shape[1:3]} (H x W)")
        print(f"  图像通道数: {images.shape[3]} (彩色RGB)")  # 注意：现在是RGB
        print(f"\n输出路径: {self.output_path}")


def batch_convert(dsec_base_path, hqf_base_path, sequences=None,
                  event_chunk_size=10_000_000, image_batch_size=50):
    dsec_base = Path(dsec_base_path)
    hqf_base = Path(hqf_base_path)
    
    if sequences is None:
        sequences = [d.name for d in dsec_base.iterdir() if d.is_dir()]
        sequences.sort()
    
    print(f"找到 {len(sequences)} 个序列待转换")
    print(f"序列列表: {sequences}\n")
    
    for seq in sequences:
        print(f"\n{'='*80}")
        print(f"处理序列: {seq}")
        print(f"{'='*80}")
        
        dsec_path = dsec_base / seq
        output_path = hqf_base / seq
        
        if not dsec_path.exists():
            print(f"警告: 路径不存在，跳过: {dsec_path}")
            continue
        
        try:
            converter = DSECToHQFConverter(
                dsec_path, output_path,
                event_chunk_size=event_chunk_size,
                image_batch_size=image_batch_size
            )
            converter.convert()
        except Exception as e:
            print(f"错误: 转换序列 {seq} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    # 示例1: 转换单个序列
    # converter = DSECToHQFConverter(
    #     dsec_path="/path/to/DSEC/test/<seq>",
    #     output_path="/path/to/DSEC/mem/test/<seq>",
    #     event_chunk_size=20_000_000,
    #     image_batch_size=50
    # )
    # converter.convert()
    
    # 示例2: 批量转换
    batch_convert(
        dsec_base_path="/path/to/DSEC/train",
        hqf_base_path="/path/to/DSEC/mem/train",
        event_chunk_size=20_000_000,
        image_batch_size=100
    )