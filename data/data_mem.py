import os
import warnings
from bisect import bisect_left
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import h5py
# local modules
from utils.data_augmentation import *
from utils.util import read_json, write_json
# local modules
from utils.util import read_json
from utils.data_augmentation import *


def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    """
    Binary search through a sorted HDF5 pre-loaded dataset (memory efficient
    as the entire DSET is _not_ loaded into RAM)
    """
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r
    while l <= r:
        mid = l + (r - l)//2;
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r


class MemMapDatasetVal(Dataset):
    """
    Validation dataset that returns random continuous sequence segments from multiple sequences.
    Each __getitem__ call returns a fixed-length continuous segment randomly sampled from one of the sequences.
    Optimized for fast batch loading.
    """

    def __init__(self, data_path, transforms={}, sensor_resolution=None, num_bins=5,
                 voxel_method=None, segment_length=16, num_samples=1000, seed=None):
        """
        Args:
            data_path: Path to folder containing multiple sequence subfolders
            transforms: Dictionary of transforms to apply
            sensor_resolution: Sensor resolution [H, W]
            num_bins: Number of bins for voxel grid
            voxel_method: Method for creating voxels (default: 'between_frames')
            segment_length: Number of consecutive frames/voxels to return in each sample
            num_samples: Total number of samples in validation set (for epoch definition)
            seed: Random seed for reproducibility
        """
        self.num_bins = num_bins
        self.data_path = data_path
        self.sensor_resolution = sensor_resolution
        self.channels = self.num_bins
        self.segment_length = segment_length
        self.num_samples = num_samples
        self.rng = np.random.RandomState(seed)
        
        # Load all sequences
        self.sequences = []
        self.load_multi_sequence_data(data_path)
        
        if voxel_method is None:
            voxel_method = {'method': 'between_frames'}
        self.voxel_method = voxel_method
        
        # Compute voxel indices for all sequences
        self.set_voxel_method()
        print(f"Using method: {self.voxel_method}")
        print(f"Segment length: {self.segment_length}")
        print(f"Total validation samples: {self.num_samples}")
        
        # Setup transforms
        transforms_list = [eval(t)(**kwargs) for t, kwargs in transforms.items()]
        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            from torchvision.transforms import Compose
            self.transform = Compose(transforms_list)

    def load_multi_sequence_data(self, root_path):
        """Load metadata for all sequences in the folder"""
        assert os.path.isdir(root_path), f'{root_path} is not a valid directory'
        
        subdirs = sorted([d for d in os.listdir(root_path) 
                         if os.path.isdir(os.path.join(root_path, d))])
        
        if len(subdirs) == 0:
            raise ValueError(f"No subdirectories found in {root_path}")
        
        for seq_idx, subdir in enumerate(subdirs):
            seq_path = os.path.join(root_path, subdir)
            
            # Check if this is a valid sequence folder
            if not (os.path.exists(os.path.join(seq_path, 'events_ts.npy')) and
                    os.path.exists(os.path.join(seq_path, 'events_xy.npy')) and
                    os.path.exists(os.path.join(seq_path, 'events_p.npy'))):
                print(f"Skipping {subdir}: not a valid sequence folder")
                continue
            
            # Load this sequence
            seq_data = self.load_sequence_data(seq_path, subdir)
            self.sequences.append(seq_data)
            
            print(f"Loaded sequence {seq_idx}: {subdir}")
        
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences found in {root_path}")
        
        print(f"Total sequences loaded: {len(self.sequences)}")

    def load_sequence_data(self, seq_path, seq_name):
        """Load data for a single sequence"""
        data = {}
        
        # Load event data
        data["t"] = np.load(os.path.join(seq_path, 'events_ts.npy'), mmap_mode='r').squeeze()
        data["xy"] = np.load(os.path.join(seq_path, 'events_xy.npy'), mmap_mode='r').squeeze()
        data["p"] = np.load(os.path.join(seq_path, 'events_p.npy'), mmap_mode='r').squeeze()
        
        # Load image data if available
        images_path = os.path.join(seq_path, 'images.npy')
        images_ts_path = os.path.join(seq_path, 'images_ts.npy')
        image_event_indices_path = os.path.join(seq_path, 'image_event_indices.npy')
        
        has_images = (os.path.exists(images_ts_path) and 
                     os.path.exists(images_path) and 
                     os.path.exists(image_event_indices_path))
        
        if has_images:
            data["frame_stamps"] = np.load(images_ts_path)
            data["images"] = np.load(images_path, mmap_mode='r')
            data["image_event_indices"] = np.load(image_event_indices_path)
            num_frames = len(data['images'])
            frame_ts = [ts.item() for ts in data["frame_stamps"]]
        else:
            num_frames = 0
            frame_ts = []
        
        # Sanity checks
        assert len(data['p']) == len(data['xy']) == len(data['t']), \
            "Number of events, timestamps and coordinates do not match"
        
        # Determine sensor resolution
        sensor_resolution = self.sensor_resolution
        if sensor_resolution is None:
            metadata_path = os.path.join(seq_path, "metadata.json")
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                sensor_resolution = metadata["sensor_resolution"]
            else:
                if has_images and num_frames > 0:
                    sensor_resolution = list(data["images"][0].shape[:2])
                else:
                    sensor_resolution = [int(np.max(data["xy"][:, 1]) + 1),
                                        int(np.max(data["xy"][:, 0]) + 1)]
            
            # Crop resolution to be multiple of 16
            sensor_resolution = [
                (sensor_resolution[0] // 16) * 16,
                (sensor_resolution[1] // 16) * 16
            ]
            
            self.sensor_resolution = sensor_resolution
        
        seq_info = {
            'name': seq_name,
            'path': seq_path,
            'filehandle': data,
            't0': data['t'][0],
            'tk': data['t'][-1],
            'num_events': len(data['p']),
            'num_frames': num_frames,
            'frame_ts': frame_ts,
            'has_images': has_images,
            'sensor_resolution': sensor_resolution
        }
        
        return seq_info

    def set_voxel_method(self):
        """Compute voxel method for all sequences"""
        for seq_data in self.sequences:
            self._compute_sequence_voxel_method(seq_data)

    def _compute_sequence_voxel_method(self, seq_data):
        """Compute voxel method for a specific sequence"""
        filehandle = seq_data['filehandle']
        
        if self.voxel_method['method'] == 'between_frames':
            assert seq_data['has_images'], "Cannot use between_frames voxel method without images"
            event_indices = self._compute_frame_indices(filehandle)
            length = seq_data['num_frames'] - 1
        elif self.voxel_method['method'] == 't_seconds':
            duration = seq_data['tk'] - seq_data['t0']
            t = self.voxel_method['t']
            sliding_window_t = self.voxel_method.get('sliding_window_t', 0)
            length = max(int(duration / (t - sliding_window_t)), 0)
            event_indices = self._compute_timeblock_indices(seq_data, length)
        elif self.voxel_method['method'] == 'k_events':
            k = self.voxel_method['k']
            sliding_window_w = self.voxel_method.get('sliding_window_w', 0)
            length = max(int(seq_data['num_events'] / (k - sliding_window_w)), 0)
            event_indices = self._compute_k_indices(seq_data, length)
        else:
            raise ValueError(f"Invalid voxel forming method: {self.voxel_method['method']}")
        
        seq_data['event_indices'] = event_indices
        seq_data['length'] = length
        print(f"  Sequence '{seq_data['name']}': {length} frames/voxels available")

    def _compute_frame_indices(self, filehandle):
        """Compute frame indices for between_frames method"""
        frame_indices = []
        start_idx = 0
        for event_idx in filehandle["image_event_indices"]:
            end_idx = event_idx[0]
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices

    def _compute_timeblock_indices(self, seq_data, length):
        """Compute time block indices for t_seconds method"""
        timeblock_indices = []
        filehandle = seq_data['filehandle']
        t0 = seq_data['t0']
        t = self.voxel_method['t']
        sliding_window_t = self.voxel_method.get('sliding_window_t', 0)
        
        start_idx = 0
        for i in range(length):
            start_time = ((t - sliding_window_t) * i) + t0
            end_time = start_time + t
            end_idx = np.searchsorted(filehandle["t"], end_time)
            timeblock_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return timeblock_indices

    def _compute_k_indices(self, seq_data, length):
        """Compute k-event indices"""
        k_indices = []
        k = self.voxel_method['k']
        sliding_window_w = self.voxel_method.get('sliding_window_w', 0)
        
        start_idx = 0
        for i in range(length):
            end_idx = start_idx + k
            if end_idx > seq_data['num_events']:
                break
            k_indices.append([start_idx, end_idx])
            start_idx += (k - sliding_window_w)
        return k_indices

    def __len__(self):
        """Return the number of validation samples"""
        return self.num_samples

    def sample_random_segment(self):
        """
        Randomly sample a sequence and a valid starting position for a segment
        Returns: (sequence_index, start_frame_index)
        """
        # Filter sequences that have enough frames for a segment
        valid_sequences = [(i, seq) for i, seq in enumerate(self.sequences) 
                          if seq['length'] >= self.segment_length]
        
        if len(valid_sequences) == 0:
            raise ValueError(f"No sequences have enough frames (>= {self.segment_length})")
        
        # Randomly select a sequence
        seq_idx, seq_data = valid_sequences[self.rng.randint(0, len(valid_sequences))]
        
        # Randomly select a starting position
        max_start = seq_data['length'] - self.segment_length
        start_idx = self.rng.randint(0, max_start + 1)
        
        return seq_idx, start_idx

    def __getitem__(self, index):
        """
        Get a random continuous segment from a random sequence
        Optimized version that batch-loads data
        """
        # Sample a random segment
        seq_idx, start_idx = self.sample_random_segment()
        seq_data = self.sequences[seq_idx]
        
        filehandle = seq_data['filehandle']
        event_indices = seq_data['event_indices']
        frame_ts = seq_data['frame_ts']
        has_images = seq_data['has_images']
        sensor_resolution = seq_data['sensor_resolution']
        
        end_idx = start_idx + self.segment_length
        
        # Batch load frames if available
        if has_images:
            # Load all frames at once (much faster than one by one)
            frames_np = filehandle['images'][start_idx:end_idx]  # [T, H, W] or [T, H, W, C]
            
            # Crop frames to sensor_resolution (16x multiple)
            frames_np = frames_np[:, :sensor_resolution[0], :sensor_resolution[1]]
            
            frames = torch.from_numpy(frames_np.astype(np.float32)) / 255.0  * 2.0 - 1
            
            # Handle grayscale vs color
            if frames.ndim == 3:
                frames = frames.unsqueeze(1)  # [T, 1, H, W]
            else:
                frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
            
            # Get timestamps
            timestamps = torch.tensor([frame_ts[i] for i in range(start_idx, end_idx)], dtype=torch.float64)
        else:
            frames = torch.zeros((self.segment_length, 1, sensor_resolution[0], sensor_resolution[1]),
                               dtype=torch.float32)
            timestamps = torch.zeros(self.segment_length, dtype=torch.float64)
        
        # Batch process events for all frames in segment
        events_list = []
        dts_list = []
        
        for i in range(self.segment_length):
            frame_idx = start_idx + i
            
            # Get event indices for this frame
            if self.voxel_method['method'] == 'between_frames':
                if frame_idx > 0:
                    _, idx0 = event_indices[frame_idx - 1]
                else:
                    idx0 = 0
                _, idx1 = event_indices[frame_idx]
            else:
                idx0, idx1 = event_indices[frame_idx]
            
            # Get events for this frame
            if idx1 > idx0:
                # Batch load events
                xy = filehandle["xy"][idx0:idx1]
                ts = filehandle["t"][idx0:idx1]
                ps = filehandle["p"][idx0:idx1]
                
                xs = xy[:, 0].astype(np.float32)
                ys = xy[:, 1].astype(np.float32)
                ps = ps * 2.0 - 1.0
                
                ts_0, ts_k = ts[0], ts[-1]
                dt = ts_k - ts_0
                
                # Convert to torch tensors
                xs = torch.from_numpy(xs)
                ys = torch.from_numpy(ys)
                ts_torch = torch.from_numpy((ts - ts_0).astype(np.float32))
                ps = torch.from_numpy(ps.astype(np.float32))
                
                # Create voxel grid
                voxel = self.get_voxel_grid_fast(xs, ys, ts_torch, ps, sensor_resolution)
            else:
                # Empty voxel grid
                voxel = torch.zeros((self.num_bins, *sensor_resolution), dtype=torch.float32)
                dt = 0.0
            
            events_list.append(voxel)
            dts_list.append(dt)
        
        # Stack events
        events = torch.stack(events_list, dim=0)  # [T, num_bins, H, W]
        dts = torch.tensor(dts_list, dtype=torch.float64)
        
        # Apply transforms (same seed for all frames in segment)
        if self.transform:
            seed = self.rng.randint(0, 2**32)
            # Apply to each frame with same seed
            transformed_frames = []
            transformed_events = []
            for i in range(self.segment_length):
                random.seed(seed)
                transformed_frames.append(self.transform(frames[i]))
                random.seed(seed)
                transformed_events.append(self.transform(events[i]))
            frames = torch.stack(transformed_frames, dim=0)
            events = torch.stack(transformed_events, dim=0)
        
        return {
            'frames': frames,
            'events': events,
            'timestamps': timestamps,
            'dts': dts,
            'sequence_name': seq_data['name'],
            'sequence_idx': seq_idx,
            'segment_start': start_idx,
            'segment_length': self.segment_length
        }

    def get_voxel_grid_fast(self, xs, ys, ts, ps, sensor_resolution):
        """
        Fast voxel grid creation using vectorized operations
        """
        voxel = torch.zeros((self.num_bins, *sensor_resolution), dtype=torch.float32)
        
        if len(ts) == 0:
            return voxel
        
        # Normalize timestamps to [0, 1]
        ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-6)
        
        # Compute bin indices
        bin_indices = (ts_norm * (self.num_bins - 1)).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        
        # Convert to integer coordinates
        x_idx = xs.long()
        y_idx = ys.long()
        
        # Create mask for valid coordinates
        valid_mask = (x_idx >= 0) & (x_idx < sensor_resolution[1]) & \
                     (y_idx >= 0) & (y_idx < sensor_resolution[0])
        
        # Filter valid events
        x_idx = x_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        bin_indices = bin_indices[valid_mask]
        ps_valid = ps[valid_mask]
        
        # Use index_put_ for faster accumulation
        if len(x_idx) > 0:
            indices = (bin_indices, y_idx, x_idx)
            voxel.index_put_(indices, ps_valid, accumulate=True)
        
        return voxel



        


def main():
    """Test the dataset"""
    import time
    
    # Configuration
    data_path = './bs_ergb_optimized'  # TODO: replace with your local BS-ERGB path
    segment_length = 20
    num_samples = 10
    
    print("="*80)
    print("Testing MemMapDatasetVal")
    print("="*80)
    
    # Create dataset
    dataset = MemMapDatasetVal(
        data_path=data_path,
        segment_length=segment_length,
        num_samples=num_samples,
        num_bins=5,
        voxel_method={'method': 'between_frames'},
        seed=42
    )
    
    print(f"\nDataset created with {len(dataset)} samples")
    print(f"Segment length: {segment_length}")
    print(f"Number of sequences: {len(dataset.sequences)}")
    print(f"Sensor resolution: {dataset.sensor_resolution}")
    
    # Test a few samples
    print("\n" + "="*80)
    print("Testing sample retrieval")
    print("="*80)
    
    for i in range(3):
        print(f"\n--- Sample {i} ---")
        start_time = time.time()
        sample = dataset[i]
        elapsed = time.time() - start_time
        
        print(f"Sequence: {sample['sequence_name']}")
        print(f"Sequence index: {sample['sequence_idx']}")
        print(f"Segment start: {sample['segment_start']}")
        print(f"Frames shape: {sample['frames'].shape}")
        print(f"Events shape: {sample['events'].shape}")
        print(f"Timestamps shape: {sample['timestamps'].shape}")
        print(f"DTs shape: {sample['dts'].shape}")
        print(f"Frame value range: [{sample['frames'].min():.3f}, {sample['frames'].max():.3f}]")
        print(f"Events value range: [{sample['events'].min():.3f}, {sample['events'].max():.3f}]")
        print(f"Time to load: {elapsed:.3f}s")
    
    # Test DataLoader
    print("\n" + "="*80)
    print("Testing with DataLoader")
    print("="*80)
    
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )
    
    print(f"DataLoader created with batch_size=4, num_workers=2")
    
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= 3:  # Only test first 3 batches
            break
        
        print(f"\nBatch {batch_idx}:")
        print(f"  Frames: {batch['frames'].shape}")
        print(f"  Events: {batch['events'].shape}")
        print(f"  Sequences: {batch['sequence_name']}")
        print(f"  Segment starts: {batch['segment_start'].tolist()}")
    
    elapsed = time.time() - start_time
    print(f"\nTime for 3 batches: {elapsed:.3f}s")
    
    # Test statistics
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    
    for seq_idx, seq in enumerate(dataset.sequences):
        print(f"\nSequence {seq_idx}: {seq['name']}")
        print(f"  Total frames: {seq['length']}")
        print(f"  Total events: {seq['num_events']}")
        print(f"  Duration: {seq['tk'] - seq['t0']:.3f}s")
        print(f"  Has images: {seq['has_images']}")
        print(f"  Available segments: {seq['length'] - segment_length + 1}")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
