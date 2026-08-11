import os
import numpy as np
import h5py
import hdf5plugin
from pathlib import Path
import cv2

def inspect_dsec_format(dsec_base_path):
    """检查DSEC数据集格式"""
    print("=" * 80)
    print("DSEC 源数据格式检查")
    print("=" * 80)
    
    # 检查目录结构
    print("\n1. 目录结构:")
    print(f"基础路径: {dsec_base_path}")
    
    # 检查图像
    image_path = os.path.join(dsec_base_path, "images/left/rectified")
    if os.path.exists(image_path):
        images = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])
        print(f"\n图像数量: {len(images)}")
        print(f"图像命名示例: {images[:5]}")
        
        # 读取第一张图像
        first_img_path = os.path.join(image_path, images[0])
        img = cv2.imread(first_img_path, cv2.IMREAD_UNCHANGED)
        print(f"图像尺寸: {img.shape}")
        print(f"图像数据类型: {img.dtype}")
        print(f"图像通道数: {img.shape[2] if len(img.shape) > 2 else 1}")
    
    # 检查曝光时间戳
    timestamp_path = os.path.join(dsec_base_path, "images/left/exposure_timestamps.txt")
    if os.path.exists(timestamp_path):
        print(f"\n2. 曝光时间戳文件:")
        print(f"文件路径: {timestamp_path}")
        with open(timestamp_path, 'r') as f:
            lines = f.readlines()
        print(f"总行数（包括注释）: {len(lines)}")
        print(f"前10行内容:")
        for i, line in enumerate(lines[:10]):
            print(f"  {i}: {line.strip()}")
        
        # 解析时间戳
        timestamps = []
        for line in lines:
            if not line.startswith('#') and line.strip():
                parts = line.strip().split(',')
                timestamps.append([int(parts[0].strip()), int(parts[1].strip())])
        timestamps = np.array(timestamps)
        print(f"\n时间戳数组形状: {timestamps.shape}")
        print(f"时间戳范围: {timestamps.min()} ~ {timestamps.max()}")
        print(f"时间戳单位: 微秒 (us)")
    
    # 检查事件数据
    events_path = os.path.join(dsec_base_path, "events/left")
    print(f"\n3. 事件数据:")
    print(f"事件目录: {events_path}")
    
    if os.path.exists(events_path):
        # DSEC通常将事件存储为.h5文件或多个npy文件
        event_files = os.listdir(events_path)
        print(f"事件文件列表: {event_files}")
        
        # 检查可能的事件文件格式
        for fname in event_files:
            full_path = os.path.join(events_path, fname)
            if fname.endswith('.npy'):
                data = np.load(full_path)
                print(f"\n  文件: {fname}")
                print(f"    形状: {data.shape}")
                print(f"    数据类型: {data.dtype}")
                print(f"    前5个元素: {data[:5]}")
            elif fname.endswith('.h5'):
                try:
                    import h5py
                    with h5py.File(full_path, 'r') as f:
                        print(f"\n  文件: {fname}")
                        print(f"    HDF5 keys: {list(f.keys())}")
                        for key in f.keys():
                            item = f[key]
                            if isinstance(item, h5py.Dataset):
                                # 处理数据集
                                print(f"      {key}: Dataset -> shape={item.shape}, dtype={item.dtype}")
                                if item.size > 0:
                                    # 根据形状决定如何取样
                                    if item.shape == ():  # 标量数据集
                                        print(f"      值: {item[()]}")
                                    elif len(item.shape) == 1:  # 一维数组
                                        print(f"      前5个元素: {item[:5]}")
                                    else:  # 多维数组
                                        # 取第一个元素
                                        print(f"      第一个元素: {item[0]}")
                            elif isinstance(item, h5py.Group):
                                # 处理组
                                print(f"      {key}: Group -> 包含 {len(item)} 个子项")
                                # 可选：打印组的子项名称
                                print(f"        子项: {list(item.keys())}")
                                for sub_key in item.keys():
                                    sub_item = item[sub_key]
                                    if isinstance(sub_item, h5py.Dataset):
                                        print(f"          {sub_key}: Dataset -> shape={sub_item.shape}, dtype={sub_item.dtype}")
                                        # 根据形状决定如何取样
                                        if sub_item.shape == ():  # 标量数据集
                                            print(f"      值: {sub_item[()]}")
                                        elif len(sub_item.shape) == 1:  # 一维数组
                                            print(f"      前5个元素: {sub_item[:5]}")
                                        else:  # 多维数组
                                            # 取第一个元素
                                            print(f"      第一个元素: {sub_item[0]}")
                                    else:
                                        print(f"          {sub_key}: 未知类型")
                            else:
                                print(f"      {key}: 未知类型")
                except ImportError:
                    print(f"\n  文件: {fname} (需要h5py库来读取)")
    
    print("\n" + "=" * 80)


def inspect_hqf_format(hqf_sample_path):
    """检查HQF目标数据格式"""
    print("\n" + "=" * 80)
    print("HQF 目标数据格式检查")
    print("=" * 80)
    
    print(f"\n目标路径: {hqf_sample_path}")
    
    if not os.path.exists(hqf_sample_path):
        print("警告: 目标路径不存在，无法检查格式")
        print("\n请提供一个HQF格式的示例数据路径，以便了解:")
        print("  - events_p.npy: 事件极性")
        print("  - events_ts.npy: 事件时间戳")
        print("  - events_xy.npy: 事件坐标")
        print("  - image_event_indices.npy: 图像与事件的索引关系")
        print("  - images_ts.npy: 图像时间戳")
        print("  - images.npy: 图像数据")
        return
    
    files_to_check = [
        'events_p.npy',
        'events_ts.npy', 
        'events_xy.npy',
        'image_event_indices.npy',
        'images_ts.npy',
        'images.npy'
    ]
    
    for fname in files_to_check:
        fpath = os.path.join(hqf_sample_path, fname)
        print(f"\n{fname}:")
        if os.path.exists(fpath):
            data = np.load(fpath)
            print(f"  形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")
            print(f"  数据范围: [{data.min()}, {data.max()}]")
            
            if len(data.shape) == 1:
                print(f"  前10个元素: {data[:10]}")
            elif len(data.shape) == 2:
                print(f"  前5行: \n{data[:5]}")
            else:
                print(f"  前3个样本: \n{data[:3]}")
                
            # 特殊说明
            if fname == 'image_event_indices.npy':
                print(f"  说明: 这个数组应该存储每张图像对应的事件索引范围")
                print(f"        例如: [[start_idx1, end_idx1], [start_idx2, end_idx2], ...]")
            elif fname == 'events_ts.npy':
                print(f"  时间戳单位: 请确认（微秒/纳秒/秒？）")
            elif fname == 'images_ts.npy':
                print(f"  时间戳单位: 请确认（微秒/纳秒/秒？）")
        else:
            print(f"  文件不存在")
    
    print("\n" + "=" * 80)


def main():
    print("\n" + "=" * 80)
    print("数据格式信息收集程序")
    print("=" * 80)
    
    # 提示用户输入路径
    print("\n请提供以下信息：\n")
    
    # DSEC源数据路径
    dsec_path = input("1. DSEC数据路径 (例: DSEC/test/interlaken_00_b): ").strip()
    if dsec_path and os.path.exists(dsec_path):
        inspect_dsec_format(dsec_path)
    else:
        print(f"路径不存在或为空: {dsec_path}")
        print("\n假设标准DSEC格式，请手动提供以下信息：")
        print("  - 事件数据格式 (.h5 或 .npy 或其他)")
        print("  - 事件数据字段名称和含义")
        print("  - 事件的坐标范围")
    
    print("\n")
    
    # HQF目标数据示例路径
    hqf_path = input("2. HQF示例数据路径 (例: HQF/boxes，如无可跳过): ").strip()
    if hqf_path:
        inspect_hqf_format(hqf_path)
    else:
        print("\n未提供HQF示例路径，请手动提供以下信息：")
        inspect_hqf_format("")
    
    print("\n" + "=" * 80)
    print("关键问题清单（请在下次对话中提供答案）")
    print("=" * 80)
    
    questions = [
        "\n【事件数据】",
        "1. DSEC事件数据存储格式是什么？(.h5文件 / 多个.npy文件 / 其他)",
        "2. 事件数据包含哪些字段？(x, y, t, p 等)",
        "3. 事件极性 p 的取值范围是什么？(0/1 或 -1/1)",
        "4. 事件时间戳单位是什么？(微秒/纳秒)",
        "5. 事件坐标 (x, y) 的范围是多少？",
        
        "\n【图像数据】",
        "6. 图像是彩色还是灰度？",
        "7. 图像尺寸是固定的吗？",
        
        "\n【时间对齐】",
        "8. 如何确定每张图像对应的事件范围？",
        "   - 使用曝光时间戳区间？",
        "   - 使用相邻图像中点？",
        "   - 其他方式？",
        
        "\n【HQF格式】",
        "9. images.npy 是否需要将所有图像堆叠成一个数组？",
        "10. images_ts.npy 存储的是曝光开始时间、结束时间、还是中点时间？",
        "11. image_event_indices.npy 的具体格式是什么？",
        "    - 是 [[start, end], ...] 这样的索引对吗？",
        "12. 所有时间戳需要归一化或转换单位吗？",
        
        "\n【其他】",
        "13. 是否需要对事件数据进行下采样或滤波？",
        "14. 是否需要处理多个序列（如 interlaken_00_b 是一个序列）？",
    ]
    
    for q in questions:
        print(q)
    
    print("\n" + "=" * 80)
    print("请运行此程序并提供以上问题的答案，我将为你生成转换程序！")
    print("=" * 80)


if __name__ == "__main__":
    main()