"""
探测 DSEC flow 数据的真实格式，为写 align_flow.py 收集事实。

只读、不改任何东西。打印：
  - forward/backward 各自 timestamps 文件的注释行、行数、每行 token 数、前后几行
  - PNG 数量、命名、shape/dtype、每通道 min/max/unique/中位

用法（在服务器上，到 .../<seq>/flow 这一层）：
  python tools/dsec/probe_flow.py \
      --flow-root /path/to/datasets/DSEC/flow/train/thun_00_a/flow
"""
import argparse
from pathlib import Path

import numpy as np
import cv2


def probe_one(flow_root: Path, direction: str):
    sub = flow_root / direction
    print('=' * 72)
    print(direction)
    print('=' * 72)

    if not sub.exists():
        print(f'  [缺失] 目录不存在: {sub}')
        return

    # ---- timestamps ----
    ts = flow_root / f'{direction}_timestamps.txt'
    print(f'\n[{ts.name}]')
    if not ts.exists():
        print('  [缺失] 时间戳文件不存在')
    else:
        comments, data_lines = [], []
        with open(ts, encoding='utf-8') as f:
            for l in f:
                l = l.rstrip('\n')
                if l.startswith('#'):
                    comments.append(l)
                elif l.strip():
                    data_lines.append(l)
        print('注释行：')
        for c in comments:
            print('   ', c)
        print(f'数据行数: {len(data_lines)}')
        print('前 3 行：')
        for l in data_lines[:3]:
            print('   ', l)
        print('后 3 行：')
        for l in data_lines[-3:]:
            print('   ', l)
        if data_lines:
            toks = data_lines[0].replace(',', ' ').split()
            try:
                as_int = [int(round(float(t))) for t in toks]
                print(f'每行 token 数: {len(toks)} | 都能解析为整数: {as_int}')
            except ValueError:
                print(f'每行 token 数: {len(toks)} | 不是整数: {toks}')

    # ---- PNG ----
    pngs = sorted(sub.glob('*.png'))
    print(f'\nPNG 数量: {len(pngs)}')
    if pngs:
        print(f'前 5 文件名: {[p.name for p in pngs[:5]]}')
        print(f'后 5 文件名: {[p.name for p in pngs[-5:]]}')
        for p in [pngs[0], pngs[len(pngs) // 2], pngs[-1]]:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f'\n{p.name}: 读取失败（可能不是标准 PNG）')
                continue
            print(f'\n{p.name}: shape={img.shape}, dtype={img.dtype}, ndim={img.ndim}')
            if img.ndim == 2:
                print(f'  min={img.min()}, max={img.max()}, unique={len(np.unique(img))}')
            else:
                for c in range(img.shape[2]):
                    ch = img[..., c]
                    print(f'  ch{c}: min={ch.min()}, max={ch.max()}, '
                          f'unique={len(np.unique(ch))}, 中位={np.median(ch):.0f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flow-root', required=True,
                    help='到 .../<seq>/flow 这一层（里面是 forward/ backward/ 两个 timestamps）')
    args = ap.parse_args()
    flow_root = Path(args.flow_root)
    print(f'flow_root = {flow_root}')

    for d in ['forward', 'backward']:
        probe_one(flow_root, d)
        print()


if __name__ == '__main__':
    main()
