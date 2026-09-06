"""MCU 的画面布局与缩放。

刻意不用 PIL/cv2:最近邻缩放用预计算的索引数组做,纯 numpy 切片,
这是 Python 层能做到的较快形态。代价是画质不如双线性,实验场景够用。

**实测提醒**:即便如此,1280x720 输出下合成仍要约 30ms/帧——
花在几次全画布级的内存拷贝上(清屏、fancy indexing 的中间数组、tobytes)。
原以为「合成开销可忽略、瓶颈在编码」,实测推翻了这个预期:
**在 Python 里做像素级合成,搬运内存本身就是大头。**
生产级 MCU 用 C/GPU 做合成,正是为了消掉这一块。
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


def grid_shape(count: int) -> tuple[int, int]:
    """n 路输入 → (列数, 行数)。1→1x1, 2→2x1, 3~4→2x2, 5~9→3x3。"""
    if count <= 1:
        return 1, 1
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    return cols, rows


@lru_cache(maxsize=32)
def _scale_index(src_h: int, src_w: int, dst_h: int, dst_w: int):
    """最近邻映射表;同一组尺寸只算一次。"""
    ys = (np.arange(dst_h) * src_h // dst_h).clip(0, src_h - 1)
    xs = (np.arange(dst_w) * src_w // dst_w).clip(0, src_w - 1)
    return ys, xs


def scale_into(canvas: np.ndarray, frame: np.ndarray, top: int, left: int,
               cell_h: int, cell_w: int) -> None:
    """把一路输入缩放后写进画布的指定格子(原地写,不产生中间大数组)。"""
    ys, xs = _scale_index(frame.shape[0], frame.shape[1], cell_h, cell_w)
    canvas[top : top + cell_h, left : left + cell_w] = frame[ys][:, xs]


def cell_rects(count: int, out_w: int, out_h: int) -> list[tuple[int, int, int, int]]:
    """返回每个格子的 (top, left, cell_h, cell_w)。"""
    cols, rows = grid_shape(count)
    cell_w, cell_h = out_w // cols, out_h // rows
    return [
        ((i // cols) * cell_h, (i % cols) * cell_w, cell_h, cell_w)
        for i in range(count)
    ]
