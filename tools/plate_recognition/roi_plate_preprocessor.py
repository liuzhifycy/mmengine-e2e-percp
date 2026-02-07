#!/usr/bin/env python3
"""
车牌ROI裁剪与预处理工具 — 面向固定摄像头、固定角度场景

提供两套方案:
  方案1 (SimpleROI):  固定ROI裁剪 + 简单预处理(灰度/二值化/resize)
  方案2 (PreciseROI): 固定ROI裁剪 + 颜色筛选 + 形态学精确定位

典型使用场景: 道闸/停车场入口, 摄像头固定, 车牌出现区域可预设
后接 LPRNet/CRNN 等轻量模型直接识别字符, 跳过检测阶段

用法:
  # 方案1: 简单ROI裁剪
  python roi_plate_preprocessor.py --image test.jpg --mode simple --roi 520,384,1069,497

  # 方案2: 精确裁剪
  python roi_plate_preprocessor.py --image test.jpg --mode precise --roi 400,300,1200,600

  # 交互式选择ROI (鼠标框选)
  python roi_plate_preprocessor.py --image test.jpg --mode precise --interactive

  # 批量测试 CCPD 数据集 (用标注bbox做ROI模拟)
  python roi_plate_preprocessor.py --eval-ccpd data/ccpd/combined --mode both --num-samples 200

  # 在 test_pic 图片上演示 (自动用 HyperLPR3 获取 bbox 做ROI)
  python roi_plate_preprocessor.py --demo --output-dir roi_demo_output
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ===========================================================================
# 方案1: SimpleROI — 固定ROI裁剪 + 简单预处理
# ===========================================================================
class SimpleROIPreprocessor:
    """
    最简方案: 直接裁剪固定区域, 做基础图像增强, 输出给识别模型

    流程:
      原图 → 裁剪ROI → 灰度化 → 对比度增强(CLAHE) → 自适应二值化(可选) → resize

    适用: ROI划得非常精确, 车牌几乎填满ROI
    优点: 极快(纯OpenCV, <1ms), 零参数调节
    缺点: ROI必须精确, 容错空间小
    """

    def __init__(self, target_size: Tuple[int, int] = (94, 24)):
        """
        Args:
            target_size: 输出尺寸 (w, h), LPRNet标准输入是 (94, 24)
        """
        self.target_w, self.target_h = target_size
        # CLAHE 对比度增强
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 2))

    def process(self, image: np.ndarray, roi: Tuple[int, int, int, int],
                return_steps: bool = False) -> dict:
        """
        执行简单ROI裁剪和预处理

        Args:
            image: 原始BGR图像
            roi: (x1, y1, x2, y2) 裁剪区域
            return_steps: 是否返回中间步骤图像(用于可视化调试)

        Returns:
            dict: {
                'plate_color': 裁剪后的彩色车牌 (h, w, 3), 用于彩色模型
                'plate_gray':  灰度增强后 (h, w), 用于灰度模型
                'plate_bin':   二值化后 (h, w), 用于传统方法
                'plate_resized': resize到标准尺寸 (target_h, target_w, 3)
                'steps': 中间步骤 (仅 return_steps=True 时)
                'time_ms': 处理耗时
            }
        """
        t0 = time.perf_counter()
        steps = {}
        x1, y1, x2, y2 = roi

        # --- Step 1: 裁剪 ROI ---
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2].copy()

        if crop.size == 0:
            return {'error': 'ROI裁剪结果为空', 'time_ms': 0}

        if return_steps:
            steps['1_crop'] = crop.copy()

        # --- Step 2: 灰度化 ---
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if return_steps:
            steps['2_gray'] = gray.copy()

        # --- Step 3: CLAHE 对比度增强 ---
        enhanced = self.clahe.apply(gray)
        if return_steps:
            steps['3_clahe'] = enhanced.copy()

        # --- Step 4: 自适应二值化 ---
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )
        if return_steps:
            steps['4_binary'] = binary.copy()

        # --- Step 5: Resize 到标准输入尺寸 ---
        plate_resized = cv2.resize(crop, (self.target_w, self.target_h),
                                   interpolation=cv2.INTER_CUBIC)
        if return_steps:
            steps['5_resized'] = plate_resized.copy()

        elapsed = (time.perf_counter() - t0) * 1000

        result = {
            'plate_color': crop,
            'plate_gray': enhanced,
            'plate_bin': binary,
            'plate_resized': plate_resized,
            'time_ms': round(elapsed, 2),
        }
        if return_steps:
            result['steps'] = steps
        return result


# ===========================================================================
# 方案2: PreciseROI — 颜色筛选 + 形态学精确定位
# ===========================================================================
class PreciseROIPreprocessor:
    """
    精确方案: 在粗ROI内用颜色+形态学找到车牌精确边界

    流程:
      原图 → 裁剪粗ROI → HSV颜色筛选(蓝/绿/黄) → 形态学闭合 → 轮廓查找
      → 选最大合理轮廓 → 透视矫正(可选) → resize

    适用: ROI划得比较粗(车牌只占ROI的一部分), 需要自动精确定位
    优点: 容错空间大, ROI不需要太精确
    缺点: 稍慢(~3-5ms), 对车牌颜色有假设
    """

    # 中国车牌颜色范围 (HSV, OpenCV H: 0-180)
    # 实测CCPD数据集中蓝牌 H 集中在 90-125, 但室外光照下可偏至 55-140
    # 因此使用较宽的范围, 宁可多匹配再靠形态学筛选
    PLATE_COLORS = {
        'blue': {
            'lower': np.array([85, 50, 50]),
            'upper': np.array([140, 255, 255]),
        },
        'blue_dark': {  # 暗光/阴影下的蓝牌
            'lower': np.array([55, 30, 30]),
            'upper': np.array([85, 255, 200]),
        },
        'green': {  # 新能源绿牌 (更严格的S范围避免匹配植被)
            'lower': np.array([35, 50, 80]),
            'upper': np.array([85, 255, 255]),
        },
        'yellow': {
            'lower': np.array([15, 60, 100]),
            'upper': np.array([40, 255, 255]),
        },
    }

    # 车牌宽高比范围 (标准蓝牌约 3.14:1, 新能源约 3.68:1)
    ASPECT_RATIO_RANGE = (2.0, 5.5)
    # 车牌面积占ROI面积的最小比例
    MIN_AREA_RATIO = 0.05

    def __init__(self, target_size: Tuple[int, int] = (94, 24),
                 plate_types: Optional[List[str]] = None):
        """
        Args:
            target_size: 输出尺寸 (w, h)
            plate_types: 要检测的车牌颜色, 默认全部 ['blue', 'green', 'yellow']
        """
        self.target_w, self.target_h = target_size
        self.plate_types = plate_types or list(self.PLATE_COLORS.keys())
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 2))

    def _color_filter(self, hsv: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        多颜色通道筛选, 返回合并掩码和颜色类型

        对每种车牌颜色做 inRange, 同类颜色合并(如 blue + blue_dark)
        取有效像素最多的颜色类型
        """
        # 按基础颜色类型分组
        color_groups = {}
        for color_name, params in self.PLATE_COLORS.items():
            base = color_name.split('_')[0]  # blue_dark -> blue
            mask = cv2.inRange(hsv, params['lower'], params['upper'])
            if base not in color_groups:
                color_groups[base] = mask
            else:
                color_groups[base] = cv2.bitwise_or(color_groups[base], mask)

        best_mask = None
        best_color = 'unknown'
        best_count = 0

        for color_name, mask in color_groups.items():
            count = cv2.countNonZero(mask)
            if count > best_count:
                best_count = count
                best_mask = mask
                best_color = color_name

        if best_mask is None:
            best_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        return best_mask, best_color

    def _morphology_refine(self, mask: np.ndarray) -> np.ndarray:
        """
        形态学操作: 先腐蚀去噪 → 闭合填充 → 开运算清理

        使用适度的核大小, 避免把非车牌区域也连成大块
        """
        # 先腐蚀: 去掉零散的小噪点
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(mask, kernel_erode, iterations=1)

        # 闭合: 水平方向连接字符区域 (核宽>高, 模拟车牌横向特征)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
        closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_close)

        # 开运算: 去除竖向细长噪声
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

        return opened

    def _find_plate_contour(self, mask: np.ndarray,
                            roi_area: int) -> Optional[np.ndarray]:
        """
        在掩码中找到最可能是车牌的轮廓

        筛选条件:
          1. 面积 > ROI面积的 MIN_AREA_RATIO
          2. 宽高比在 ASPECT_RATIO_RANGE 内
          3. 取面积最大的那个
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < roi_area * self.MIN_AREA_RATIO:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle = rect
            if rw == 0 or rh == 0:
                continue

            # 确保宽>高
            if rw < rh:
                rw, rh = rh, rw
                angle += 90

            aspect = rw / rh
            if self.ASPECT_RATIO_RANGE[0] <= aspect <= self.ASPECT_RATIO_RANGE[1]:
                candidates.append((area, cnt, rect))

        if not candidates:
            return None

        # 返回面积最大的候选
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _perspective_correct(self, image: np.ndarray,
                             contour: np.ndarray) -> np.ndarray:
        """
        对车牌做透视矫正, 输出正面视角

        用最小外接矩形的4个角点做透视变换
        """
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # 排序角点: 左上, 右上, 右下, 左下
        box = self._order_points(box.astype(np.float32))

        # 目标尺寸
        dst_w, dst_h = self.target_w * 3, self.target_h * 3  # 先放大3倍保留细节
        dst = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(box, dst)
        warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
        return warped

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        将4个点排列为: 左上, 右上, 右下, 左下
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # 左上: x+y 最小
        rect[2] = pts[np.argmax(s)]   # 右下: x+y 最大
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]   # 右上: y-x 最小
        rect[3] = pts[np.argmax(d)]   # 左下: y-x 最大
        return rect

    def _edge_based_detect(self, crop: np.ndarray,
                           return_steps: bool = False) -> Tuple[Optional[np.ndarray], dict]:
        """
        基于Sobel边缘的车牌定位 (颜色无关, 更通用)

        原理: 车牌字符产生密集的竖向边缘, 水平闭合后形成矩形
        """
        steps = {}
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Sobel 水平梯度
        sobel_x = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
        _, sobel_bin = cv2.threshold(
            sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if return_steps:
            steps['edge_sobel'] = sobel_bin.copy()

        # 水平闭合连接字符, 竖向适度
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        closed = cv2.morphologyEx(sobel_bin, cv2.MORPH_CLOSE, kernel)

        # 开运算去噪
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2)
        if return_steps:
            steps['edge_morph'] = opened.copy()

        roi_area = crop.shape[0] * crop.shape[1]
        contour = self._find_plate_contour(opened, roi_area)
        return contour, steps

    def process(self, image: np.ndarray, roi: Tuple[int, int, int, int],
                return_steps: bool = False) -> dict:
        """
        执行精确车牌定位和裁剪

        策略: 颜色筛选 和 边缘检测 并行, 从两组候选中选最优

        Args:
            image: 原始BGR图像
            roi: (x1, y1, x2, y2) 粗略ROI区域
            return_steps: 是否返回中间步骤图像

        Returns:
            dict: {
                'plate_color': 精确裁剪的彩色车牌
                'plate_corrected': 透视矫正后的车牌
                'plate_resized': resize到标准尺寸
                'plate_type': 检测到的车牌颜色类型
                'precise_bbox': 在ROI内的精确bbox (rx1,ry1,rx2,ry2)
                'global_bbox': 在原图中的精确bbox (gx1,gy1,gx2,gy2)
                'method': 使用的定位方法 (color/edge/fallback)
                'steps': 中间步骤 (仅 return_steps=True 时)
                'time_ms': 处理耗时
            }
        """
        t0 = time.perf_counter()
        steps = {}
        x1, y1, x2, y2 = roi

        # --- Step 1: 裁剪粗 ROI ---
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2].copy()

        if crop.size == 0:
            return {'error': 'ROI裁剪结果为空', 'time_ms': 0}

        roi_area = crop.shape[0] * crop.shape[1]
        if return_steps:
            steps['1_roi_crop'] = crop.copy()

        # --- Step 2: 两路并行检测 ---
        # 路径A: HSV颜色筛选
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        color_mask, plate_type = self._color_filter(hsv)
        color_refined = self._morphology_refine(color_mask)
        color_contour = self._find_plate_contour(color_refined, roi_area)

        if return_steps:
            steps['2a_color_mask'] = color_mask.copy()
            steps['2a_color_morph'] = color_refined.copy()

        # 路径B: Sobel边缘检测
        edge_contour, edge_steps = self._edge_based_detect(crop, return_steps)
        if return_steps:
            steps.update({f'2b_{k}': v for k, v in edge_steps.items()})

        # --- Step 3: 选择最佳候选 ---
        # 优先用颜色的结果(如果面积合理), 否则用边缘的
        contour = None
        method = 'fallback'

        candidates = []
        if color_contour is not None:
            ca = cv2.contourArea(color_contour)
            # 颜色候选的面积应在ROI的 5%-60%
            if 0.05 * roi_area < ca < 0.60 * roi_area:
                candidates.append(('color', color_contour, ca))
        if edge_contour is not None:
            ea = cv2.contourArea(edge_contour)
            if 0.05 * roi_area < ea < 0.60 * roi_area:
                candidates.append(('edge', edge_contour, ea))

        if candidates:
            # 在合理候选中, 优先选颜色; 如果只有边缘也用边缘
            # 如果两者都有, 选宽高比更接近3.14:1的
            best_score = -1
            for mname, cnt, area in candidates:
                rect = cv2.minAreaRect(cnt)
                (_, _), (rw, rh), _ = rect
                if rw < rh:
                    rw, rh = rh, rw
                if rh == 0:
                    continue
                aspect = rw / rh
                # 标准车牌宽高比约 3.14, 新能源约 3.68
                score = 1.0 / (abs(aspect - 3.4) + 0.1)
                if mname == 'color':
                    score *= 1.2  # 颜色匹配给一点加分
                if score > best_score:
                    best_score = score
                    contour = cnt
                    method = mname

        if contour is None:
            # 完全回退: 直接用整个ROI
            elapsed = (time.perf_counter() - t0) * 1000
            plate_resized = cv2.resize(crop, (self.target_w, self.target_h),
                                       interpolation=cv2.INTER_CUBIC)
            result = {
                'plate_color': crop,
                'plate_corrected': crop,
                'plate_resized': plate_resized,
                'plate_type': plate_type,
                'precise_bbox': (0, 0, crop.shape[1], crop.shape[0]),
                'global_bbox': (x1, y1, x2, y2),
                'fallback': True,
                'method': 'fallback',
                'time_ms': round(elapsed, 2),
            }
            if return_steps:
                result['steps'] = steps
            return result

        if return_steps:
            vis = crop.copy()
            cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
            steps['3_best_contour'] = vis

        # --- Step 4: 精确裁剪 (外接矩形) ---
        rx, ry, rw, rh = cv2.boundingRect(contour)
        # 上下左右各扩展一点, 保留边框
        pad_x = int(rw * 0.05)
        pad_y = int(rh * 0.08)
        rx1 = max(0, rx - pad_x)
        ry1 = max(0, ry - pad_y)
        rx2 = min(crop.shape[1], rx + rw + pad_x)
        ry2 = min(crop.shape[0], ry + rh + pad_y)

        precise_crop = crop[ry1:ry2, rx1:rx2].copy()
        if return_steps:
            steps['4_precise_crop'] = precise_crop.copy()

        # --- Step 5: 透视矫正 ---
        corrected = self._perspective_correct(crop, contour)
        if return_steps:
            steps['5_corrected'] = corrected.copy()

        # --- Step 6: Resize 到标准尺寸 ---
        plate_resized = cv2.resize(corrected, (self.target_w, self.target_h),
                                   interpolation=cv2.INTER_CUBIC)
        if return_steps:
            steps['6_resized'] = plate_resized.copy()

        elapsed = (time.perf_counter() - t0) * 1000

        global_bbox = (x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2)

        result = {
            'plate_color': precise_crop,
            'plate_corrected': corrected,
            'plate_resized': plate_resized,
            'plate_type': plate_type,
            'precise_bbox': (rx1, ry1, rx2, ry2),
            'global_bbox': global_bbox,
            'fallback': False,
            'method': method,
            'time_ms': round(elapsed, 2),
        }
        if return_steps:
            result['steps'] = steps
        return result


# ===========================================================================
# 工具函数
# ===========================================================================
def interactive_select_roi(image: np.ndarray) -> Tuple[int, int, int, int]:
    """鼠标框选ROI"""
    clone = image.copy()
    # 缩放到屏幕可显示
    max_dim = 1200
    scale = 1.0
    h, w = clone.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        clone = cv2.resize(clone, None, fx=scale, fy=scale)

    roi = cv2.selectROI("Select ROI (drag mouse, then press ENTER)",
                        clone, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # 还原到原始坐标
    x, y, w, h = roi
    x1 = int(x / scale)
    y1 = int(y / scale)
    x2 = int((x + w) / scale)
    y2 = int((y + h) / scale)
    return (x1, y1, x2, y2)


def expand_roi(roi: Tuple[int, int, int, int],
               ratio: float = 0.5,
               img_shape: Tuple[int, int] = None) -> Tuple[int, int, int, int]:
    """
    将精确bbox向外扩展一定比例, 模拟"粗ROI"

    Args:
        roi: (x1, y1, x2, y2) 精确bbox
        ratio: 扩展比例(每边), 0.5 表示宽高各扩50%
        img_shape: (h, w) 用于裁剪边界
    """
    x1, y1, x2, y2 = roi
    w = x2 - x1
    h = y2 - y1
    dx = int(w * ratio)
    dy = int(h * ratio)
    nx1, ny1 = x1 - dx, y1 - dy
    nx2, ny2 = x2 + dx, y2 + dy
    if img_shape is not None:
        ih, iw = img_shape[:2]
        nx1, ny1 = max(0, nx1), max(0, ny1)
        nx2, ny2 = min(iw, nx2), min(ih, ny2)
    return (nx1, ny1, nx2, ny2)


def save_steps_visualization(steps: dict, prefix: str, output_dir: str):
    """将中间步骤保存为图片"""
    os.makedirs(output_dir, exist_ok=True)
    for name, img in steps.items():
        if img is None:
            continue
        path = os.path.join(output_dir, f'{prefix}_{name}.jpg')
        if len(img.shape) == 2:
            cv2.imwrite(path, img)
        else:
            cv2.imwrite(path, img)


def make_comparison_image(original: np.ndarray,
                          roi: Tuple[int, int, int, int],
                          simple_result: dict,
                          precise_result: dict) -> np.ndarray:
    """
    生成对比可视化图: 原图(标ROI) | 方案1结果 | 方案2结果
    """
    vis = original.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(vis, 'ROI', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2)

    # 如果有精确bbox, 也画出来
    if 'global_bbox' in precise_result and not precise_result.get('fallback'):
        gx1, gy1, gx2, gy2 = precise_result['global_bbox']
        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        cv2.putText(vis, f"Precise ({precise_result.get('plate_type', '?')})",
                    (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # 组装对比图
    target_h = 120
    parts = [vis]

    # 方案1结果
    if 'plate_color' in simple_result:
        p1 = simple_result['plate_color']
        p1_h = target_h
        p1_w = int(p1.shape[1] * target_h / p1.shape[0])
        p1_resized = cv2.resize(p1, (p1_w, p1_h))
        label1 = np.zeros((30, p1_w, 3), dtype=np.uint8)
        cv2.putText(label1, f"Simple ({simple_result['time_ms']}ms)",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panel1 = np.vstack([label1, p1_resized])
        parts.append(panel1)

    # 方案2结果
    for key in ['plate_color', 'plate_corrected']:
        if key in precise_result:
            p2 = precise_result[key]
            p2_h = target_h
            p2_w = int(p2.shape[1] * target_h / p2.shape[0])
            if p2_w < 10:
                continue
            p2_resized = cv2.resize(p2, (p2_w, p2_h))
            tag = 'Precise' if key == 'plate_color' else 'Corrected'
            label2 = np.zeros((30, p2_w, 3), dtype=np.uint8)
            cv2.putText(label2, f"{tag} ({precise_result['time_ms']}ms)",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panel2 = np.vstack([label2, p2_resized])
            parts.append(panel2)

    return vis  # 返回标注后的原图


# ===========================================================================
# 评估: 在 CCPD 数据集上测试
# ===========================================================================
def eval_ccpd(data_dir: str, mode: str = 'both', num_samples: int = 200,
              output_dir: str = None):
    """
    用 CCPD 标注的 bbox 模拟固定ROI场景, 对比两种方案的裁剪效果

    原理: CCPD标注提供了车牌的精确bbox, 我们将其扩展50%作为"粗ROI",
    然后分别用方案1和方案2处理, 评估方案2能否在粗ROI中重新找到精确位置

    评估指标: IoU (方案2找到的bbox vs 标注bbox)
    """
    ann_path = os.path.join(data_dir, 'test.json')
    img_dir = os.path.join(data_dir, 'test', 'images')

    with open(ann_path) as f:
        ann = json.load(f)

    # 建立 image_id -> annotation 的映射
    id2ann = {}
    for a in ann['annotations']:
        id2ann[a['image_id']] = a

    simple_proc = SimpleROIPreprocessor()
    precise_proc = PreciseROIPreprocessor()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    simple_times = []
    precise_times = []
    ious = []
    fallback_count = 0
    total = min(num_samples, len(ann['images']))

    print(f'\n{"="*60}')
    print(f'CCPD 评估: {total} 张图片')
    print(f'数据目录: {data_dir}')
    print(f'{"="*60}\n')

    for idx, img_info in enumerate(ann['images'][:total]):
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            continue

        # 标注 bbox (COCO格式: x, y, w, h)
        a = id2ann.get(img_info['id'])
        if a is None:
            continue
        bx, by, bw, bh = a['bbox']
        gt_bbox = (int(bx), int(by), int(bx + bw), int(by + bh))

        # 模拟粗ROI: 扩展50%
        coarse_roi = expand_roi(gt_bbox, ratio=0.5, img_shape=image.shape)

        # 方案1
        if mode in ('simple', 'both'):
            r1 = simple_proc.process(image, coarse_roi)
            simple_times.append(r1['time_ms'])

        # 方案2
        if mode in ('precise', 'both'):
            r2 = precise_proc.process(image, coarse_roi)
            precise_times.append(r2['time_ms'])

            if r2.get('fallback'):
                fallback_count += 1
            else:
                # 计算 IoU
                pred_bbox = r2['global_bbox']
                iou = compute_iou(gt_bbox, pred_bbox)
                ious.append(iou)

        # 保存部分可视化
        if output_dir and idx < 20:
            vis = image.copy()
            x1, y1, x2, y2 = coarse_roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            gx1, gy1, gx2, gy2 = gt_bbox
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

            if mode in ('precise', 'both') and not r2.get('fallback'):
                px1, py1, px2, py2 = r2['global_bbox']
                cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 0), 2)

            cv2.imwrite(os.path.join(output_dir, f'eval_{idx:04d}.jpg'), vis)

            # 保存方案2裁剪结果
            if mode in ('precise', 'both'):
                cv2.imwrite(
                    os.path.join(output_dir, f'eval_{idx:04d}_precise.jpg'),
                    r2.get('plate_corrected', r2.get('plate_color', np.zeros((24,94,3), dtype=np.uint8)))
                )

        if (idx + 1) % 50 == 0:
            print(f'  进度: {idx+1}/{total}')

    # 统计结果
    print(f'\n{"="*60}')
    print(f'评估结果')
    print(f'{"="*60}')

    if simple_times:
        print(f'\n方案1 (SimpleROI):')
        print(f'  平均耗时: {np.mean(simple_times):.2f} ms')
        print(f'  最大耗时: {np.max(simple_times):.2f} ms')

    if precise_times:
        print(f'\n方案2 (PreciseROI):')
        print(f'  平均耗时: {np.mean(precise_times):.2f} ms')
        print(f'  最大耗时: {np.max(precise_times):.2f} ms')
        print(f'  回退次数: {fallback_count}/{total} '
              f'({fallback_count/total*100:.1f}%)')

    if ious:
        ious = np.array(ious)
        print(f'\n精确定位 IoU 统计:')
        print(f'  平均 IoU:  {ious.mean():.3f}')
        print(f'  中位 IoU:  {np.median(ious):.3f}')
        print(f'  IoU > 0.7: {(ious > 0.7).sum()}/{len(ious)} '
              f'({(ious > 0.7).mean()*100:.1f}%)')
        print(f'  IoU > 0.5: {(ious > 0.5).sum()}/{len(ious)} '
              f'({(ious > 0.5).mean()*100:.1f}%)')

    if output_dir:
        print(f'\n可视化结果保存在: {output_dir}')

    return {
        'simple_times': simple_times,
        'precise_times': precise_times,
        'ious': ious.tolist() if len(ious) > 0 else [],
        'fallback_rate': fallback_count / total if total > 0 else 0,
    }


def compute_iou(box1, box2):
    """计算两个bbox的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# ===========================================================================
# Demo: 在 test_pic 图片上演示
# ===========================================================================
def run_demo(test_pic_dir: str, output_dir: str):
    """
    在 test_pic 图片上演示两种方案

    先用 HyperLPR3 获取车牌精确位置, 然后:
      1. 用精确bbox直接做方案1 (模拟"ROI划得很精确")
      2. 扩展50%后做方案2 (模拟"ROI划得比较粗")
    """
    os.makedirs(output_dir, exist_ok=True)

    # 尝试导入 HyperLPR3 获取车牌位置
    try:
        import hyperlpr3 as lpr3
        catcher = lpr3.LicensePlateCatcher()
        has_lpr = True
    except ImportError:
        print('Warning: HyperLPR3 not available, using hardcoded ROIs')
        has_lpr = False

    simple_proc = SimpleROIPreprocessor()
    precise_proc = PreciseROIPreprocessor()

    image_files = sorted([
        f for f in os.listdir(test_pic_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        print(f'No images found in {test_pic_dir}')
        return

    print(f'\n{"="*60}')
    print(f'Demo: 两种ROI预处理方案对比')
    print(f'图片目录: {test_pic_dir}')
    print(f'输出目录: {output_dir}')
    print(f'{"="*60}\n')

    for fname in image_files:
        img_path = os.path.join(test_pic_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue

        stem = Path(fname).stem
        print(f'\n--- {fname} ({image.shape[1]}x{image.shape[0]}) ---')

        # 获取车牌位置
        if has_lpr:
            results = catcher(image)
            if not results:
                print(f'  HyperLPR3 未检测到车牌, 跳过')
                continue
            plate_text, conf, ptype, bbox = results[0]
            bx1, by1, bx2, by2 = [int(v) for v in bbox]
            gt_bbox = (bx1, by1, bx2, by2)
            print(f'  HyperLPR3 检测: {plate_text} (conf={conf:.3f})')
            print(f'  精确 bbox: {gt_bbox}')
        else:
            # 回退: 取图像下半部分中间区域作为ROI
            h, w = image.shape[:2]
            gt_bbox = (w // 4, h // 2, w * 3 // 4, h * 3 // 4)
            print(f'  使用默认ROI: {gt_bbox}')

        # ===== 方案1: 精确ROI (模拟ROI划得很准) =====
        print(f'\n  [方案1] SimpleROI (精确ROI裁剪):')
        r1 = simple_proc.process(image, gt_bbox, return_steps=True)
        print(f'    耗时: {r1["time_ms"]:.2f} ms')
        print(f'    裁剪尺寸: {r1["plate_color"].shape}')
        print(f'    标准化尺寸: {r1["plate_resized"].shape}')

        # 保存步骤
        save_steps_visualization(
            r1.get('steps', {}), f'{stem}_simple', output_dir
        )
        cv2.imwrite(
            os.path.join(output_dir, f'{stem}_simple_final.jpg'),
            r1['plate_resized']
        )

        # ===== 方案2: 粗ROI → 精确定位 =====
        coarse_roi = expand_roi(gt_bbox, ratio=0.5, img_shape=image.shape)
        print(f'\n  [方案2] PreciseROI (粗ROI→精确定位):')
        print(f'    粗ROI: {coarse_roi}')
        r2 = precise_proc.process(image, coarse_roi, return_steps=True)
        print(f'    耗时: {r2["time_ms"]:.2f} ms')
        print(f'    车牌颜色: {r2.get("plate_type", "unknown")}')
        print(f'    是否回退: {r2.get("fallback", False)}')

        if not r2.get('fallback'):
            iou = compute_iou(gt_bbox, r2['global_bbox'])
            print(f'    精确定位bbox: {r2["global_bbox"]}')
            print(f'    vs 标注bbox IoU: {iou:.3f}')

        # 保存步骤
        save_steps_visualization(
            r2.get('steps', {}), f'{stem}_precise', output_dir
        )
        cv2.imwrite(
            os.path.join(output_dir, f'{stem}_precise_final.jpg'),
            r2['plate_resized']
        )
        if 'plate_corrected' in r2:
            cv2.imwrite(
                os.path.join(output_dir, f'{stem}_precise_corrected.jpg'),
                r2['plate_corrected']
            )

        # ===== 对比可视化 =====
        vis = image.copy()
        # 粗ROI (黄色)
        cx1, cy1, cx2, cy2 = coarse_roi
        cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 255, 255), 3)
        cv2.putText(vis, 'Coarse ROI', (cx1, cy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        # 精确bbox (蓝色 = 标注)
        cv2.rectangle(vis, (gt_bbox[0], gt_bbox[1]),
                      (gt_bbox[2], gt_bbox[3]), (255, 0, 0), 2)
        cv2.putText(vis, 'GT bbox', (gt_bbox[0], gt_bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # 方案2定位 (绿色)
        if not r2.get('fallback'):
            px1, py1, px2, py2 = r2['global_bbox']
            cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(vis, f'Precise ({r2["plate_type"]})',
                        (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, f'{stem}_comparison.jpg'), vis)

    print(f'\n所有结果已保存到: {output_dir}')


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='车牌ROI裁剪与预处理 (方案1: 简单裁剪 / 方案2: 精确定位)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='输入图片路径')
    group.add_argument('--demo', action='store_true',
                       help='在 test_pic 图片上演示')
    group.add_argument('--eval-ccpd', type=str, metavar='DIR',
                       help='在CCPD数据集上评估, 指定 combined 目录')

    parser.add_argument('--mode', type=str, default='both',
                        choices=['simple', 'precise', 'both'],
                        help='方案选择 (default: both)')
    parser.add_argument('--roi', type=str, default=None,
                        help='ROI坐标: x1,y1,x2,y2')
    parser.add_argument('--interactive', action='store_true',
                        help='交互式鼠标框选ROI')
    parser.add_argument('--output-dir', type=str, default='roi_demo_output',
                        help='输出目录 (default: roi_demo_output)')
    parser.add_argument('--num-samples', type=int, default=200,
                        help='CCPD评估样本数 (default: 200)')
    parser.add_argument('--test-pic-dir', type=str, default=None,
                        help='test_pic 目录 (demo模式)')

    args = parser.parse_args()

    if args.demo:
        # 自动找 test_pic 目录
        test_pic = args.test_pic_dir
        if test_pic is None:
            candidates = [
                'test_pic',
                '../test_pic',
                os.path.join(os.path.dirname(__file__), '..', '..', 'test_pic'),
            ]
            for c in candidates:
                if os.path.isdir(c):
                    test_pic = c
                    break
        if test_pic is None:
            print('Error: 找不到 test_pic 目录, 请用 --test-pic-dir 指定')
            sys.exit(1)

        run_demo(test_pic, args.output_dir)

    elif args.eval_ccpd:
        eval_ccpd(args.eval_ccpd, mode=args.mode,
                  num_samples=args.num_samples,
                  output_dir=args.output_dir)

    elif args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f'Error: 无法读取图片 {args.image}')
            sys.exit(1)

        # 获取ROI
        if args.interactive:
            roi = interactive_select_roi(image)
            print(f'选择的ROI: {roi}')
        elif args.roi:
            roi = tuple(int(x) for x in args.roi.split(','))
        else:
            print('Error: 需要 --roi 或 --interactive 指定ROI')
            sys.exit(1)

        os.makedirs(args.output_dir, exist_ok=True)
        stem = Path(args.image).stem

        if args.mode in ('simple', 'both'):
            proc = SimpleROIPreprocessor()
            result = proc.process(image, roi, return_steps=True)
            print(f'\n[方案1] SimpleROI: {result["time_ms"]:.2f} ms')
            save_steps_visualization(
                result.get('steps', {}), f'{stem}_simple', args.output_dir
            )
            cv2.imwrite(
                os.path.join(args.output_dir, f'{stem}_simple_final.jpg'),
                result['plate_resized']
            )

        if args.mode in ('precise', 'both'):
            proc = PreciseROIPreprocessor()
            result = proc.process(image, roi, return_steps=True)
            print(f'\n[方案2] PreciseROI: {result["time_ms"]:.2f} ms')
            print(f'  车牌颜色: {result.get("plate_type", "unknown")}')
            print(f'  回退: {result.get("fallback", False)}')
            save_steps_visualization(
                result.get('steps', {}), f'{stem}_precise', args.output_dir
            )
            cv2.imwrite(
                os.path.join(args.output_dir, f'{stem}_precise_final.jpg'),
                result['plate_resized']
            )

        print(f'\n结果保存在: {args.output_dir}')


if __name__ == '__main__':
    main()
