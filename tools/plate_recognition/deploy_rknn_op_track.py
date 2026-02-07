#!/usr/bin/env python3
"""
解决解码问题
E2E_HZTK RK3588 NPU 部署推理脚本

支持：单张图片、图片目录、本地视频文件、RTSP视频流

使用方法:
    python deploy_rknn_board_dis.py --image ./test.jpg
    python deploy_rknn_board_dis.py --image-dir ./test_pic --output-dir ./results
    python deploy_rknn_board_dis.py --rtsp "rtsp://admin:password@ip:554/stream1"
    python deploy_rknn_board_dis.py --video ./test_video.mp4
    python deploy_rknn_board_dis.py --benchmark
"""

import cv2
import numpy as np
import math
import time
import os
import argparse
import sys

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from rknnlite.api import RKNNLite

    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    print("Warning: rknnlite not available, running in debug mode")

# 车牌字符集
PLATE_CHARS = [
    "blank", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "云", "京", "冀", "吉", "学", "宁", "川", "挂", "新", "晋", "桂", "民",
    "沪", "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤", "航", "苏",
    "蒙", "藏", "警", "豫", "贵", "赣", "辽", "鄂", "闽", "陕", "青", "鲁",
    "黑", "领", "使", "澳",
]

PLATE_TYPES = {0: "蓝牌", 1: "绿牌", 2: "黄牌"}


# ============================================================================
# 车牌格式校验 (用于过滤明显错误的识别结果)
# ============================================================================

# 省份简称 (31个省+军/警/学/领/使/港/澳)
PROVINCES = set(
    "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁"
    "军警学领使港澳"
)

# 车牌字母位 (位置1): A-Z, 无 I/O
PLATE_LETTERS = set("ABCDEFGHJKLMNPQRSTUVWXYZ")

# 车牌号码/字母位 (位置2+): 0-9 A-Z, 无 I/O
PLATE_ALPHANUMS = set("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")

# 新能源尾字: D(纯电) F(混动)
NEW_ENERGY_SUFFIX = set("DF")


def validate_plate_format(plate_text: str, confidence: float) -> dict:
    """
    车牌格式校验函数
    
    基于中国车牌结构规则校验识别结果:
      - 普通蓝/黄牌: 省 + 字母 + 5位(字母/数字) = 7位
      - 新能源绿牌: 省 + 字母 + 6位 = 8位
    
    Args:
        plate_text: 识别的车牌文本
        confidence: 原始置信度
    
    Returns:
        {
            'text': 原文本,
            'confidence': 原置信度,
            'valid': bool,
            'adjusted_confidence': 校验后置信度,
            'issues': 问题列表,
            'level': 'accepted' / 'suspicious' / 'rejected',
        }
    """
    issues = []
    penalty = 0.0  # 置信度惩罚

    # --- Level 1: 长度校验 ---
    n = len(plate_text)
    if n == 0:
        return {
            'text': plate_text, 'confidence': confidence,
            'valid': False, 'adjusted_confidence': 0.0,
            'issues': ['空结果'], 'level': 'rejected',
        }
    if n < 7:
        issues.append(f'长度过短({n}<7)')
        penalty += 0.5
    elif n > 8:
        issues.append(f'长度过长({n}>8)')
        penalty += 0.4

    # --- Level 2: 位置0 — 省份汉字 ---
    if n >= 1:
        if plate_text[0] not in PROVINCES:
            issues.append(f'位置0非省份字符({plate_text[0]})')
            penalty += 0.3

    # --- Level 3: 位置1 — 字母 ---
    if n >= 2:
        if plate_text[1] not in PLATE_LETTERS:
            issues.append(f'位置1非字母({plate_text[1]})')
            penalty += 0.2

    # --- Level 4: 位置2+ — 字母/数字 ---
    if n >= 3:
        for i in range(2, n):
            c = plate_text[i]
            if c not in PLATE_ALPHANUMS:
                issues.append(f'位置{i}非法字符({c})')
                penalty += 0.15
                break  # 只报第一个

    # --- 新能源格式检查 ---
    if n == 8:
        has_ne_prefix = plate_text[2] in NEW_ENERGY_SUFFIX
        has_ne_suffix = plate_text[7] in NEW_ENERGY_SUFFIX
        if not has_ne_prefix and not has_ne_suffix:
            issues.append('8位但无D/F标记')
            penalty += 0.1

    # --- 综合判定 ---
    adjusted = max(0.0, confidence - penalty)

    if not issues:
        level = 'accepted'
        valid = True
    elif penalty >= 0.4:
        level = 'rejected'
        valid = False
    else:
        level = 'suspicious'
        valid = True  # 宽松模式: suspicious 也通过

    return {
        'text': plate_text,
        'confidence': confidence,
        'valid': valid,
        'adjusted_confidence': round(adjusted, 4),
        'issues': issues,
        'level': level,
    }


def is_valid_plate_format(plate_text: str) -> bool:
    """快速判断车牌格式是否合法"""
    n = len(plate_text)
    if n < 7 or n > 8:
        return False
    if plate_text[0] not in PROVINCES:
        return False
    if n >= 2 and plate_text[1] not in PLATE_LETTERS:
        return False
    return True


# ------------------*****-----------------------------
# ------------------*****-----------------------------
class ROIProcessor:
    """ROI区域处理器 - 确保每辆车只有一个ID"""

    def __init__(self, roi_coords, history_frames=10):
        """
        参数:
            roi_coords: (x1, y1, x2, y2) ROI区域坐标
            history_frames: 历史帧数，用于众数计算
        """
        self.roi_coords = roi_coords
        self.history_frames = history_frames

        # 跟踪数据结构
        self.tracked_cars = {}  # {track_id: {'enter_frame': X, 'exit_frame': Y, 'plates': [], ...}}
        self.active_tracks = {}  # 当前在ROI内的车辆 {track_id: {'last_pos': (x,y), 'enter_frame': X, 'last_seen': frame}}
        self.next_track_id = 0

        # 新增：上一帧的检测结果缓存，用于连续性匹配
        self.prev_frame_detections = {}  # {track_id: {'bbox': bbox, 'plate': plate, 'frame': frame}}
        self.frame_count = 0

        # 匹配参数
        self.MAX_DISTANCE = 500  # 最大匹配距离（像素）
        self.MAX_SKIP_FRAMES = 5  # 最大跳帧数（允许丢失的帧数）

    def is_in_roi(self, bbox):
        """判断车牌是否在ROI区域内"""
        x1, y1, x2, y2 = bbox
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi_coords

        # 检查车牌中心点是否在ROI内
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        return (roi_x1 <= center_x <= roi_x2 and
                roi_y1 <= center_y <= roi_y2)

    def get_vehicle_center(self, bbox):
        """获取车辆中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def calculate_distance(self, bbox1, bbox2):
        """计算两个bbox中心点的距离"""
        center1 = self.get_vehicle_center(bbox1)
        center2 = self.get_vehicle_center(bbox2)

        return math.sqrt((center1[0] - center2[0]) ** 2 +
                         (center1[1] - center2[1]) ** 2)

    def plates_match(self, plate1, plate2):
        """判断两个车牌是否匹配"""
        if not plate1 or not plate2:
            return False

        # 完全匹配
        if plate1 == plate2:
            return True

        # 允许少量字符差异（考虑到识别误差）
        # 例如：皖AK329K 和 皖AK329 视为匹配
        min_len = min(len(plate1), len(plate2))
        if min_len >= 5:  # 至少有5个字符才进行模糊匹配
            match_count = sum(1 for i in range(min_len) if plate1[i] == plate2[i])
            return match_count >= min_len - 1  # 允许1个字符差异

        return False

    def find_best_match(self, plate_number, bbox, current_frame):
        """
        为当前检测寻找最佳匹配的跟踪ID
        返回: (track_id, match_score)
        """
        best_id = None
        best_score = -1

        # 优先匹配活动中的车辆
        for track_id, track_info in self.active_tracks.items():
            if track_id not in self.tracked_cars:
                continue

            # 跳过太久没见的跟踪（可能已离开）
            last_seen = track_info.get('last_seen', 0)
            if current_frame - last_seen > self.MAX_SKIP_FRAMES:
                continue

            # 计算匹配分数
            score = 0

            # 1. 位置匹配分数（基于距离）
            if track_id in self.prev_frame_detections:
                prev_bbox = self.prev_frame_detections[track_id]['bbox']
                distance = self.calculate_distance(bbox, prev_bbox)

                if distance < self.MAX_DISTANCE:
                    # 距离越近，分数越高
                    distance_score = max(0, 1 - distance / self.MAX_DISTANCE)
                    score += distance_score * 0.6  # 位置权重60%

            # 2. 车牌匹配分数
            if self.tracked_cars[track_id]['plates']:
                # 获取该跟踪的历史车牌众数
                from collections import Counter
                history_plates = self.tracked_cars[track_id]['plates']
                plate_counter = Counter(history_plates)
                if plate_counter:
                    most_common_plate, _ = plate_counter.most_common(1)[0]

                    if self.plates_match(plate_number, most_common_plate):
                        score += 0.4  # 车牌匹配权重40%

            # 更新最佳匹配
            if score > best_score:
                best_score = score
                best_id = track_id

        # 如果找到足够好的匹配（分数>0.3）
        if best_score > 0.3:
            return best_id, best_score

        return None, 0

    def track_vehicle(self, plate_number, bbox, frame_count, confidence):
        """
        跟踪车辆进出ROI过程 - 修复版
        确保每辆车在ROI过程中只有一个ID
        返回: (track_id, action, is_active)
        """
        # 计算车辆中心
        center_x, center_y = self.get_vehicle_center(bbox)

        # 检查是否在ROI内
        in_roi_now = self.is_in_roi(bbox)

        # 如果不是在ROI内，直接返回
        if not in_roi_now:
            # 检查是否有活动跟踪应该结束
            for track_id in list(self.active_tracks.keys()):
                track_info = self.active_tracks[track_id]
                last_seen = track_info.get('last_seen', 0)

                # 如果超过MAX_SKIP_FRAMES帧没看到，认为已离开
                if frame_count - last_seen > self.MAX_SKIP_FRAMES:
                    del self.active_tracks[track_id]

            return None, 'none', False

        # 寻找最佳匹配
        matched_id, match_score = self.find_best_match(plate_number, bbox, frame_count)

        action = 'update'
        is_active = True

        if matched_id is not None:
            # 更新现有跟踪
            self.active_tracks[matched_id]['last_pos'] = (center_x, center_y)
            self.active_tracks[matched_id]['last_seen'] = frame_count

            # 如果是首次检测到该车辆在ROI内，记录为进入
            if matched_id not in self.tracked_cars:
                self.tracked_cars[matched_id] = {
                    'enter_frame': frame_count,
                    'exit_frame': frame_count,
                    'plates': [],
                    'positions': [],
                    'confidences': []
                }
                action = 'enter'
                print(f"[DEBUG] Track {matched_id}: NEW ENTRY with score {match_score:.2f}")
            else:
                # 检查是否为重新进入（之前离开过）
                if matched_id not in self.active_tracks:
                    action = 'reenter'
                    print(f"[DEBUG] Track {matched_id}: RE-ENTER with score {match_score:.2f}")

            # 添加当前检测结果
            self.tracked_cars[matched_id]['plates'].append(plate_number)
            self.tracked_cars[matched_id]['positions'].append(bbox)
            self.tracked_cars[matched_id]['confidences'].append(confidence)
            self.tracked_cars[matched_id]['exit_frame'] = frame_count

            # 保存到上一帧缓存
            self.prev_frame_detections[matched_id] = {
                'bbox': bbox,
                'plate': plate_number,
                'frame': frame_count
            }

        elif in_roi_now:
            # 新车辆进入ROI
            matched_id = self.next_track_id
            self.next_track_id += 1

            # 添加到活动跟踪
            self.active_tracks[matched_id] = {
                'last_pos': (center_x, center_y),
                'enter_frame': frame_count,
                'last_seen': frame_count
            }

            # 初始化跟踪记录
            self.tracked_cars[matched_id] = {
                'enter_frame': frame_count,
                'exit_frame': frame_count,
                'plates': [plate_number],
                'positions': [bbox],
                'confidences': [confidence]
            }

            # 保存到上一帧缓存
            self.prev_frame_detections[matched_id] = {
                'bbox': bbox,
                'plate': plate_number,
                'frame': frame_count
            }

            action = 'enter'
            is_active = True
            print(f"[DEBUG] Track {matched_id}: COMPLETELY NEW VEHICLE")

        # 清理旧的缓存
        self.cleanup_old_cache(frame_count)

        return matched_id, action, is_active

    def cleanup_old_cache(self, current_frame):
        """清理过期的缓存"""
        # 清理上一帧缓存
        to_remove = []
        for track_id, det_info in self.prev_frame_detections.items():
            if current_frame - det_info['frame'] > self.MAX_SKIP_FRAMES:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.prev_frame_detections[track_id]

        # 清理长时间未活动的跟踪
        to_remove = []
        for track_id, track_info in self.active_tracks.items():
            last_seen = track_info.get('last_seen', 0)
            if current_frame - last_seen > self.MAX_SKIP_FRAMES * 2:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.active_tracks[track_id]

    def get_final_result(self, track_id):
        """获取车辆的最终识别结果（众数）"""
        if track_id not in self.tracked_cars:
            return None

        track_data = self.tracked_cars[track_id]

        # 如果只有一次检测，直接返回
        if len(track_data['plates']) == 1:
            return {
                'plate': track_data['plates'][0],
                'confidence': track_data['confidences'][0],
                'enter_frame': track_data['enter_frame'],
                'exit_frame': track_data['exit_frame'],
                'enter_position': track_data['positions'][0],
                'exit_position': track_data['positions'][-1],
                'mode_confidence': 1.0,
                'total_frames': 1
            }

        # 计算众数
        from collections import Counter
        plate_counter = Counter(track_data['plates'])
        most_common_plate, count = plate_counter.most_common(1)[0]

        # 计算平均置信度
        total_confidence = sum(track_data['confidences'])
        avg_confidence = total_confidence / len(track_data['confidences'])

        # 众数置信度
        mode_confidence = count / len(track_data['plates'])

        return {
            'plate': most_common_plate,
            'confidence': avg_confidence,
            'enter_frame': track_data['enter_frame'],
            'exit_frame': track_data['exit_frame'],
            'enter_position': track_data['positions'][0],
            'exit_position': track_data['positions'][-1],
            'mode_confidence': mode_confidence,
            'total_frames': len(track_data['plates'])
        }

    def get_active_track_count(self):
        """获取当前在ROI内的车辆数量"""
        return len(self.active_tracks)

    def get_completed_track_count(self):
        """获取已完成的跟踪数量"""
        return len([tid for tid in self.tracked_cars.keys() if tid not in self.active_tracks])

# ------------------*****-----------------------------

def get_chinese_font(size=20):
    """获取中文字体"""
    if not PIL_AVAILABLE:
        return None

    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue

    try:
        return ImageFont.load_default()
    except Exception:
        return None


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Letterbox 预处理"""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def nms(boxes, scores, iou_thresh=0.5):
    """NMS"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


def create_masked_image(frame, roi_coords):
    """
    创建掩码图像：只保留ROI区域，其他区域归0

    参数:
        frame: 原始图像
        roi_coords: (x1, y1, x2, y2) ROI坐标

    返回:
        masked_frame: 掩码后的图像（只保留ROI区域）
    """
    # 创建掩码图像副本
    masked_frame = np.zeros_like(frame)

    # 确保ROI坐标在图像范围内
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi_coords

    # 裁剪坐标到图像范围内
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # 只保留ROI区域的内容
    if x2 > x1 and y2 > y1:
        masked_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    return masked_frame


class PlateRecognizerRKNN:
    """基于 RKNN 的车牌识别器 (NHWC 输入格式)"""

    def __init__(self, model_dir: str = "./"):
        self.model_dir = model_dir
        self.det_size = 640

        if not RKNN_AVAILABLE:
            print("Warning: RKNN not available, methods will return empty results")
            self.det_rknn = None
            self.rec_rknn = None
            self.cls_rknn = None
            return

        det_path = os.path.join(model_dir, 'hztk_det.rknn')
        rec_path = os.path.join(model_dir, 'hztk_rec.rknn')
        cls_path = os.path.join(model_dir, 'hztk_cls.rknn')

        print("Loading RKNN models...")

        self.det_rknn = RKNNLite()
        ret = self.det_rknn.load_rknn(det_path)
        if ret != 0:
            raise RuntimeError(f"Load det model failed: {det_path}")
        self.det_rknn.init_runtime()
        print(f"  Det model loaded: {det_path}")

        self.rec_rknn = RKNNLite()
        ret = self.rec_rknn.load_rknn(rec_path)
        if ret != 0:
            raise RuntimeError(f"Load rec model failed: {rec_path}")
        self.rec_rknn.init_runtime()
        print(f"  Rec model loaded: {rec_path}")

        self.cls_rknn = RKNNLite()
        ret = self.cls_rknn.load_rknn(cls_path)
        if ret != 0:
            raise RuntimeError(f"Load cls model failed: {cls_path}")
        self.cls_rknn.init_runtime()
        print(f"  Cls model loaded: {cls_path}")

        print("PlateRecognizerRKNN initialized (NHWC format)")

    def preprocess_det(self, image: np.ndarray) -> tuple:
        """检测模型预处理 - NHWC 格式"""
        padded, ratio, (dw, dh) = letterbox(image, (self.det_size, self.det_size))

        # BGR -> RGB, 归一化, 保持 NHWC
        img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)  # [1, H, W, C] - NHWC

        return img, ratio, (dw, dh)

    def postprocess_det(self, output: np.ndarray, ratio: tuple,
                        pad_size: tuple, orig_shape: tuple,
                        conf_thresh: float = 0.5) -> list:
        """检测后处理"""
        output = output.squeeze()
        dw, dh = pad_size
        orig_h, orig_w = orig_shape

        mask = output[:, 4] > conf_thresh
        filtered = output[mask]

        if len(filtered) == 0:
            return []

        cx, cy, bw, bh = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
        confs = filtered[:, 4]

        x1 = (cx - bw / 2 - dw) / ratio[0]
        y1 = (cy - bh / 2 - dh) / ratio[1]
        x2 = (cx + bw / 2 - dw) / ratio[0]
        y2 = (cy + bh / 2 - dh) / ratio[1]

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        valid = (x2 - x1) > 10
        x1, y1, x2, y2, confs = x1[valid], y1[valid], x2[valid], y2[valid], confs[valid]

        if len(x1) == 0:
            return []

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(boxes, confs, 0.5)

        detections = []
        for i in keep:
            detections.append((
                int(boxes[i, 0]), int(boxes[i, 1]),
                int(boxes[i, 2]), int(boxes[i, 3]),
                float(confs[i])
            ))

        return detections

    def preprocess_rec(self, plate_img: np.ndarray) -> np.ndarray:
        """识别模型预处理 - NHWC 格式"""
        imgH, imgW = 48, 160
        h, w = plate_img.shape[:2]
        wh_ratio = w / float(h)

        max_wh_ratio = max(wh_ratio, imgW / imgH)
        target_w = int(imgH * max_wh_ratio)
        target_w = max(min(target_w, 160), 48)

        ratio_imgH = math.ceil(imgH * wh_ratio)
        ratio_imgH = max(ratio_imgH, 48)
        resized_w = target_w if ratio_imgH > target_w else int(ratio_imgH)

        resized = cv2.resize(plate_img, (resized_w, imgH))
        resized = resized.astype(np.float32)
        resized = (resized - 127.5) / 127.5

        # Padding, 保持 NHWC
        padded = np.zeros((imgH, imgW, 3), dtype=np.float32)
        padded[:, 0:resized_w, :] = resized

        return np.expand_dims(padded, 0)  # [1, H, W, C] - NHWC

    def decode_plate(self, output: np.ndarray) -> tuple:
        """解码车牌"""
        prod = output.squeeze()
        indices = np.argmax(prod, axis=-1)
        max_probs = np.max(prod, axis=-1)

        chars, confs = [], []
        prev_idx = -1
        for i, idx in enumerate(indices):
            if idx == 0 or idx == prev_idx:
                prev_idx = idx
                continue
            if idx < len(PLATE_CHARS):
                chars.append(PLATE_CHARS[int(idx)])
                confs.append(float(max_probs[i]))
            prev_idx = idx

        return "".join(chars), float(np.mean(confs)) if confs else 0.0

    def preprocess_cls(self, plate_img: np.ndarray) -> np.ndarray:
        """分类模型预处理 - NHWC 格式"""
        img = cv2.resize(plate_img, (96, 96))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, 0)  # [1, H, W, C] - NHWC

    def classify_plate(self, output: np.ndarray) -> tuple:
        """分类解码"""
        output = output.squeeze()
        idx = int(np.argmax(output))
        conf = float(output[idx])
        # wang
        # -
        # return PLATE_TYPES.get(idx, "未知"), conf
        # +
        plate_type = ""
        return plate_type, conf

    def recognize(self, image: np.ndarray, conf_thresh: float = 0.5) -> list:
        """完整识别流程"""
        if self.det_rknn is None:
            return []

        results = []

        # 检测
        det_input, ratio, pad_size = self.preprocess_det(image)
        det_output = self.det_rknn.inference(inputs=[det_input])[0]
        detections = self.postprocess_det(det_output, ratio, pad_size,
                                          image.shape[:2], conf_thresh)

        # 识别每个检测框
        for x1, y1, x2, y2, det_conf in detections:
            plate_img = image[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            # OCR
            rec_input = self.preprocess_rec(plate_img)
            rec_output = self.rec_rknn.inference(inputs=[rec_input])[0]
            plate_number, rec_conf = self.decode_plate(rec_output)

            # 分类
            cls_input = self.preprocess_cls(plate_img)
            cls_output = self.cls_rknn.inference(inputs=[cls_input])[0]
            #plate_type, cls_conf = self.classify_plate(cls_output)

            results.append({
                'bbox': [x1, y1, x2, y2],
                'det_conf': det_conf,
                'plate_number': plate_number,
                'rec_conf': rec_conf,

            })

        return results

    def recognize_with_masked_det(self, original_image: np.ndarray, masked_image: np.ndarray,
                                  conf_thresh: float = 0.5) -> list:
        """
        使用掩码图像进行检测，但使用原始图像进行车牌裁剪和识别

        参数:
            original_image: 原始图像（用于车牌裁剪和识别）
            masked_image: 掩码图像（只保留ROI区域，用于检测）
            conf_thresh: 置信度阈值
        """
        if self.det_rknn is None:
            return []

        results = []

        # 使用掩码图像进行检测
        det_input, ratio, pad_size = self.preprocess_det(masked_image)
        det_output = self.det_rknn.inference(inputs=[det_input])[0]
        detections = self.postprocess_det(det_output, ratio, pad_size,
                                          original_image.shape[:2], conf_thresh)

        # 识别每个检测框（使用原始图像裁剪车牌区域）
        for x1, y1, x2, y2, det_conf in detections:
            # 从原始图像中裁剪车牌区域
            plate_img = original_image[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            # OCR
            rec_input = self.preprocess_rec(plate_img)
            rec_output = self.rec_rknn.inference(inputs=[rec_input])[0]
            plate_number, rec_conf = self.decode_plate(rec_output)

            # 分类
            cls_input = self.preprocess_cls(plate_img)
            cls_output = self.cls_rknn.inference(inputs=[cls_input])[0]
            #plate_type, cls_conf = self.classify_plate(cls_output)

            results.append({
                'bbox': [x1, y1, x2, y2],
                'det_conf': det_conf,
                'plate_number': plate_number,
                'rec_conf': rec_conf,

            })

        return results

    def benchmark(self, num_iters: int = 50, warmup: int = 5) -> dict:
        """性能测试 - NHWC 格式"""
        if self.det_rknn is None:
            return {'error': 'RKNN not available'}

        det_input = np.random.randn(1, self.det_size, self.det_size, 3).astype(np.float32)
        rec_input = np.random.randn(1, 48, 160, 3).astype(np.float32)
        cls_input = np.random.randn(1, 96, 96, 3).astype(np.float32)

        print(f"Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            self.det_rknn.inference(inputs=[det_input])
            self.rec_rknn.inference(inputs=[rec_input])
            self.cls_rknn.inference(inputs=[cls_input])

        print(f"Benchmarking ({num_iters} iterations)...")

        start = time.time()
        for _ in range(num_iters):
            self.det_rknn.inference(inputs=[det_input])
        det_time = (time.time() - start) / num_iters * 1000

        start = time.time()
        for _ in range(num_iters):
            self.rec_rknn.inference(inputs=[rec_input])
        rec_time = (time.time() - start) / num_iters * 1000

        start = time.time()
        for _ in range(num_iters):
            self.cls_rknn.inference(inputs=[cls_input])
        cls_time = (time.time() - start) / num_iters * 1000

        total = det_time + rec_time + cls_time

        return {
            'detection_ms': det_time,
            'recognition_ms': rec_time,
            'classification_ms': cls_time,
            'total_ms': total,
            'fps': 1000 / total,
        }

    def release(self):
        """释放资源"""
        if self.det_rknn:
            self.det_rknn.release()
        if self.rec_rknn:
            self.rec_rknn.release()
        if self.cls_rknn:
            self.cls_rknn.release()


def visualize_result(image: np.ndarray, results: list, font=None) -> np.ndarray:
    """可视化结果"""
    result_img = image.copy()

    for res in results:
        x1, y1, x2, y2 = res['bbox']
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # wang
        # -
        # label = f"{res['plate_number']} ({res['plate_type']})"
        # +
        label = f"{res['plate_number']}"

        if PIL_AVAILABLE and font is not None:
            pil_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            text_y = max(y1 - 30, 5)
            try:
                bbox = draw.textbbox((x1, text_y), label, font=font)
                draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(0, 255, 0))
                draw.text((x1, text_y), label, font=font, fill=(0, 0, 0))
            except Exception:
                draw.text((x1, text_y), label, font=font, fill=(0, 255, 0))
            result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(result_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

    return result_img


# ========== 修复HEVC解码问题的函数 ==========
def robust_video_capture(cap, max_retries=3, skip_frames_on_error=5):
    """
    稳健的视频帧读取函数，专门解决HEVC解码问题

    参数:
        cap: cv2.VideoCapture对象
        max_retries: 最大重试次数
        skip_frames_on_error: 发生错误时跳过的帧数

    返回:
        (ret, frame, error_message)
    """
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            ret, frame = cap.read()
            read_time = (time.time() - start_time) * 1000

            if not ret:
                # 尝试跳过一些帧来清空错误缓冲区
                for _ in range(skip_frames_on_error):
                    cap.grab()
                time.sleep(0.01)
                continue

            # 如果读取时间过长（可能遇到解码问题）
            if read_time > 500:  # 超过500ms认为有问题
                print(f"[HEVC WARN] 帧读取时间过长: {read_time:.1f}ms, 尝试跳过...")
                for _ in range(skip_frames_on_error):
                    cap.grab()
                continue

            # 检查帧是否有效
            if frame is None or frame.size == 0:
                print(f"[HEVC WARN] 读取到空帧，跳过...")
                for _ in range(skip_frames_on_error):
                    cap.grab()
                continue

            return ret, frame, None

        except Exception as e:
            error_msg = f"[HEVC ERROR] 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}"
            print(error_msg)

            # 等待后重试
            time.sleep(0.1 * (attempt + 1))

    return False, None, "无法读取视频帧，已达到最大重试次数"


def reset_video_capture(video_source, is_rtsp=False):
    """
    重置视频捕获对象，解决长时间运行后的解码问题

    新增重置函数，定期重新初始化VideoCapture
    """
    print(f"[RESET] 重置视频捕获对象...")

    if is_rtsp:
        # 对于RTSP，使用TCP传输
        if video_source.startswith('rtsp://'):
            tcp_url = f"{video_source}?rtsp_transport=tcp"
        else:
            tcp_url = video_source

        # 尝试不同的参数组合
        rtsp_options = [
            tcp_url,
            f"{video_source}?rtsp_transport=tcp&buffer_size=1024000",
            f"{video_source}?rtsp_transport=udp",
            video_source  # 原始URL
        ]

        for i, url in enumerate(rtsp_options):
            try:
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小化缓冲区
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5秒超时

                # 测试读取
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"[RESET] RTSP重连成功 (选项{i + 1})")
                    # 如果需要，重置到开头
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return cap
                else:
                    cap.release()
            except Exception as e:
                print(f"[RESET] RTSP重连失败 (选项{i + 1}): {e}")
    else:
        # 本地文件
        try:
            cap = cv2.VideoCapture(video_source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

            ret, test_frame = cap.read()
            if ret:
                # 重置到开头
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("[RESET] 本地视频重连成功")
                return cap
            else:
                cap.release()
        except Exception as e:
            print(f"[RESET] 本地视频重连失败: {e}")

    return None


def process_video_stream(recognizer, video_source, conf_thresh: float = 0.5,
                         display: bool = True, save_dir: str = None,
                         save_all: bool = False, save_interval: int = 1,
                         roi_coords: tuple = (800, 1000, 2800, 2000),
                         history_frames: int = 10, is_rtsp: bool = False):
    """
    通用视频流处理函数 - 只保存进入ROI帧 + 众数识别

    增加HEVC解码错误处理
    """
    # 检查是否有显示器，没有则自动禁用显示
    if display:
        display_available = os.environ.get('DISPLAY') is not None
        if not display_available:
            print("Warning: No display available, disabling GUI")
            display = False

    # 打开视频源
    if is_rtsp:
        print(f"Connecting to RTSP: {video_source}")
        if video_source.startswith('rtsp://'):
            tcp_url = f"{video_source}?rtsp_transport=tcp"
        else:
            tcp_url = video_source

        cap = cv2.VideoCapture(tcp_url, cv2.CAP_FFMPEG)  # 使用TCP URL
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小化缓冲区减少延迟
        video_type = "RTSP流"
    else:
        print(f"Loading local video: {video_source}")
        cap = cv2.VideoCapture(video_source)
        video_type = "本地视频"

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_rtsp else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        if is_rtsp:
            print(f"无法打开RTSP流: {video_source}")
            print("请检查：")
            print("1. 网络是否可达")
            print("2. 用户名和密码是否正确")
            print("3. 摄像头是否支持RTSP且该流存在")
        else:
            print(f"无法打开视频文件: {video_source}")
            print("请检查：")
            print("1. 文件路径是否正确")
            print("2. 文件格式是否支持 (mp4, avi, mov等)")
            print("3. 视频文件是否损坏")
        return

    # 显示视频信息
    print("=" * 60)
    if is_rtsp:
        print(f"RTSP流信息:")
        print(f"  分辨率: {width} x {height}")
    else:
        print(f"视频信息:")
        print(f"  分辨率: {width} x {height}")
        print(f"  帧率: {fps:.2f} FPS")
        print(f"  总帧数: {total_frames}")
    print(f"  ROI区域: {roi_coords}")
    print("=" * 60)
    print("开始处理视频...")
    print(f"  视频类型: {video_type}")
    print(f"  HEVC解码修复: 已启用")
    print(f"  置信度阈值: {conf_thresh}")
    print(f"  保存策略: 只保存车辆进入ROI的第一帧")
    if save_dir:
        print(f"  保存目录: {save_dir}")
    else:
        print(f"  保存目录: 未设置 (使用 -o 或 --output-dir 指定)")
    print("  按 Ctrl+C 退出")
    print("=" * 60)

    font = get_chinese_font(size=48) if PIL_AVAILABLE else None
    frame_count = 0
    fps_list = []
    last_print_time = time.time()
    last_valid_frame = None  # 初始化最后有效帧
    reconnect_count = 0  # 重连计数器，用于指数退避
    reset_interval = 600  # 每600帧重置一次VideoCapture（防止HEVC累积错误）
    hevc_error_count = 0  # HEVC错误计数器
    max_hevc_errors = 10  # 最大HEVC错误次数

    # 修改点4: 添加视频处理统计
    video_stats = {
        'total_reads': 0,
        'successful_reads': 0,
        'hevc_errors': 0,
        'reset_count': 0,
        'last_reset_frame': 0
    }

    # 创建ROI处理器
    roi_processor = ROIProcessor(roi_coords, history_frames=history_frames)

    # 结果记录文件
    result_file = None
    entered_tracks = set()  # 已保存进入图片的跟踪ID
    completed_tracks = set()  # 已完成的跟踪（车辆已离开ROI）

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if is_rtsp:
            # 对于RTSP，使用时间戳作为文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_name = f"rtsp_{timestamp}"
        else:
            # 对于本地视频，使用文件名
            video_name = os.path.splitext(os.path.basename(video_source))[0]

        result_file = os.path.join(save_dir, f"{video_name}_results.txt")

        # 清空或创建结果文件
        with open(result_file, 'a') as f:
            f.write(f"车牌识别结果记录 - {video_name}\n")
            if is_rtsp:
                f.write(f"RTSP地址: {video_source}\n")
            else:
                f.write(f"视频文件: {video_source}\n")
            f.write(f"ROI区域: {roi_coords}\n")
            f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"保存策略: 只保存车辆进入ROI的第一帧，识别结果取众数\n")
            f.write(f"HEVC解码修复: 已启用\n")
            f.write("=" * 60 + "\n")

    enter_frames = {}  # 缓存进入帧 {track_id: enter_frame_image}

    # 辅助函数：创建黑色背景帧
    def create_black_frame():
        """创建黑色背景帧"""
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(black_frame, "VIDEO ENDED - NO FRAME AVAILABLE",
                    (width // 4, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    5, (255, 0, 0), 2)
        return black_frame

    # 辅助函数：保存进入ROI的图像
    def save_enter_image(frame, track_id, plate_number, bbox, frame_num):
        """保存车辆进入ROI的图像"""
        if not save_dir or track_id in entered_tracks:
            return None

        save_path = os.path.join(save_dir, f"enter_track_{track_id}_frame_{frame_num:06d}.jpg")

        # 保存进入帧供后续使用
        if frame is not None:
            enter_frames[track_id] = frame.copy()

        # 安全检查
        if frame is None:
            print(f"[WARN] 无法保存跟踪 {track_id} 的进入图像：帧为None")
            return None

        try:
            # 使用PIL处理中文（如果可用）
            if PIL_AVAILABLE and font:
                # 将OpenCV图像转换为PIL格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(pil_img)

                # 绘制车牌框（绿色）
                x1, y1, x2, y2 = bbox
                draw.rectangle([(x1, y1), (x2, y2)],
                               outline=(0, 255, 0), width=3)

                # 绘制信息
                title_text = "车辆进入ROI"
                plate_text = f"初始车牌: {plate_number}"
                track_text = f"跟踪ID: {track_id}"
                frame_text = f"帧号: {frame_num}"

                # 绘制标题
                draw.text((10, 30), title_text, font=font, fill=(255, 0, 255))

                # 绘制车牌信息
                text_y = y1 - 30
                draw.text((x1, text_y), plate_text, font=font, fill=(0, 255, 0))

                text_y -= 60
                draw.text((x1, text_y), track_text, font=font, fill=(255, 255, 0))

                text_y -= 60
                draw.text((x1, text_y), frame_text, font=font, fill=(255, 255, 255))

                # 转换回OpenCV格式
                save_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                # 回退到OpenCV绘制
                save_frame = frame.copy()

                # 绘制车牌框
                x1, y1, x2, y2 = bbox
                cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # 绘制信息（英文）
                title_text = "Vehicle Enter ROI"
                plate_text = f"Initial Plate: {plate_number}"
                track_text = f"Track ID: {track_id}"
                frame_text = f"Frame: {frame_num}"

                cv2.putText(save_frame, title_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)

                cv2.putText(save_frame, plate_text,
                            (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (0, 255, 0), 2)

                cv2.putText(save_frame, track_text,
                            (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (255, 255, 0), 2)

                cv2.putText(save_frame, frame_text,
                            (x1, y1 - 90), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (255, 255, 255), 2)

            cv2.imwrite(save_path, save_frame)
            entered_tracks.add(track_id)
            return save_path
        except Exception as e:
            print(f"[ERROR] 保存进入图像失败 (跟踪 {track_id}): {e}")
            return None

    # 辅助函数：保存最终识别结果图像
    def save_final_result_image(frame, track_id, final_result, enter_frame_num, exit_frame_num, is_forced_exit=False):
        """保存最终识别结果图像"""
        if not save_dir:
            return None

        # 安全检查
        if frame is None:
            print(f"[WARN] 无法保存跟踪 {track_id} 的最终结果图像：帧为None")
            return None

        try:
            save_frame = frame.copy()
        except AttributeError as e:
            print(f"[WARN] 无法保存跟踪 {track_id} 的最终结果图像：{e}")
            return None

        save_path = os.path.join(save_dir, f"final_track_{track_id}_result.jpg")

        # 使用PIL处理中文（如果可用）
        if PIL_AVAILABLE and font:
            # 将OpenCV图像转换为PIL格式
            frame_rgb = cv2.cvtColor(save_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)

            # 绘制信息标题
            if is_forced_exit:
                title_text = f"强制结束识别结果（视频中断）"
                title_color = (255, 100, 100)  # 淡红色
            else:
                title_text = "最终识别结果（众数）"
                title_color = (0, 255, 255)  # 青色

            draw.text((10, 30), title_text, font=font, fill=title_color)

            # 绘制识别结果信息
            info_y = 120
            info_lines = [
                f"跟踪ID: {track_id}",
                f"最终车牌: {final_result['plate']}",
                f"众数置信度: {final_result['mode_confidence']:.3f}",
                f"平均置信度: {final_result['confidence']:.3f}",
                f"进入帧: {enter_frame_num:06d}",
                f"离开帧: {exit_frame_num:06d}",
                f"总检测帧数: {final_result['total_frames']}",
            ]

            if is_forced_exit:
                info_lines.append("状态: 强制结束 (视频中断时车牌仍在ROI内)")
            else:
                info_lines.append("状态: 自然离开")

            for line in info_lines:
                draw.text((20, info_y), line, font=font, fill=(255, 0, 0))
                info_y += 60

            # 转换回OpenCV格式
            save_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # 回退到OpenCV绘制
            # 绘制信息标题
            if is_forced_exit:
                title_text = "Final Result (Forced Exit - Video Ended)"
                title_color = (100, 100, 255)  # BGR
            else:
                title_text = "Final Recognition Result (Mode)"
                title_color = (255, 255, 0)  # BGR

            cv2.putText(save_frame, title_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, title_color, 2)

            info_y = 120
            info_lines = [
                f"Track ID: {track_id}",
                f"Final Plate: {final_result['plate']}",
                f"Mode Confidence: {final_result['mode_confidence']:.3f}",
                f"Avg Confidence: {final_result['confidence']:.3f}",
                f"Enter Frame: {enter_frame_num:06d}",
                f"Exit Frame: {exit_frame_num:06d}",
                f"Total Frames: {final_result['total_frames']}",
            ]

            if is_forced_exit:
                info_lines.append("Status: Forced Exit (Plate still in ROI)")
            else:
                info_lines.append("Status: Natural Exit")

            for line in info_lines:
                cv2.putText(save_frame, line, (20, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
                info_y += 60

        try:
            cv2.imwrite(save_path, save_frame)
            return save_path
        except Exception as e:
            print(f"[ERROR] 保存图像失败: {e}")
            return None

    # 视频处理循环 - 增强版本，处理HEVC错误
    try:
        while True:
            video_stats['total_reads'] += 1

            # 定期重置VideoCapture以防止HEVC累积错误
            if frame_count > 0 and frame_count - video_stats['last_reset_frame'] >= reset_interval:
                print(f"[RESET] 达到重置间隔 ({reset_interval}帧)，重置VideoCapture...")
                cap.release()
                time.sleep(0.5)

                new_cap = reset_video_capture(video_source, is_rtsp)
                if new_cap:
                    cap = new_cap
                    video_stats['reset_count'] += 1
                    video_stats['last_reset_frame'] = frame_count
                    print(f"[RESET] 重置成功，继续处理...")
                else:
                    print(f"[RESET] 重置失败，尝试继续...")

            # 使用增强的稳健读取函数
            ret, frame, error_msg = robust_video_capture(cap, max_retries=3, skip_frames_on_error=10)

            if not ret:
                video_stats['hevc_errors'] += 1
                hevc_error_count += 1

                if hevc_error_count >= max_hevc_errors:
                    print(f"[HEVC ERROR] 达到最大HEVC错误次数 ({max_hevc_errors})，尝试重置...")
                    cap.release()
                    time.sleep(1)

                    new_cap = reset_video_capture(video_source, is_rtsp)
                    if new_cap:
                        cap = new_cap
                        hevc_error_count = 0
                        video_stats['reset_count'] += 1
                        video_stats['last_reset_frame'] = frame_count
                        print(f"[HEVC ERROR] 重置成功，继续处理...")
                        continue
                    else:
                        print(f"[HEVC ERROR] 重置失败，退出处理...")
                        break

                if is_rtsp:
                    # RTSP断流重连
                    print(f"[WARN] 读取帧失败，第{reconnect_count}次尝试重连...")
                    reconnect_count += 1

                    # 逐步增加重连等待时间（1s, 2s, 4s...）
                    wait_time = min(2 ** reconnect_count, 10)
                    time.sleep(wait_time)

                    # 释放原资源
                    cap.release()

                    new_cap = reset_video_capture(video_source, is_rtsp)
                    if new_cap:
                        cap = new_cap
                        print(f"[INFO] 重连成功，等待{wait_time}秒后继续")
                        time.sleep(1)  # 给流一些稳定时间
                        reconnect_count = 0  # 重置重连计数
                        hevc_error_count = 0  # 重置HEVC错误计数
                    else:
                        print(f"[ERROR] 重连失败，{wait_time}秒后再次尝试")
                    continue
                else:
                    # 本地视频结束 - 处理所有未完成的跟踪
                    print("[INFO] 视频结束，处理ROI内剩余车辆...")

                    # 强制结束所有活动跟踪
                    for track_id in list(roi_processor.active_tracks.keys()):
                        if track_id not in completed_tracks:
                            # 获取众数结果
                            final_result = roi_processor.get_final_result(track_id)
                            if final_result:
                                completed_tracks.add(track_id)

                                # 写入强制结束记录
                                if result_file:
                                    with open(result_file, 'a') as f:
                                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                                        f.write(f"{timestamp} | 跟踪ID: {track_id} | ")
                                        f.write(f"识别结果: {final_result['plate']} | ")
                                        f.write(f"置信度: {final_result['confidence']:.3f} | ")
                                        f.write(f"众数置信度: {final_result['mode_confidence']:.3f} | ")
                                        f.write(f"进入帧: {final_result['enter_frame']:06d} | ")
                                        f.write(f"离开帧: {frame_count:06d} (视频结束) | ")
                                        f.write(f"总检测帧数: {final_result['total_frames']} | ")
                                        f.write(f"状态: 强制结束 (车牌仍在ROI内)\n")

                                # 保存最终结果图像
                                if save_dir:
                                    # 使用进入时的帧或最后有效帧
                                    reference_frame = None
                                    if track_id in enter_frames:
                                        reference_frame = enter_frames[track_id]
                                        print(f"[INFO] 使用进入帧保存跟踪 {track_id} 的结果")
                                    elif last_valid_frame is not None:
                                        reference_frame = last_valid_frame
                                        print(f"[INFO] 使用最后有效帧保存跟踪 {track_id} 的结果")
                                    else:
                                        # 创建黑色背景帧
                                        reference_frame = create_black_frame()
                                        print(f"[INFO] 创建黑色背景帧保存跟踪 {track_id} 的结果")

                                    save_path = save_final_result_image(
                                        reference_frame, track_id, final_result,
                                        final_result['enter_frame'], frame_count,
                                        is_forced_exit=True  # 标记为强制结束
                                    )
                                    if save_path:
                                        print(f"[SAVE] 保存强制结束识别结果: {save_path}")

                                    # 清理缓存
                                    if track_id in enter_frames:
                                        del enter_frames[track_id]
                    break  # 退出循环
            else:
                # 正常读取到帧
                video_stats['successful_reads'] += 1
                last_valid_frame = frame.copy()
                frame_count += 1
                reconnect_count = 0
                hevc_error_count = 0  # 重置HEVC错误计数

                if frame is None or frame.size == 0:
                    print(f"[WARN] 帧{frame_count}解码异常，跳过")
                    continue

                # 进度显示（仅本地视频）
                if not is_rtsp and total_frames > 0 and frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"[进度] {frame_count}/{total_frames} 帧 ({progress:.1f}%)")

            # 如果frame是None（RTSP重连时），跳过本循环
            if frame is None:
                continue

            # 创建掩码图像：只保留ROI区域
            masked_frame = create_masked_image(frame, roi_coords)

            # 处理检测（使用掩码图像）
            start = time.time()
            # 使用新的方法：用掩码图像检测，但用原始图像进行车牌识别
            results = recognizer.recognize_with_masked_det(frame, masked_frame, conf_thresh)
            elapsed = time.time() - start
            process_fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_list.append(process_fps)

            # 处理每个检测到的车牌
            current_detections = []

            for res in results:
                plate_number = res['plate_number']
                bbox = res['bbox']
                confidence = res['rec_conf']

                # === 车牌格式校验 ===
                validation = validate_plate_format(plate_number, confidence)
                
                # 如果格式校验不为 accepted，跳过此检测结果
                if validation['level'] != 'accepted':
                    # 可选：打印被拒绝的结果用于调试
                    # print(f"[REJECT] 车牌格式不合法: {plate_number} | 问题: {validation['issues']}")
                    continue
                
                # 使用校验后调整的置信度
                adjusted_confidence = validation['adjusted_confidence']

                # 检查是否在ROI内
                in_roi = roi_processor.is_in_roi(bbox)

                if in_roi:
                    # 跟踪车辆进出ROI (使用调整后的置信度)
                    track_id, action, is_active = roi_processor.track_vehicle(
                        plate_number, bbox, frame_count, adjusted_confidence
                    )

                    if track_id is not None:
                        # 添加到当前显示
                        current_detections.append({
                            'bbox': bbox,
                            'plate_number': plate_number,
                            'track_id': track_id,
                            'confidence': adjusted_confidence,
                            'is_active': is_active,
                            'format_level': validation['level']  # 记录格式校验级别
                        })

                        # 处理进入事件
                        if action == 'enter' and result_file:
                            # 写入进入记录 (包含格式校验信息)
                            with open(result_file, 'a') as f:
                                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                                f.write(f"{timestamp} | Frame {frame_count:06d} | ")
                                f.write(f"车辆进入 | 跟踪ID: {track_id} | ")
                                f.write(f"初始车牌: {plate_number} | ")
                                f.write(f"位置: {bbox} | ")
                                f.write(f"原始置信度: {confidence:.3f} | ")
                                f.write(f"调整置信度: {adjusted_confidence:.3f} | ")
                                f.write(f"格式校验: {validation['level']}\n")

                            # 保存进入ROI的图像
                            if save_dir:
                                save_path = save_enter_image(frame, track_id, plate_number, bbox, frame_count)
                                if save_path:
                                    print(f"[SAVE] 保存进入ROI图像: {save_path} | 跟踪ID: {track_id}")

                            # 打印进入信息 (包含格式校验状态)
                            fmt_info = f" [{validation['level']}]" if validation['level'] != 'accepted' else ""
                            print(f"[ENTER] 跟踪ID {track_id}: 车辆进入ROI | 初始车牌: {plate_number}{fmt_info}")

                        # 处理自然离开事件
                        elif action == 'exit':
                            # 车辆离开ROI，计算最终众数结果
                            final_result = roi_processor.get_final_result(track_id)
                            if final_result and track_id not in completed_tracks:
                                completed_tracks.add(track_id)

                                # 写入最终识别结果
                                if result_file:
                                    with open(result_file, 'a') as f:
                                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                                        f.write(f"{timestamp} | 跟踪ID: {track_id} | ")
                                        f.write(f"识别结果: {final_result['plate']} | ")
                                        f.write(f"置信度: {final_result['confidence']:.3f} | ")
                                        f.write(f"众数置信度: {final_result['mode_confidence']:.3f} | ")
                                        f.write(f"进入帧: {final_result['enter_frame']:06d} | ")
                                        f.write(f"离开帧: {final_result['exit_frame']:06d} | ")
                                        f.write(f"总检测帧数: {final_result['total_frames']} | ")
                                        f.write(f"状态: 自然离开\n")

                                # 保存最终结果图像
                                if save_dir and frame is not None:
                                    # 使用当前帧或进入帧
                                    reference_frame = None
                                    if track_id in enter_frames:
                                        reference_frame = enter_frames[track_id]
                                    else:
                                        reference_frame = frame

                                    save_path = save_final_result_image(
                                        reference_frame, track_id, final_result,
                                        final_result['enter_frame'], final_result['exit_frame'],
                                        is_forced_exit=False  # 自然离开
                                    )
                                    if save_path:
                                        print(f"[SAVE] 保存自然离开识别结果: {save_path}")

                                    # 清理缓存
                                    if track_id in enter_frames:
                                        del enter_frames[track_id]

                                print(f"[EXIT] 跟踪ID {track_id}: 车辆离开ROI | 识别结果: {final_result['plate']} | "
                                      f"众数置信度: {final_result['mode_confidence']:.2f} | "
                                      f"检测帧数: {final_result['total_frames']}")

            # 每5秒打印一次统计信息
            now = time.time()
            if now - last_print_time >= 5.0:
                avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
                active_count = roi_processor.get_active_track_count()
                completed_count = roi_processor.get_completed_track_count()

                # 计算读取成功率
                read_success_rate = (video_stats['successful_reads'] / video_stats['total_reads'] * 100) if video_stats[
                                                                                                                'total_reads'] > 0 else 0

                stat_msg = f"[STAT] 已处理: {frame_count}"
                if not is_rtsp:
                    stat_msg += f"/{total_frames} 帧 ({frame_count / total_frames * 100:.1f}%)"
                stat_msg += f" | 处理FPS: {avg_fps:.1f}"
                if is_rtsp and fps > 0:
                    stat_msg += f" | 视频FPS: {fps:.1f}"
                stat_msg += f" | ROI内车辆: {active_count} | 已完成跟踪: {completed_count}"
                stat_msg += f" | 读取成功率: {read_success_rate:.1f}%"
                stat_msg += f" | HEVC错误: {video_stats['hevc_errors']}"
                stat_msg += f" | 重置次数: {video_stats['reset_count']}"
                if save_dir:
                    stat_msg += f" | 已保存图像: {len(entered_tracks)}张"

                print(stat_msg)
                last_print_time = now

            # 可视化显示（使用原始图像，不显示掩码效果）
            if display or (save_dir and save_all and frame_count % save_interval == 0):
                # 使用原始图像进行可视化
                vis_frame = visualize_result(frame, current_detections, font)
                avg_fps = np.mean(fps_list[-30:]) if fps_list else 0

                # 显示统计信息
                cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

                active_count = roi_processor.get_active_track_count()
                cv2.putText(vis_frame, f"Active: {active_count}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 2)

                # 显示视频统计
                read_success_rate = (video_stats['successful_reads'] / video_stats['total_reads'] * 100) if video_stats[
                                                                                                                'total_reads'] > 0 else 0
                cv2.putText(vis_frame, f"Read: {read_success_rate:.1f}%", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 2)

                # 显示已保存图像数量
                if save_dir:
                    cv2.putText(vis_frame, f"Saved: {len(entered_tracks)}", (10, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)

                # 显示视频进度条（仅本地视频）
                if not is_rtsp and width > 600 and total_frames > 0:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    progress_width = int((frame_count / total_frames) * (width - 40))
                    cv2.rectangle(vis_frame, (20, height - 30),
                                  (20 + progress_width, height - 10),
                                  (0, 255, 0), -1)
                    cv2.rectangle(vis_frame, (20, height - 30),
                                  (width - 20, height - 10),
                                  (255, 255, 255), 2)
                    cv2.putText(vis_frame, f"{progress:.1f}%",
                                (width - 100, height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

                if display:
                    window_title = f"Plate Recognition - {'RTSP' if is_rtsp else os.path.basename(video_source)}"
                    cv2.imshow(window_title, vis_frame)

                    # 调整播放速度
                    wait_time = max(1, int(1000 / (fps if fps > 0 else 30))) if not is_rtsp else 1
                    key = cv2.waitKey(wait_time) & 0xFF

                    if key == ord('q'):
                        print("[INFO] 用户退出")
                        break
                    elif key == ord(' '):  # 空格键暂停/继续
                        print("[INFO] 暂停，按任意键继续...")
                        cv2.waitKey(0)
                    elif key == ord('r'):  # 手动重置
                        print("[INFO] 手动重置VideoCapture...")
                        cap.release()
                        time.sleep(0.5)
                        new_cap = reset_video_capture(video_source, is_rtsp)
                        if new_cap:
                            cap = new_cap
                            video_stats['reset_count'] += 1
                            video_stats['last_reset_frame'] = frame_count
                            print("[INFO] 手动重置成功")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断 (Ctrl+C)")

    except Exception as e:
        print(f"\n[ERROR] 处理出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()

        # 保存最终统计
        if result_file and save_dir:
            with open(result_file, 'a') as f:
                f.write("=" * 60 + "\n")
                f.write(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总帧数: {frame_count}")
                if not is_rtsp:
                    f.write(f" (共 {total_frames})")
                f.write("\n")

                # 视频处理统计
                read_success_rate = (video_stats['successful_reads'] / video_stats['total_reads'] * 100) if video_stats[
                                                                                                                'total_reads'] > 0 else 0
                f.write(f"视频读取统计:\n")
                f.write(f"  总读取次数: {video_stats['total_reads']}\n")
                f.write(f"  成功读取: {video_stats['successful_reads']}\n")
                f.write(f"  读取成功率: {read_success_rate:.1f}%\n")
                f.write(f"  HEVC错误次数: {video_stats['hevc_errors']}\n")
                f.write(f"  VideoCapture重置次数: {video_stats['reset_count']}\n")

                active_count = roi_processor.get_active_track_count()
                completed_count = roi_processor.get_completed_track_count()
                total_tracks = len(roi_processor.tracked_cars)

                f.write(f"检测到车辆总数: {total_tracks}\n")
                f.write(f"当前ROI内车辆: {active_count}\n")
                f.write(f"已完成跟踪车辆: {completed_count}\n")
                f.write(f"保存进入图像: {len(entered_tracks)}张\n")

                # 列出所有识别结果
                if roi_processor.tracked_cars:
                    f.write("识别结果汇总:\n")
                    for track_id in roi_processor.tracked_cars.keys():
                        result = roi_processor.get_final_result(track_id)
                        if result:
                            f.write(f"  跟踪ID {track_id}: {result['plate']} ")
                            f.write(f"(进入帧: {result['enter_frame']}, ")
                            f.write(f"离开帧: {result['exit_frame']}, ")
                            f.write(f"众数置信度: {result['mode_confidence']:.2f})\n")

                if fps_list:
                    avg_process_fps = np.mean(fps_list)
                    f.write(f"平均处理FPS: {avg_process_fps:.1f}\n")
                    if not is_rtsp and fps > 0:
                        efficiency = (avg_process_fps / fps * 100)
                        f.write(f"处理效率: {efficiency:.1f}% (相对视频原始帧率)\n")

        # 打印最终统计
        print("\n" + "=" * 60)
        print("处理统计:")
        print(f"  总帧数: {frame_count}")
        if not is_rtsp:
            print(f"  处理完成度: {frame_count / total_frames * 100:.1f}%")

        # 视频读取统计
        read_success_rate = (video_stats['successful_reads'] / video_stats['total_reads'] * 100) if video_stats[
                                                                                                        'total_reads'] > 0 else 0
        print(f"  视频读取统计:")
        print(f"    总读取次数: {video_stats['total_reads']}")
        print(f"    成功读取: {video_stats['successful_reads']}")
        print(f"    读取成功率: {read_success_rate:.1f}%")
        print(f"    HEVC错误次数: {video_stats['hevc_errors']}")
        print(f"    VideoCapture重置次数: {video_stats['reset_count']}")

        active_count = roi_processor.get_active_track_count()
        completed_count = roi_processor.get_completed_track_count()
        total_tracks = len(roi_processor.tracked_cars)

        print(f"  检测到车辆总数: {total_tracks}")
        print(f"  当前ROI内车辆: {active_count}")
        print(f"  已完成跟踪车辆: {completed_count}")
        print(f"  保存进入图像: {len(entered_tracks)}张")

        if fps_list:
            avg_process_fps = np.mean(fps_list)
            print(f"  平均处理FPS: {avg_process_fps:.1f}")
            if not is_rtsp and fps > 0:
                efficiency = (avg_process_fps / fps * 100)
                print(f"  处理效率: {efficiency:.1f}%")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='E2E_HZTK RK3588 NPU Deployment')
    parser.add_argument('--model-dir', type=str, default='./', help='RKNN models directory')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image-dir', type=str, help='Image directory')
    parser.add_argument('--output-dir', '-o', type=str, help='Output directory for saving visualized images')
    parser.add_argument('--rtsp', type=str, help='RTSP stream URL')
    parser.add_argument('--no-display', action='store_true', help='Disable display for RTSP')
    parser.add_argument('--save-all', action='store_true', help='Save all frames (not just detected ones)')
    parser.add_argument('--save-interval', type=int, default=1, help='Save every N frames (default: 1)')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--conf-thresh', type=float, default=0.5, help='Confidence threshold')

    parser.add_argument('--video', type=str, help='Local video file path')
    parser.add_argument('--roi-x1', type=int, default=800, help='ROI左上角X坐标')
    parser.add_argument('--roi-y1', type=int, default=1200, help='ROI左上角Y坐标')
    parser.add_argument('--roi-x2', type=int, default=2800, help='ROI右下角X坐标')
    parser.add_argument('--roi-y2', type=int, default=2000, help='ROI右下角Y坐标')
    parser.add_argument('--history-frames', type=int, default=10, help='众数计算的历史帧数')

    # 新增参数：HEVC相关
    parser.add_argument('--reset-interval', type=int, default=500, help='重置VideoCapture的间隔帧数（默认: 500）')
    parser.add_argument('--max-hevc-errors', type=int, default=10, help='最大HEVC错误次数（默认: 10）')

    args = parser.parse_args()

    recognizer = PlateRecognizerRKNN(args.model_dir)

    try:
        if args.benchmark:
            print("\n" + "=" * 50)
            print("Performance Benchmark (RK3588 NPU)")
            print("=" * 50)

            metrics = recognizer.benchmark()

            if 'error' in metrics:
                print(f"Error: {metrics['error']}")
            else:
                print(f"\nResults:")
                print(f"  Detection:      {metrics['detection_ms']:.2f} ms")
                print(f"  Recognition:    {metrics['recognition_ms']:.2f} ms")
                print(f"  Classification: {metrics['classification_ms']:.2f} ms")
                print(f"  Total:          {metrics['total_ms']:.2f} ms")
                print(f"  FPS:            {metrics['fps']:.1f}")

        elif args.rtsp:
            # 使用统一的process_video_stream处理RTSP流
            process_video_stream(
                recognizer,
                args.rtsp,
                args.conf_thresh,
                display=not args.no_display,
                save_dir=args.output_dir,
                save_all=args.save_all,
                save_interval=args.save_interval,
                roi_coords=(args.roi_x1, args.roi_y1, args.roi_x2, args.roi_y2),
                history_frames=args.history_frames,
                is_rtsp=True  # 标记为RTSP流
            )

        elif args.image:
            image = cv2.imread(args.image)
            if image is None:
                print(f"Error: Cannot read {args.image}")
                sys.exit(1)

            # 创建掩码图像
            masked_image = create_masked_image(image, (args.roi_x1, args.roi_y1, args.roi_x2, args.roi_y2))

            start = time.time()
            # 使用掩码图像进行检测
            results = recognizer.recognize_with_masked_det(image, masked_image, args.conf_thresh)
            elapsed = time.time() - start

            print(f"\nResults for {args.image} ({elapsed * 1000:.1f} ms):")
            print(f"ROI区域: ({args.roi_x1}, {args.roi_y1}, {args.roi_x2}, {args.roi_y2})")
            if results:
                for i, res in enumerate(results):
                    print(f"  [{i + 1}] {res['plate_number']} ({res['plate_type']})")
                    print(f"      bbox: {res['bbox']}, det: {res['det_conf']:.3f}, rec: {res['rec_conf']:.3f}")
            else:
                print("  No plate detected")

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                font = get_chinese_font(size=24)
                # 可视化时使用原始图像
                vis_img = visualize_result(image, results, font)
                output_path = os.path.join(args.output_dir, os.path.basename(args.image))
                cv2.imwrite(output_path, vis_img)
                print(f"Saved: {output_path}")

        elif args.image_dir:
            image_files = [f for f in os.listdir(args.image_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                print(f"No images found in {args.image_dir}")
                sys.exit(1)

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)

            font = get_chinese_font(size=24)
            total_time = 0
            total_images = 0

            for img_file in image_files:
                img_path = os.path.join(args.image_dir, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                # 创建掩码图像
                masked_image = create_masked_image(image, (args.roi_x1, args.roi_y1, args.roi_x2, args.roi_y2))

                start = time.time()
                # 使用掩码图像进行检测
                results = recognizer.recognize_with_masked_det(image, masked_image, args.conf_thresh)
                elapsed = time.time() - start

                total_time += elapsed
                total_images += 1

                print(f"{img_file}: ", end="")
                if results:
                    print(f"{results[0]['plate_number']} ({elapsed * 1000:.1f}ms)")
                else:
                    print(f"No plate ({elapsed * 1000:.1f}ms)")

                if args.output_dir:
                    # 可视化时使用原始图像
                    vis_img = visualize_result(image, results, font)
                    output_path = os.path.join(args.output_dir, img_file)
                    cv2.imwrite(output_path, vis_img)

            if total_images > 0:
                avg = total_time / total_images * 1000
                print(f"\nProcessed {total_images} images, avg: {avg:.1f} ms ({1000 / avg:.1f} FPS)")

        elif args.video:  # 处理本地视频
            if not os.path.exists(args.video):
                print(f"错误: 视频文件不存在 '{args.video}'")
                sys.exit(1)

            # 使用统一的process_video_stream处理本地视频
            process_video_stream(
                recognizer,
                args.video,
                args.conf_thresh,
                display=not args.no_display,
                save_dir=args.output_dir,
                save_all=args.save_all,
                save_interval=args.save_interval,
                roi_coords=(args.roi_x1, args.roi_y1, args.roi_x2, args.roi_y2),
                history_frames=args.history_frames,
                is_rtsp=False  # 标记为本地视频
            )

        else:
            parser.print_help()

    finally:
        recognizer.release()


if __name__ == '__main__':
    main()