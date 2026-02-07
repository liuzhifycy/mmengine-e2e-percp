#!/usr/bin/env python3
"""
车牌识别后处理模块

三个层次的优化, 不改模型, 不重训练:

  1. PlateFormatValidator   — 车牌格式校验 (结构规则过滤明显错误)
  2. MultiFrameVoter        — 多帧投票器 (固定摄像头场景, 时序聚合)
  3. RefineBeforeRecognize   — ROI精裁后再识别 (对抗ROI不精确)

用法:
  # 单帧后处理 (格式校验)
  validator = PlateFormatValidator()
  result = validator.validate("贵A07433D", 0.95)

  # 多帧投票 (持续喂入)
  voter = MultiFrameVoter(min_hits=3, conf_threshold=0.7)
  voter.feed("贵A07433D", 0.95)
  voter.feed("贵A07433D", 0.92)
  voter.feed("贵A07438D", 0.61)
  final = voter.decide()  # → ("贵A07433D", 0.935, 'confirmed')

  # 精裁+识别
  refiner = RefineBeforeRecognize(recognizer, preprocessor)
  text, conf = refiner.recognize(image, coarse_roi)

  # 评估脚本
  python plate_postprocess.py --eval data/ccpd/processed/recognition/test
"""

import re
import time
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================================
# 中国车牌规则常量
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

# ============================================================================
# 方案2: 车牌格式校验
# ============================================================================
class PlateFormatValidator:
    """
    基于中国车牌结构规则的格式校验器

    中国车牌格式:
      普通蓝/黄牌:  省 + 字母 + 5位(字母/数字)         = 7位
      新能源绿牌:   省 + 字母 + 1位小字(DF) + 5位      = 8位
                   或 省 + 字母 + 5位 + 1位小字(DF)    = 8位
      警/军/使馆牌: 格式不同, 这里做宽松匹配

    校验层次:
      Level 1: 长度 (7或8位)
      Level 2: 位置0是省份汉字
      Level 3: 位置1是字母A-Z (无I/O)
      Level 4: 位置2+是合法字符 (字母/数字, 无I/O)

    不通过校验的结果标记为 rejected, 由调用方决定丢弃还是降权
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: True=必须完全符合格式才accept,
                    False=宽松模式, 只拒绝明显错误 (长度异常/省份缺失)
        """
        self.strict = strict

    def validate(self, plate_text: str, confidence: float) -> dict:
        """
        校验一个识别结果

        Returns:
            {
                'text': 原文本,
                'confidence': 原置信度,
                'valid': bool,
                'adjusted_confidence': 校验后置信度 (不合格的降低),
                'issues': 问题列表,
                'level': 'accepted' / 'suspicious' / 'rejected',
            }
        """
        issues = []
        penalty = 0.0  # 置信度惩罚

        # --- Level 1: 长度 ---
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
        # n == 7 或 8 是正常的

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
            # 新能源: 位置2是D/F, 或位置7是D/F
            has_ne_prefix = plate_text[2] in NEW_ENERGY_SUFFIX
            has_ne_suffix = plate_text[7] in NEW_ENERGY_SUFFIX
            if not has_ne_prefix and not has_ne_suffix:
                # 8位但不像新能源, 可疑
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
            valid = not self.strict

        return {
            'text': plate_text,
            'confidence': confidence,
            'valid': valid,
            'adjusted_confidence': round(adjusted, 4),
            'issues': issues,
            'level': level,
        }

    def is_valid_format(self, plate_text: str) -> bool:
        """快速判断格式是否合法"""
        n = len(plate_text)
        if n < 7 or n > 8:
            return False
        if plate_text[0] not in PROVINCES:
            return False
        if plate_text[1] not in PLATE_LETTERS:
            return False
        return True


# ============================================================================
# 方案3: 多帧投票器
# ============================================================================
class MultiFrameVoter:
    """
    多帧投票器 — 固定摄像头场景的时序聚合

    原理:
      车辆在摄像头前停留数秒 → 多帧识别 → 取众数
      低置信度帧直接丢弃, 格式不合法的降权

    状态机:
      idle → collecting (收到第一个有效结果)
           → confirmed (达到 min_hits 一致)
           → idle (超时无新帧, 或车牌变化说明换车了)

    使用场景:
      固定摄像头, 持续识别, 每帧调 feed() 喂入结果,
      调 decide() 获取当前最佳判定
    """

    def __init__(self,
                 min_hits: int = 3,
                 conf_threshold: float = 0.7,
                 window_size: int = 10,
                 timeout_sec: float = 5.0,
                 format_validator: PlateFormatValidator = None):
        """
        Args:
            min_hits:       同一车牌号至少出现几次才确认
            conf_threshold: 低于此置信度的帧直接丢弃
            window_size:    滑动窗口大小 (保留最近N帧)
            timeout_sec:    超时秒数, 超过则重置状态
            format_validator: 格式校验器, None则不校验
        """
        self.min_hits = min_hits
        self.conf_threshold = conf_threshold
        self.window_size = window_size
        self.timeout_sec = timeout_sec
        self.validator = format_validator or PlateFormatValidator()

        # 滑动窗口: 存 (text, confidence, timestamp)
        self._window: deque = deque(maxlen=window_size)
        self._last_feed_time: float = 0.0
        self._state: str = 'idle'  # idle / collecting / confirmed
        self._confirmed_text: str = ''
        self._confirmed_conf: float = 0.0

    def reset(self):
        """重置状态 (换车了)"""
        self._window.clear()
        self._state = 'idle'
        self._confirmed_text = ''
        self._confirmed_conf = 0.0

    def feed(self, plate_text: str, confidence: float,
             timestamp: float = None) -> dict:
        """
        喂入一帧识别结果

        Args:
            plate_text:  识别文本
            confidence:  置信度
            timestamp:   时间戳 (None则用当前时间)

        Returns:
            {
                'accepted': 此帧是否被接受,
                'reason': 接受/拒绝原因,
                'state': 当前状态,
                'best_text': 当前最佳候选,
                'best_conf': 当前最佳置信度,
                'hit_count': 最佳候选出现次数,
            }
        """
        now = timestamp or time.time()

        # 超时重置
        if self._last_feed_time > 0 and (now - self._last_feed_time) > self.timeout_sec:
            self.reset()
        self._last_feed_time = now

        # --- 门控1: 置信度 ---
        if confidence < self.conf_threshold:
            return self._make_response(False, f'置信度过低({confidence:.2f}<{self.conf_threshold})')

        # --- 门控2: 格式校验 ---
        vr = self.validator.validate(plate_text, confidence)
        if vr['level'] == 'rejected':
            return self._make_response(False, f'格式不合法: {vr["issues"]}')

        # 使用校验后的置信度
        adj_conf = vr['adjusted_confidence']

        # --- 加入窗口 ---
        self._window.append((plate_text, adj_conf, now))

        if self._state == 'idle':
            self._state = 'collecting'

        # --- 检查是否可以确认 ---
        # 如果已确认, 且新帧也是同一车牌, 更新置信度
        if self._state == 'confirmed' and plate_text == self._confirmed_text:
            self._confirmed_conf = max(self._confirmed_conf, adj_conf)
            return self._make_response(True, '已确认, 更新置信度')

        # 如果已确认但来了不同车牌, 可能换车了
        if self._state == 'confirmed' and plate_text != self._confirmed_text:
            # 统计窗口里新车牌出现几次
            new_count = sum(1 for t, c, ts in self._window if t == plate_text)
            if new_count >= self.min_hits:
                # 新车确认
                self._confirmed_text = plate_text
                confs = [c for t, c, ts in self._window if t == plate_text]
                self._confirmed_conf = float(np.mean(confs))
                return self._make_response(True, '新车牌确认')
            else:
                return self._make_response(True, '收集中, 可能换车')

        # collecting 状态: 统计投票
        best_text, best_count, best_conf = self._get_best_candidate()

        if best_count >= self.min_hits:
            self._state = 'confirmed'
            self._confirmed_text = best_text
            self._confirmed_conf = best_conf
            return self._make_response(True, f'确认({best_count}票)')

        return self._make_response(True, f'收集中({best_count}/{self.min_hits})')

    def decide(self) -> Tuple[str, float, str]:
        """
        获取当前最佳判定

        Returns:
            (plate_text, confidence, state)
            state: 'confirmed' / 'collecting' / 'idle'
        """
        if self._state == 'confirmed':
            return self._confirmed_text, self._confirmed_conf, 'confirmed'

        if self._state == 'collecting' and self._window:
            best_text, best_count, best_conf = self._get_best_candidate()
            return best_text, best_conf, 'collecting'

        return '', 0.0, 'idle'

    def _get_best_candidate(self) -> Tuple[str, int, float]:
        """从窗口中找出现次数最多的车牌及其平均置信度"""
        if not self._window:
            return '', 0, 0.0

        counter = Counter()
        conf_sums: Dict[str, List[float]] = {}
        for text, conf, ts in self._window:
            counter[text] += 1
            conf_sums.setdefault(text, []).append(conf)

        best_text, best_count = counter.most_common(1)[0]
        best_conf = float(np.mean(conf_sums[best_text]))
        return best_text, best_count, best_conf

    def _make_response(self, accepted: bool, reason: str) -> dict:
        best_text, best_count, best_conf = self._get_best_candidate() \
            if self._window else ('', 0, 0.0)

        return {
            'accepted': accepted,
            'reason': reason,
            'state': self._state,
            'best_text': self._confirmed_text if self._state == 'confirmed' else best_text,
            'best_conf': self._confirmed_conf if self._state == 'confirmed' else best_conf,
            'hit_count': best_count,
        }


# ============================================================================
# 方案4: ROI 精裁后再识别
# ============================================================================
class RefineBeforeRecognize:
    """
    对抗 ROI 不精确: 先在粗 ROI 内精确定位车牌边界, 再送识别

    核心思路:
      上面的实验证明, 识别模型对背景像素极度敏感 —— 5% 的扩展就掉 20% 准确率.
      原因是 encode_image_for_rec 把整张裁剪图缩放到 48px 高, 背景挤占字符像素.

      解决: 在送入识别之前, 先用颜色/边缘方法在粗 ROI 内找到车牌精确边界,
      裁到紧贴车牌再送识别. 这样即使 ROI 有 30% 的偏差, 最终输入识别模型的
      图片依然是车牌填满画面的.

    流程:
      粗 ROI 裁剪 → PreciseROI 精确定位 → 紧贴裁剪 → encode → 识别
                                           ↓ 失败则 fallback
                                     直接用粗裁剪 → encode → 识别
    """

    def __init__(self, recognizer, preprocessor=None):
        """
        Args:
            recognizer:    LightweightPlateRecognizer 实例
            preprocessor:  PreciseROIPreprocessor 实例 (None则自动创建)
        """
        self.recognizer = recognizer

        if preprocessor is None:
            # 延迟导入, 避免循环依赖
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from roi_plate_preprocessor import PreciseROIPreprocessor
            # target_size 对精裁不重要, 因为我们拿 plate_color 而非 plate_resized
            self.preprocessor = PreciseROIPreprocessor(target_size=(160, 48))
        else:
            self.preprocessor = preprocessor

        self.validator = PlateFormatValidator()

    def recognize(self, image: np.ndarray,
                  roi: Tuple[int, int, int, int]) -> dict:
        """
        精裁 + 识别

        策略:
          1. 用 PreciseROI 在粗 ROI 内找到精确车牌边界
          2. 如果找到了 (非 fallback), 用精裁图识别
          3. 如果 fallback, 也用粗裁剪识别 (但标记 refine_failed)
          4. 对两种情况都做格式校验

        Args:
            image: 完整原图 (BGR)
            roi:   (x1, y1, x2, y2) 粗略 ROI

        Returns:
            {
                'plate_text':     识别结果,
                'confidence':     原始置信度,
                'adj_confidence': 格式校验后置信度,
                'valid':          格式是否合法,
                'refine_method':  精裁方法 ('color'/'edge'/'fallback'),
                'refine_ms':      精裁耗时,
                'recognize_ms':   识别耗时,
                'total_ms':       总耗时,
            }
        """
        t_total = time.perf_counter()

        # --- Step 1: 精裁 ---
        t0 = time.perf_counter()
        prep = self.preprocessor.process(image, roi)
        refine_ms = (time.perf_counter() - t0) * 1000

        if 'error' in prep:
            return {
                'plate_text': '', 'confidence': 0.0,
                'adj_confidence': 0.0, 'valid': False,
                'refine_method': 'error', 'refine_ms': refine_ms,
                'recognize_ms': 0.0,
                'total_ms': (time.perf_counter() - t_total) * 1000,
            }

        is_fallback = prep.get('fallback', False)
        method = prep.get('method', 'fallback')

        # 选择送入识别的图片
        if not is_fallback and 'plate_corrected' in prep:
            plate_img = prep['plate_corrected']
        elif not is_fallback and 'plate_color' in prep:
            plate_img = prep['plate_color']
        else:
            # fallback: 直接用粗裁剪的彩色图
            plate_img = prep['plate_color']

        # --- Step 2: 识别 ---
        t1 = time.perf_counter()
        plate_text, confidence = self.recognizer.recognize(plate_img)
        recognize_ms = (time.perf_counter() - t1) * 1000

        # --- Step 3: 格式校验 ---
        vr = self.validator.validate(plate_text, confidence)

        total_ms = (time.perf_counter() - t_total) * 1000

        return {
            'plate_text': plate_text,
            'confidence': confidence,
            'adj_confidence': vr['adjusted_confidence'],
            'valid': vr['valid'],
            'format_issues': vr['issues'],
            'refine_method': method,
            'refine_ms': round(refine_ms, 2),
            'recognize_ms': round(recognize_ms, 2),
            'total_ms': round(total_ms, 2),
        }


# ============================================================================
# 组合: 完整的后处理管线
# ============================================================================
class PlateRecognitionPostProcessor:
    """
    把三个后处理方案串联起来的完整管线

    单帧流程:
      ROI精裁(方案4) → 识别 → 格式校验(方案2) → 多帧投票(方案3) → 输出

    用法:
      pp = PlateRecognitionPostProcessor(recognizer)
      # 每帧调用:
      result = pp.process_frame(image, roi)
      # result['final_text'] 是最终结果
    """

    def __init__(self, recognizer,
                 min_hits: int = 3,
                 conf_threshold: float = 0.7,
                 enable_refine: bool = True,
                 enable_vote: bool = True):
        """
        Args:
            recognizer:     LightweightPlateRecognizer
            min_hits:       多帧投票确认票数
            conf_threshold: 置信度门槛
            enable_refine:  是否启用 ROI 精裁
            enable_vote:    是否启用多帧投票
        """
        self.recognizer = recognizer
        self.validator = PlateFormatValidator()
        self.enable_refine = enable_refine
        self.enable_vote = enable_vote

        if enable_refine:
            self.refiner = RefineBeforeRecognize(recognizer)
        else:
            self.refiner = None

        if enable_vote:
            self.voter = MultiFrameVoter(
                min_hits=min_hits,
                conf_threshold=conf_threshold,
                format_validator=self.validator,
            )
        else:
            self.voter = None

    def process_frame(self, image: np.ndarray,
                      roi: Tuple[int, int, int, int],
                      timestamp: float = None) -> dict:
        """
        处理单帧

        Returns:
            {
                'raw_text':        原始识别结果,
                'raw_confidence':  原始置信度,
                'valid':           格式是否合法,
                'format_issues':   格式问题列表,
                'refine_method':   精裁方法,
                'vote_state':      投票状态,
                'final_text':      最终输出 (投票确认后),
                'final_confidence': 最终置信度,
                'total_ms':        总耗时,
            }
        """
        t0 = time.perf_counter()

        # --- Step 1: 识别 (带或不带精裁) ---
        if self.refiner:
            rec = self.refiner.recognize(image, roi)
            raw_text = rec['plate_text']
            raw_conf = rec['confidence']
            adj_conf = rec['adj_confidence']
            valid = rec['valid']
            issues = rec.get('format_issues', [])
            method = rec['refine_method']
        else:
            # 不精裁, 直接裁剪+识别
            x1, y1, x2, y2 = roi
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = image[y1:y2, x1:x2]
            raw_text, raw_conf = self.recognizer.recognize(crop)
            vr = self.validator.validate(raw_text, raw_conf)
            adj_conf = vr['adjusted_confidence']
            valid = vr['valid']
            issues = vr['issues']
            method = 'direct'

        # --- Step 2: 多帧投票 ---
        if self.voter:
            vote_resp = self.voter.feed(raw_text, raw_conf, timestamp)
            final_text, final_conf, vote_state = self.voter.decide()
        else:
            final_text = raw_text if valid else ''
            final_conf = adj_conf
            vote_state = 'single_frame'

        total_ms = (time.perf_counter() - t0) * 1000

        return {
            'raw_text': raw_text,
            'raw_confidence': raw_conf,
            'adj_confidence': adj_conf,
            'valid': valid,
            'format_issues': issues,
            'refine_method': method,
            'vote_state': vote_state,
            'final_text': final_text,
            'final_confidence': final_conf,
            'total_ms': round(total_ms, 2),
        }

    def reset_voter(self):
        """重置投票器 (换车了)"""
        if self.voter:
            self.voter.reset()


# ============================================================================
# 评估脚本
# ============================================================================
def eval_postprocess(test_dir: str, num_samples: int = 0):
    """
    在 CCPD 预裁剪测试集上评估后处理效果

    对比:
      - baseline: 纯识别
      - +格式校验: 丢弃格式不合法的
      - +格式校验+置信度门槛: 丢弃低置信度的
    """
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, os.path.dirname(__file__))
    from lightweight_plate_recognizer import LightweightPlateRecognizer

    recognizer = LightweightPlateRecognizer()
    validator = PlateFormatValidator()

    files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
    if num_samples > 0:
        files = files[:num_samples]

    total = 0
    # baseline
    baseline_correct = 0
    # +格式校验: 丢弃 rejected
    fmt_correct = 0
    fmt_answered = 0
    # +格式校验+置信度: 丢弃 rejected 和 conf < 0.8
    fmtconf_correct = 0
    fmtconf_answered = 0

    print(f'\n{"="*70}')
    print(f'后处理评估: {test_dir} ({len(files)} 张)')
    print(f'{"="*70}\n')

    for idx, fname in enumerate(files):
        stem = Path(fname).stem
        parts = stem.rsplit('_', 1)
        gt = parts[0] if len(parts) >= 2 else stem

        img = cv2.imread(os.path.join(test_dir, fname))
        if img is None:
            continue

        pred, conf = recognizer.recognize(img)
        vr = validator.validate(pred, conf)
        total += 1

        # baseline
        if pred == gt:
            baseline_correct += 1

        # +格式校验
        if vr['level'] != 'rejected':
            fmt_answered += 1
            if pred == gt:
                fmt_correct += 1

        # +格式校验+置信度
        if vr['level'] != 'rejected' and conf >= 0.8:
            fmtconf_answered += 1
            if pred == gt:
                fmtconf_correct += 1

        if (idx + 1) % 1000 == 0:
            print(f'  进度: {idx+1}/{len(files)}')

    print(f'\n{"="*70}')
    print(f'结果对比')
    print(f'{"="*70}')
    print(f'  总样本: {total}')
    print()

    def show(name, correct, answered, total):
        acc = correct / answered * 100 if answered else 0
        cov = answered / total * 100 if total else 0
        rej = total - answered
        print(f'  {name}:')
        print(f'    准确率:   {correct}/{answered} = {acc:.1f}%')
        print(f'    覆盖率:   {answered}/{total} = {cov:.1f}%  (拒答 {rej})')
        print(f'    有效准确: {correct}/{total} = {correct/total*100:.1f}%')
        print()

    show('Baseline (纯识别)', baseline_correct, total, total)
    show('+格式校验 (丢弃rejected)', fmt_correct, fmt_answered, total)
    show('+格式+置信度≥0.8', fmtconf_correct, fmtconf_answered, total)

    # 分析: 被格式校验拒绝的样本中有多少本来就是错的
    print('  格式校验过滤效果:')
    fmt_rejected_wrong = total - fmt_answered - (baseline_correct - fmt_correct)
    # 更准确地: 被拒绝的里面正确的有 baseline_correct - fmt_correct 个
    fmt_rejected_total = total - fmt_answered
    fmt_rejected_correct = baseline_correct - fmt_correct
    fmt_rejected_wrong = fmt_rejected_total - fmt_rejected_correct
    print(f'    被拒绝的 {fmt_rejected_total} 个中:')
    print(f'      本来就错的: {fmt_rejected_wrong} (正确拒绝)')
    print(f'      本来对的:   {fmt_rejected_correct} (误杀)')


def eval_refine(data_dir: str, num_samples: int = 500):
    """
    评估方案4 (精裁后识别) 在不同 ROI 扩展下的改善

    对比:
      - 直接用扩展后的粗 ROI 识别
      - 精裁后再识别
    """
    import os, sys, json
    sys.path.insert(0, os.path.dirname(__file__))
    from lightweight_plate_recognizer import LightweightPlateRecognizer
    from roi_plate_preprocessor import PreciseROIPreprocessor, expand_roi

    recognizer = LightweightPlateRecognizer()
    refiner = RefineBeforeRecognize(recognizer)

    # 加载 CCPD 全图
    ann_path = os.path.join(data_dir, 'test.json')
    img_dir = os.path.join(data_dir, 'test', 'images')
    with open(ann_path) as f:
        ann = json.load(f)
    id2ann = {a['image_id']: a for a in ann['annotations']}

    N = min(num_samples, len(ann['images']))

    # 先获取基准 (0% 扩展的识别结果)
    print(f'\n{"="*70}')
    print(f'方案4评估: 精裁后识别 vs 直接识别 (CCPD {N}张)')
    print(f'{"="*70}\n')

    print('获取基准 (0% 扩展)...')
    baseline = {}
    for idx, img_info in enumerate(ann['images'][:N]):
        a = id2ann.get(img_info['id'])
        if a is None:
            continue
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            continue
        bx, by, bw, bh = a['bbox']
        x1, y1 = int(bx), int(by)
        x2, y2 = int(bx + bw), int(by + bh)
        crop = image[y1:y2, x1:x2]
        text, conf = recognizer.recognize(crop)
        baseline[img_info['id']] = (text, conf, image, a)

    print(f'  基准: {len(baseline)} 张\n')

    # 对不同扩展比例测试
    print(f'{"扩展":>6} | {"直接识别一致率":>14} | {"精裁后一致率":>12} | {"改善":>6} | {"精裁耗时":>8}')
    print('-' * 65)

    for ratio in [0.05, 0.10, 0.20, 0.30]:
        direct_match = 0
        refine_match = 0
        count = 0
        refine_times = []

        for img_id, (gt_text, gt_conf, image, a) in baseline.items():
            bx, by, bw, bh = a['bbox']
            x1, y1 = int(bx), int(by)
            x2, y2 = int(bx + bw), int(by + bh)
            h, w = image.shape[:2]

            # 扩展 ROI
            coarse = expand_roi((x1, y1, x2, y2), ratio=ratio,
                                img_shape=image.shape)

            # 直接识别
            cx1, cy1, cx2, cy2 = coarse
            crop = image[cy1:cy2, cx1:cx2]
            direct_text, _ = recognizer.recognize(crop)

            # 精裁后识别
            rec = refiner.recognize(image, coarse)
            refine_text = rec['plate_text']
            refine_times.append(rec['refine_ms'])

            if direct_text == gt_text:
                direct_match += 1
            if refine_text == gt_text:
                refine_match += 1
            count += 1

        d_pct = direct_match / count * 100
        r_pct = refine_match / count * 100
        avg_t = np.mean(refine_times)
        print(f'{ratio*100:>5.0f}% | {direct_match:>5}/{count} = {d_pct:>5.1f}% | '
              f'{refine_match:>5}/{count} = {r_pct:>5.1f}% | '
              f'{r_pct - d_pct:>+5.1f}% | {avg_t:>6.1f}ms')


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    import argparse, os

    parser = argparse.ArgumentParser(description='车牌识别后处理评估')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--eval', type=str, metavar='DIR',
                       help='在预裁剪测试集上评估格式校验效果')
    group.add_argument('--eval-refine', type=str, metavar='DIR',
                       help='评估精裁方案 (CCPD combined 目录)')
    parser.add_argument('--num-samples', type=int, default=0)

    args = parser.parse_args()

    if args.eval:
        eval_postprocess(args.eval, args.num_samples)
    elif args.eval_refine:
        eval_refine(args.eval_refine, args.num_samples or 500)
