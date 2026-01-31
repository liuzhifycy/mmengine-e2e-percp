#!/usr/bin/env python
"""
End-to-end License Plate Recognition Pipeline

This script combines:
1. YOLO11 for plate detection (our trained model)
2. HyperLPR3 for plate recognition (pre-trained OCR)

Usage:
    python inference_pipeline.py --image path/to/image.jpg
    python inference_pipeline.py --image_dir path/to/images/ --output_dir path/to/output/
"""

import argparse
import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import hyperlpr3 as lpr3


def get_chinese_font(size=24):
    """Get a font that supports Chinese characters."""
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    # Fallback to default font
    return ImageFont.load_default()


def put_chinese_text(img, text, position, font_size=24, color=(0, 255, 0)):
    """
    Draw Chinese text on an OpenCV image using PIL.
    
    Args:
        img: OpenCV image (BGR format)
        text: Text to draw (supports Chinese)
        position: (x, y) position for text
        font_size: Font size
        color: Text color in BGR format
        
    Returns:
        Modified image
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Get font
    font = get_chinese_font(font_size)
    
    # Convert BGR color to RGB
    color_rgb = (color[2], color[1], color[0])
    
    # Draw text
    draw.text(position, text, font=font, fill=color_rgb)
    
    # Convert back to BGR
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result

# Add mmengine-lite to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector, inference_detector


class PlateRecognitionPipeline:
    """End-to-end license plate recognition pipeline."""
    
    def __init__(
        self,
        yolo_config: str = None,
        yolo_checkpoint: str = None,
        device: str = 'cuda:0',
        use_yolo_detection: bool = True,
        conf_threshold: float = 0.5,
    ):
        """
        Initialize the pipeline.
        
        Args:
            yolo_config: Path to YOLO11 config file
            yolo_checkpoint: Path to YOLO11 checkpoint
            device: Device to run inference on
            use_yolo_detection: Whether to use YOLO11 for detection (vs HyperLPR3's built-in)
            conf_threshold: Confidence threshold for detection
        """
        self.device = device
        self.use_yolo_detection = use_yolo_detection
        self.conf_threshold = conf_threshold
        
        # Initialize HyperLPR3 for recognition
        self.lpr = lpr3.LicensePlateCatcher()
        
        # Initialize YOLO11 detector if using our trained model
        if use_yolo_detection and yolo_config and yolo_checkpoint:
            print(f"Loading YOLO11 detector from {yolo_checkpoint}")
            self.detector = init_detector(yolo_config, yolo_checkpoint, device=device)
        else:
            self.detector = None
            print("Using HyperLPR3's built-in detection")
    
    def detect_plates_yolo(self, image: np.ndarray) -> list:
        """
        Detect plates using YOLO11.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of [x1, y1, x2, y2, confidence] for each detected plate
        """
        result = inference_detector(self.detector, image)
        
        # Extract bounding boxes
        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        
        # Filter by confidence
        detections = []
        for bbox, score in zip(bboxes, scores):
            if score >= self.conf_threshold:
                x1, y1, x2, y2 = bbox
                detections.append([int(x1), int(y1), int(x2), int(y2), float(score)])
        
        return detections
    
    def recognize_plate_region(self, plate_img: np.ndarray) -> tuple:
        """
        Recognize text in a cropped plate region.
        
        Args:
            plate_img: Cropped plate image
            
        Returns:
            (plate_text, confidence) or (None, 0) if recognition fails
        """
        # Use HyperLPR3 on the cropped region
        results = self.lpr(plate_img)
        
        if results and len(results) > 0:
            # Return the first (best) result
            plate_text = results[0][0]
            confidence = results[0][1]
            return plate_text, confidence
        
        return None, 0.0
    
    def correct_plate_perspective(self, image: np.ndarray, bbox: list) -> np.ndarray:
        """
        Apply perspective correction to plate region (placeholder for future enhancement).
        Currently just crops the region with some padding.
        
        Args:
            image: Full image
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            Cropped and optionally corrected plate image
        """
        x1, y1, x2, y2 = bbox[:4]
        h, w = image.shape[:2]
        
        # Add some padding
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.1)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        plate_img = image[y1:y2, x1:x2]
        
        return plate_img
    
    def process_image(self, image_path: str) -> dict:
        """
        Process a single image through the full pipeline.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with detection and recognition results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Failed to read image: {image_path}'}
        
        result = {
            'image_path': image_path,
            'image_size': [image.shape[1], image.shape[0]],
            'plates': []
        }
        
        if self.use_yolo_detection and self.detector:
            # Use our trained YOLO11 for detection
            detections = self.detect_plates_yolo(image)
            
            for det in detections:
                x1, y1, x2, y2, det_conf = det
                
                # Crop and correct plate region
                plate_img = self.correct_plate_perspective(image, [x1, y1, x2, y2])
                
                # Recognize plate text
                plate_text, rec_conf = self.recognize_plate_region(plate_img)
                
                result['plates'].append({
                    'bbox': [x1, y1, x2, y2],
                    'detection_confidence': det_conf,
                    'plate_number': plate_text,
                    'recognition_confidence': rec_conf
                })
        else:
            # Use HyperLPR3's built-in detection + recognition
            results = self.lpr(image)
            for res in results:
                plate_text, confidence, plate_type, bbox = res[0], res[1], res[2], res[3]
                result['plates'].append({
                    'bbox': [int(x) for x in bbox],
                    'detection_confidence': float(confidence),
                    'plate_number': plate_text,
                    'recognition_confidence': float(confidence),
                    'plate_type': int(plate_type)
                })
        
        return result
    
    def visualize_result(self, image_path: str, result: dict, output_path: str = None,
                         gt_plate: str = None, show_comparison: bool = False):
        """
        Visualize detection and recognition results with Chinese text support.
        
        Args:
            image_path: Path to input image
            result: Pipeline result dictionary
            output_path: Path to save visualization (optional)
            gt_plate: Ground truth plate number (optional)
            show_comparison: Whether to show GT vs Pred comparison
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        font_size = 28
        
        for plate in result.get('plates', []):
            bbox = plate['bbox']
            x1, y1, x2, y2 = bbox[:4] if len(bbox) == 4 else bbox
            
            # Draw bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            
            # Draw plate number using PIL for Chinese support
            plate_text = plate.get('plate_number', 'Unknown')
            if plate_text:
                # Prepare display text
                conf = plate.get('recognition_confidence', 0)
                if show_comparison and gt_plate:
                    is_correct = (plate_text == gt_plate)
                    status = "OK" if is_correct else "X"
                    display_text = f"Pred: {plate_text} [{status}]"
                    color = (0, 255, 0) if is_correct else (0, 0, 255)
                else:
                    display_text = f"{plate_text} ({conf:.2f})"
                    color = (0, 255, 0)
                
                # Draw background rectangle for text
                text_y = max(int(y1) - 35, 5)
                cv2.rectangle(image, (int(x1), text_y), 
                            (int(x1) + len(display_text) * 18, text_y + 32), 
                            (255, 255, 255), -1)
                
                # Draw text using PIL
                image = put_chinese_text(image, display_text, 
                                        (int(x1) + 2, text_y + 2), 
                                        font_size=font_size, color=color)
        
        # Draw ground truth at the bottom if provided
        if gt_plate:
            h, w = image.shape[:2]
            gt_text = f"GT: {gt_plate}"
            cv2.rectangle(image, (0, h - 40), (len(gt_text) * 18 + 10, h), (255, 255, 255), -1)
            image = put_chinese_text(image, gt_text, (5, h - 35), font_size=font_size, color=(255, 128, 0))
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image


def parse_ccpd_filename(filename: str) -> str:
    """
    Parse ground truth plate number from CCPD filename format.
    
    CCPD filename format:
    tilt_degree-x1&y1_x2&y2-bbox...-plate_indices-...
    
    Plate indices are encoded as integers mapping to Chinese characters.
    """
    # Character mapping
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
                "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
                "新", "警", "学", "O"]
    
    chars = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
            "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
            "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
            "6", "7", "8", "9", "O"]
    
    try:
        # Extract plate indices from filename
        parts = filename.replace('.jpg', '').split('-')
        if len(parts) >= 5:
            plate_indices = parts[4].split('_')
            
            # First index is province
            plate_number = provinces[int(plate_indices[0])]
            
            # Remaining indices are alphanumeric characters
            for idx in plate_indices[1:]:
                plate_number += chars[int(idx)]
            
            return plate_number
    except (IndexError, ValueError):
        pass
    
    return None


def evaluate_on_test_set(pipeline, test_dir: str, max_samples: int = None,
                         output_dir: str = None, save_visualizations: bool = False) -> dict:
    """
    Evaluate the pipeline on the test set.
    
    Args:
        pipeline: PlateRecognitionPipeline instance
        test_dir: Directory containing test images
        max_samples: Maximum number of samples to evaluate
        output_dir: Directory to save visualizations
        save_visualizations: Whether to save visualization images
        
    Returns:
        Evaluation metrics dictionary
    """
    from tqdm import tqdm
    
    # Get test images
    image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    if max_samples:
        image_files = image_files[:max_samples]
    
    total = 0
    detection_correct = 0
    recognition_correct = 0
    partial_correct = 0  # At least 5 characters match
    
    results = []
    
    # Create output directories if saving visualizations
    if save_visualizations and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'correct'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'wrong'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'not_detected'), exist_ok=True)
    
    for img_file in tqdm(image_files, desc="Evaluating"):
        img_path = os.path.join(test_dir, img_file)
        
        # Get ground truth from filename
        gt_plate = parse_ccpd_filename(img_file)
        if gt_plate is None:
            continue
        
        total += 1
        
        # Run inference
        result = pipeline.process_image(img_path)
        
        is_correct = False
        is_detected = False
        pred_plate = None
        
        if result.get('plates'):
            is_detected = True
            detection_correct += 1
            pred_plate = result['plates'][0].get('plate_number', '')
            
            if pred_plate == gt_plate:
                recognition_correct += 1
                is_correct = True
            elif pred_plate and len(pred_plate) >= 5:
                # Check partial match (at least 5 characters)
                matches = sum(1 for p, g in zip(pred_plate, gt_plate) if p == g)
                if matches >= 5:
                    partial_correct += 1
        
        # Save visualization
        if save_visualizations and output_dir:
            if is_correct:
                vis_dir = os.path.join(output_dir, 'correct')
            elif is_detected:
                vis_dir = os.path.join(output_dir, 'wrong')
            else:
                vis_dir = os.path.join(output_dir, 'not_detected')
            
            vis_path = os.path.join(vis_dir, img_file)
            pipeline.visualize_result(img_path, result, vis_path, 
                                      gt_plate=gt_plate, show_comparison=True)
        
        results.append({
            'image': img_file,
            'ground_truth': gt_plate,
            'prediction': pred_plate,
            'detected': is_detected,
            'correct': is_correct
        })
    
    metrics = {
        'total_samples': total,
        'detection_rate': detection_correct / total if total > 0 else 0,
        'recognition_accuracy': recognition_correct / total if total > 0 else 0,
        'partial_accuracy': (recognition_correct + partial_correct) / total if total > 0 else 0,
    }
    
    return metrics, results


def main():
    parser = argparse.ArgumentParser(description='License Plate Recognition Pipeline')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--image_dir', type=str, help='Directory of images to process')
    parser.add_argument('--output_dir', type=str, help='Output directory for visualizations')
    parser.add_argument('--yolo_config', type=str, 
                       default='/home/ubuntu/e2e-pecp-pdp/mmengine-lite/configs/plate_recognition/yolo11m_plate_detection.py',
                       help='Path to YOLO11 config')
    parser.add_argument('--yolo_checkpoint', type=str,
                       default='/home/ubuntu/e2e-pecp-pdp/mmengine-lite/work_dirs/yolo11m_plate_detection/best_coco_plate_precision_epoch_15.pth',
                       help='Path to YOLO11 checkpoint')
    parser.add_argument('--use_hyperlpr_detection', action='store_true',
                       help='Use HyperLPR3 built-in detection instead of YOLO11')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate on test set')
    parser.add_argument('--test_dir', type=str,
                       default='/home/ubuntu/e2e-pecp-pdp/mmengine-lite/data/ccpd/combined/test/images/',
                       help='Test image directory')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples for evaluation')
    parser.add_argument('--save_vis', action='store_true',
                       help='Save visualization images during evaluation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    use_yolo = not args.use_hyperlpr_detection
    pipeline = PlateRecognitionPipeline(
        yolo_config=args.yolo_config if use_yolo else None,
        yolo_checkpoint=args.yolo_checkpoint if use_yolo else None,
        device=args.device,
        use_yolo_detection=use_yolo,
        conf_threshold=args.conf_threshold,
    )
    
    if args.evaluate:
        # Evaluation mode
        print(f"\nEvaluating on test set: {args.test_dir}")
        print(f"Save visualizations: {args.save_vis}")
        metrics, results = evaluate_on_test_set(
            pipeline, args.test_dir, args.max_samples,
            output_dir=args.output_dir, save_visualizations=args.save_vis
        )
        
        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Detection rate: {metrics['detection_rate']:.4f} ({metrics['detection_rate']*100:.2f}%)")
        print(f"Recognition accuracy: {metrics['recognition_accuracy']:.4f} ({metrics['recognition_accuracy']*100:.2f}%)")
        print(f"Partial accuracy (>=5 chars): {metrics['partial_accuracy']:.4f} ({metrics['partial_accuracy']*100:.2f}%)")
        
        # Save results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
                json.dump({'metrics': metrics, 'results': results}, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output_dir}/evaluation_results.json")
            if args.save_vis:
                print(f"Visualizations saved to {args.output_dir}/[correct|wrong|not_detected]/")
    
    elif args.image:
        # Single image mode
        result = pipeline.process_image(args.image)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, os.path.basename(args.image))
            pipeline.visualize_result(args.image, result, output_path)
    
    elif args.image_dir:
        # Batch processing mode
        image_files = [f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        for img_file in image_files[:10]:  # Process first 10 for demo
            img_path = os.path.join(args.image_dir, img_file)
            result = pipeline.process_image(img_path)
            
            print(f"\n{img_file}:")
            for plate in result.get('plates', []):
                print(f"  Plate: {plate.get('plate_number', 'Unknown')}, "
                      f"Confidence: {plate.get('recognition_confidence', 0):.4f}")
            
            if args.output_dir:
                output_path = os.path.join(args.output_dir, img_file)
                pipeline.visualize_result(img_path, result, output_path)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
