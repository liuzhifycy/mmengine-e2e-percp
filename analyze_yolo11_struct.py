import sys
from ultralytics import YOLO

def analyze_model():
    print("Loading YOLO11m model...")
    try:
        model = YOLO("yolo11m.pt")
        # 强制初始化
        _ = model.info()
        
        print("\n" + "="*80)
        print("YAML Configuration:")
        print("="*80)
        
        # 直接打印 YAML 配置
        # 通常存储在 model.model.yaml 或 model.yaml 中
        if hasattr(model.model, 'yaml'):
            import pprint
            pprint.pprint(model.model.yaml)
        else:
            print("No YAML config found on model object.")

    except Exception as e:
        print(f"Error analyzing model: {e}")

if __name__ == "__main__":
    analyze_model()
