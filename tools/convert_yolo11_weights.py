import torch
import sys
from ultralytics import YOLO
from mmlite.models.detectors.yolo11 import YOLO11

def convert():
    print("Loading Official YOLO11m...")
    official_model = YOLO("yolo11m.pt").model
    official_sd = official_model.state_dict()
    
    print("Initializing MMYOLO11...")
    mm_model = YOLO11()
    mm_sd = mm_model.state_dict()
    
    new_sd = {}
    
    # -------------------------------------------------
    # Helper to copy Conv weights
    # -------------------------------------------------
    def copy_conv(src_prefix, dst_prefix):
        # Conv2d
        new_sd[f"{dst_prefix}.conv.conv.weight"] = official_sd[f"{src_prefix}.conv.weight"]
        # BN
        new_sd[f"{dst_prefix}.conv.bn.weight"] = official_sd[f"{src_prefix}.bn.weight"]
        new_sd[f"{dst_prefix}.conv.bn.bias"] = official_sd[f"{src_prefix}.bn.bias"]
        new_sd[f"{dst_prefix}.conv.bn.running_mean"] = official_sd[f"{src_prefix}.bn.running_mean"]
        new_sd[f"{dst_prefix}.conv.bn.running_var"] = official_sd[f"{src_prefix}.bn.running_var"]
        new_sd[f"{dst_prefix}.conv.bn.num_batches_tracked"] = official_sd[f"{src_prefix}.bn.num_batches_tracked"]

    # -------------------------------------------------
    # Helper to copy C3k2/C2f/C3 weights
    # -------------------------------------------------
    def copy_c3k2(src_prefix, dst_prefix, n_blocks):
        # cv1, cv2
        copy_conv(f"{src_prefix}.cv1", f"{dst_prefix}.cv1")
        copy_conv(f"{src_prefix}.cv2", f"{dst_prefix}.cv2")
        
        # Bottlenecks in m
        for i in range(n_blocks):
            # Check if it is a Bottleneck or C3k or PSABlock
            # Simplest way: try to copy cv1/cv2
            # If C3k/Bottleneck
            if f"{src_prefix}.m.{i}.cv1.conv.weight" in official_sd:
                copy_conv(f"{src_prefix}.m.{i}.cv1", f"{dst_prefix}.m.{i}.cv1")
                copy_conv(f"{src_prefix}.m.{i}.cv2", f"{dst_prefix}.m.{i}.cv2")
            
            # If PSABlock (cv1, cv2, attn, ffn)
            # PSABlock in my implementation: attn, ffn. 
            # Wait, my PSABlock doesn't have cv1/cv2 at top level? 
            # Let's check my PSABlock implementation.
            # My PSABlock: self.attn, self.ffn.
            # Ultra PSABlock: self.attn, self.ffn. (Based on block.py)
            # Ultra Attention: qkv, proj, pe. All Conv.
            # Ultra FFN: Sequential(Conv, Conv).
            
            # Let's handle PSABlock case explicitly if needed.
            # Or just rely on recursion if I wrote a recursive copier? 
            # No, flat mapping is safer for now.

    # Since C3k2 structure is complex (nested C3k, Bottleneck, PSA),
    # I will use a smarter approach: Recursive Key Mapping
    # Because "layer.0" in Ultra maps to "backbone.stem.0" in Mine.
    # I can just replace the prefix!
    
    # Define Prefix Map
    prefix_map = {
        "model.0": "backbone.stem.0",
        "model.1": "backbone.stem.1",
        "model.2": "backbone.stage1.0",
        "model.3": "backbone.stage1.1",
        "model.4": "backbone.stage2.0",
        "model.5": "backbone.stage2.1",
        "model.6": "backbone.stage3.0",
        "model.7": "backbone.stage3.1",
        "model.8": "backbone.stage4.0",
        "model.9": "backbone.stage4.1",
        "model.10": "backbone.stage4.2",
        
        # Neck
        "model.13": "neck.c3k2_1",
        "model.16": "neck.c3k2_2",
        "model.17": "neck.down1",
        "model.19": "neck.c3k2_3",
        "model.20": "neck.down2",
        "model.22": "neck.c3k2_4",
        
        # Head
        "model.23": "bbox_head",
    }
    
    # -------------------------------------------------
    # Main Loop
    # -------------------------------------------------
    for key, val in official_sd.items():
        # 1. Determine which layer this belongs to
        # Key format: "model.X.rest_of_key"
        parts = key.split('.')
        layer_idx = int(parts[1])
        layer_key = f"model.{layer_idx}"
        
        if layer_key not in prefix_map:
            # Upsample, Concat layers have no weights
            continue
            
        dst_layer_prefix = prefix_map[layer_key]
        rest_of_key = '.'.join(parts[2:])
        
        # 2. Handle ConvModule wrapping difference
        # Ultra: cv1.conv.weight -> Mine: cv1.conv.conv.weight
        # Ultra: cv1.bn.weight   -> Mine: cv1.conv.bn.weight
        
        new_key_suffix = rest_of_key
        
        # Replace 'conv.weight' -> 'conv.conv.weight'
        # Replace 'bn.weight' -> 'conv.bn.weight'
        # CAREFUL: "conv" might appear multiple times or not be a YOLOConv
        
        # Heuristic:
        # If the parameter belongs to a YOLOConv, it will look like '...conv.weight' or '...bn.weight'
        # In my implementation, almost all Convs are YOLOConvs (wrapped in ConvModule).
        # Exception: DFL (pure nn.Conv2d), Head's final Conv2d (pure nn.Conv2d).
        
        is_pure_conv = False
        # Head final convs are pure in Ultra: model.23.cv2.I.2 (Conv2d)
        if layer_idx == 23:
            # cv2.X.2 is the final 1x1 conv
            if "cv2" in rest_of_key and rest_of_key.endswith(".2.weight"): is_pure_conv = True
            if "cv2" in rest_of_key and rest_of_key.endswith(".2.bias"): is_pure_conv = True
            if "cv3" in rest_of_key and rest_of_key.endswith(".2.weight"): is_pure_conv = True
            if "cv3" in rest_of_key and rest_of_key.endswith(".2.bias"): is_pure_conv = True
            if "dfl" in rest_of_key: is_pure_conv = True
            
        if not is_pure_conv:
            if "conv.weight" in rest_of_key:
                new_key_suffix = new_key_suffix.replace("conv.weight", "conv.conv.weight")
            elif "bn." in rest_of_key:
                new_key_suffix = new_key_suffix.replace("bn.", "conv.bn.")
                
        # Special case: SPPF
        # model.9.m (MaxPool) -> no weights
        # model.9.cv1 -> Wrapped
        
        # Special case: Attention (in C2PSA)
        # qkv, proj, pe are YOLOConvs in my impl.
        # Ultra: qkv (Conv) -> qkv.conv, qkv.bn
        # Mine: qkv (YOLOConv) -> qkv.conv.conv, qkv.conv.bn
        # So the heuristic above works!
        
        final_key = f"{dst_layer_prefix}.{new_key_suffix}"
        
        # 3. Validation
        if final_key in mm_sd:
            if mm_sd[final_key].shape == val.shape:
                new_sd[final_key] = val
            else:
                print(f"SHAPE MISMATCH: {key} -> {final_key}")
                print(f"Official: {val.shape}, Mine: {mm_sd[final_key].shape}")
        else:
            print(f"MISSING KEY: {final_key} (derived from {key})")
            # Try to debug why
            # Maybe some module names are different?
            pass

    print(f"Converted {len(new_sd)} / {len(mm_sd)} keys.")
    
    # Save
    torch.save(new_sd, "yolo11m_mm.pth")
    print("Saved to yolo11m_mm.pth")

if __name__ == "__main__":
    convert()
