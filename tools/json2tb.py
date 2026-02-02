import json
import sys
import os
import time
from torch.utils.tensorboard import SummaryWriter

def convert(json_path, log_dir, follow=False):
    writer = SummaryWriter(log_dir)
    print(f"Monitoring {json_path} -> {log_dir}")
    
    with open(json_path, 'r') as f:
        # Initial read
        while True:
            line = f.readline()
            if not line:
                if not follow:
                    break
                time.sleep(1)
                continue
            
            try:
                data = json.loads(line)
                step = data.get('step', data.get('iter', 0))
                for key, value in data.items():
                    if isinstance(value, (int, float)) and key not in ['step', 'epoch', 'iter']:
                        writer.add_scalar(key, value, step)
            except json.JSONDecodeError:
                continue
                
    writer.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python json2tb.py <json_path> <log_dir> [--follow]")
        sys.exit(1)
    
    follow = "--follow" in sys.argv
    convert(sys.argv[1], sys.argv[2], follow)
