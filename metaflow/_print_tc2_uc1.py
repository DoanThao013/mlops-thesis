"""
Helper: in ra danh sách lệnh `python3 train_uc1_metaflow.py run ...` cho TC2 sweep,
đọc từ shared.models_mnist.TC2_CONFIGS. Dùng bởi run_uc1_all.sh.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.models_mnist import TC2_CONFIGS

for cfg in TC2_CONFIGS:
    print(f"--lr {cfg['lr']} --batch_size {cfg['batch_size']} --epochs {cfg['epochs']}")
