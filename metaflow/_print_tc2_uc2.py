"""
Helper: in ra danh sách flags `--lr ... --batch_size ... --epochs ...` cho TC2 sweep,
đọc từ shared.config_phobert.TC2_CONFIGS. Dùng bởi run_uc2_all.sh.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config_phobert import TC2_CONFIGS

for cfg in TC2_CONFIGS:
    print(f"--lr {cfg['lr']} --batch_size {cfg['batch_size']} --epochs {cfg['epochs']}")
