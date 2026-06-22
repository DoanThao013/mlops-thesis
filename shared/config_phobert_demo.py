"""
Shared PhoBERT fine-tuning config — DEMO VERSION (epochs=1, NUM_RUNS=1)
Dung de chay demo minh hoa pipeline, KHONG dung de lay so lieu bao cao.
So lieu chinh thuc lay tu config_phobert.py (epochs=3, NUM_RUNS=3).
"""

# --- MODEL ---
PHOBERT_MODEL_NAME = "vinai/phobert-base"
NUM_LABELS = 3          # UIT-VSFC: 0=NEG, 1=NEU, 2=POS
MAX_LENGTH = 128        # Giam tu 256 de tiet kiem RAM tren e2-medium

# --- HYPERPARAMETERS ---
HPARAMS = {
    "lr":           2e-5,
    "batch_size":   2,          # Batch nho de vua RAM 4GB
    "epochs":       1,          # DEMO: giam tu 3 → 1 de tiet kiem thoi gian
    "optimizer":    "AdamW",
    "weight_decay": 0.01,
    "max_length":   MAX_LENGTH,
    "gradient_accumulation_steps": 8,   # Effective batch = 2 x 8 = 16
    "gradient_checkpointing": True,
}

# --- DATASET & SAMPLING ---
DATASET_NAME  = "uitnlp/vietnamese_students_feedback"
SAMPLE_SIZE   = 500     # Giam data de chay duoc tren e2-medium trong thoi gian hop ly
SEED          = 42      # Co dinh seed dam bao tat ca framework lay cung mau
STRATIFIED    = True    # Lay mau phan tang theo nhan NEG/NEU/POS

# --- EXPERIMENT ---
NUM_RUNS      = 1       # DEMO: giam tu 3 → 1, chi can 1 run de minh hoa
LABEL_COL     = "sentiment"
TEXT_COL      = "sentence"

# TC2 — config sweep
TC2_CONFIGS = [
    {"lr": 2e-5, "batch_size": 2, "epochs": 1},  # DEMO: chi chay 1 config toi uu nhat
]

# Label mapping
LABEL_MAP = {0: "NEG", 1: "NEU", 2: "POS"}
