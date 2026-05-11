#!/bin/bash
# ============================================================
# Script chạy toàn bộ UC1 trên Metaflow (GCP VM)
# Chạy: bash run_uc1_all.sh
# ============================================================

set -e
cd "$(dirname "$0")/uc1_metaflow"

echo "=========================================="
echo " UC1 MNIST — Metaflow (GCP VM)"
echo "=========================================="

# --- PHẦN 1: Repeat 3 lần SimpleNN (TC7) ---
echo ""
echo ">>> SimpleNN × 3 runs (TC7)"
python3 -u train_uc1_metaflow.py run --model SimpleNN --lr 0.01 --batch_size 64
python3 -u train_uc1_metaflow.py run --model SimpleNN --lr 0.01 --batch_size 64
python3 -u train_uc1_metaflow.py run --model SimpleNN --lr 0.01 --batch_size 64

# --- PHẦN 2: Repeat 3 lần CNN (TC7) ---
echo ""
echo ">>> CNN × 3 runs (TC7)"
python3 -u train_uc1_metaflow.py run --model CNN --lr 0.01 --batch_size 64
python3 -u train_uc1_metaflow.py run --model CNN --lr 0.01 --batch_size 64
python3 -u train_uc1_metaflow.py run --model CNN --lr 0.01 --batch_size 64

# --- PHẦN 3: TC2 Config Sweep ---
echo ""
echo ">>> TC2 Config Sweep (3 configs)"
python3 -u train_uc1_metaflow.py run --model SimpleNN --lr 0.001 --batch_size 32
python3 -u train_uc1_metaflow.py run --model SimpleNN --lr 0.01 --batch_size 64
python3 -u train_uc1_metaflow.py run --model SimpleNN --lr 0.05 --batch_size 128

# --- Xem kết quả ---
echo ""
echo ">>> Kết quả:"
cd ..
python3 -u view_results.py

echo ""
echo "=========================================="
echo " ✅ UC1 Metaflow HOÀN THÀNH!"
echo "=========================================="
