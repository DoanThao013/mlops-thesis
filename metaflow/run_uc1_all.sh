#!/bin/bash
# ============================================================
# Script chạy toàn bộ UC1 trên Metaflow (GCP VM)
# Chạy: bash run_uc1_all.sh
#
# Repeat dùng default từ shared/models_mnist.py (HPARAMS).
# TC2 sweep dùng configs từ shared/models_mnist.py (TC2_CONFIGS) — đồng bộ
# với MLflow (single source of truth).
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/uc1_metaflow"

echo "=========================================="
echo " UC1 MNIST — Metaflow (GCP VM)"
echo "=========================================="

# --- PHAN 1: Repeat 3 lan moi model (TC7) — dung default tu shared HPARAMS ---
echo ""
echo ">>> SimpleNN x 3 runs (TC7)"
python3 -u train_uc1_metaflow.py run --model SimpleNN
python3 -u train_uc1_metaflow.py run --model SimpleNN
python3 -u train_uc1_metaflow.py run --model SimpleNN

echo ""
echo ">>> DeepNN x 3 runs (TC7)"
python3 -u train_uc1_metaflow.py run --model DeepNN
python3 -u train_uc1_metaflow.py run --model DeepNN
python3 -u train_uc1_metaflow.py run --model DeepNN

echo ""
echo ">>> CNN x 3 runs (TC7)"
python3 -u train_uc1_metaflow.py run --model CNN
python3 -u train_uc1_metaflow.py run --model CNN
python3 -u train_uc1_metaflow.py run --model CNN

# --- PHAN 2: TC2 Config Sweep (read tu shared/) ---
echo ""
echo ">>> TC2 Config Sweep (3 configs from shared.models_mnist.TC2_CONFIGS)"
while IFS= read -r flags; do
    [ -z "$flags" ] && continue
    echo ">>> python3 train_uc1_metaflow.py run --model SimpleNN $flags"
    eval "python3 -u train_uc1_metaflow.py run --model SimpleNN $flags"
done < <(python3 "$SCRIPT_DIR/_print_tc2_uc1.py")

# --- Xem ket qua ---
echo ""
echo ">>> Ket qua:"
cd "$SCRIPT_DIR"
python3 -u view_results.py

echo ""
echo "=========================================="
echo " UC1 Metaflow HOAN THANH!"
echo "=========================================="
