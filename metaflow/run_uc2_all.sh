#!/bin/bash
# ============================================================
# Script chay toan bo UC2 tren Metaflow (GCP VM e2-medium)
# Chay: bash run_uc2_all.sh
#
# Repeat dùng default từ shared/config_phobert.py (HPARAMS).
# TC2 sweep dùng configs từ shared/config_phobert.py (TC2_CONFIGS) — đồng bộ
# với MLflow (single source of truth).
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/uc2_metaflow"

echo "=========================================="
echo " UC2 PhoBERT — Metaflow (GCP VM e2-medium)"
echo " Repeat config: tu shared.config_phobert.HPARAMS"
echo " Data: tu shared (SAMPLE_SIZE, SEED)"
echo "=========================================="

# --- PHAN 1: Repeat 3 lan (TC7) — dung default tu shared HPARAMS ---
# Moi run dung 1 training_seed khac (43, 44, 45) de co variance — sampling seed van la 42
echo ""
echo ">>> PhoBERT x 3 runs (TC7)"
python3 -u train_uc2_metaflow.py run --training_seed 43
python3 -u train_uc2_metaflow.py run --training_seed 44
python3 -u train_uc2_metaflow.py run --training_seed 45

# --- PHAN 2: TC2 Config Sweep (read tu shared/) ---
echo ""
echo ">>> TC2 Config Sweep (3 configs from shared.config_phobert.TC2_CONFIGS)"
while IFS= read -r flags; do
    [ -z "$flags" ] && continue
    echo ">>> python3 train_uc2_metaflow.py run $flags"
    eval "python3 -u train_uc2_metaflow.py run $flags"
done < <(python3 "$SCRIPT_DIR/_print_tc2_uc2.py")

# --- Xem ket qua ---
echo ""
echo ">>> Ket qua:"
cd "$SCRIPT_DIR"
python3 -u view_results.py

echo ""
echo "=========================================="
echo " UC2 Metaflow HOAN THANH!"
echo "=========================================="
