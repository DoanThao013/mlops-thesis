![Python](https://img.shields.io/badge/python-3.10-blue) ![MLflow](https://img.shields.io/badge/mlflow-2.19.0-important) ![PhoBERT](https://img.shields.io/badge/PhoBERT-base-green)
# MLOps Thesis — Experimental Evaluation of MLOps Platforms

##  Team Information
* **Doan Thanh Thao** – 23521466
* **Duong Thanh Huyen** – 23520659
* **Supervisor:** M.Sc. Le Anh Tuan

---

##  Experimental Environment
* **MLflow Server:** Google Cloud Platform (GCP) - `e2-medium` (2 vCPU, 4GB RAM)
* **OS:** Ubuntu 22.04 LTS
* **Python:** 3.10
* **MLflow:** 2.19.0
* **UC1 Training:** GCP CPU (PyTorch 2.11.0+cpu)
* **UC2 Training:** GCP VM `e2-medium` CPU (PyTorch + HuggingFace Transformers, gradient accumulation)

---

##  Tech Stack & Tools
- **DVC (Data Version Control)**: Manages data versioning and prevents large datasets from being pushed to GitHub.
- **MLflow**: Tracks hyperparameters, metrics, and manages model artifacts/registry.
- **GCP (Google Cloud Platform)**: Remote environment for hosting the MLflow tracking server.
- **GCP VM (e2-medium)**: CPU-only environment for both UC1 and UC2 training. UC2 uses gradient accumulation (batch=2, accum=8, effective batch=16) and gradient checkpointing to fit within 4GB RAM.
- **HuggingFace Transformers**: Pre-trained PhoBERT model and tokenizer for Vietnamese NLP.

---

##  Project Structure

```text
mlops-thesis/
├── configs/                    # System configurations and setup guides
│   ├── mlflow.service          # MLflow Server systemd service configuration
│   └── setup_mlflow_gcp.md    # Documentation for GCP environment setup
├── results/                    # Experimental artifacts and logs
│   ├── uc1_mlflow_run_log.txt         # UC1 pipeline execution logs
│   ├── mlflow_final_backup.db.gz     # Compressed backup of MLflow SQLite database
│   └── mlflow_uc2_results.csv        # UC2 PhoBERT results (6 runs)
├── uc1_mnist/                  # Source code for Use Case 1
│   ├── data/                   # Data directory (Managed by DVC)
│   │   └── MNIST.dvc           # DVC data pointer file
│   └── train_uc1.py           # UC1 training pipeline script
├── uc2_phobert/                # Source code for Use Case 2
│   ├── data/                   # Data directory (UIT-VSFC, gitignored)
│   └── train_uc2.py           # UC2 training pipeline script (GCP VM CPU)
├── .gitignore
├── requirements.txt            # Project dependencies
└── README.md
```

---

## UC1 MNIST Results — MLflow Tracking

Training on GCP CPU (`e2-medium`). 9 automated runs including hyperparameter sweeps:

| Model / Scenario | Accuracy | TC7 (Avg. Time) | Learning Rate | Batch Size |
| :--- | :---: | :---: | :---: | :---: |
| **SimpleNN** | 0.9348 | 93.0s | 0.01 | 64 |
| **CNN (Best)** | **0.9770** | 452.4s | 0.01 | 64 |

> **Analysis:** CNN achieved 97.70% accuracy but took 4.8x longer than SimpleNN due to architectural complexity and CPU-only training.

---

## UC2 PhoBERT Vietnamese Sentiment — MLflow Tracking

Training on GCP VM `e2-medium` CPU. Stratified 500-sample subset (seed=42). 6 runs (3 repeat + 3 TC2 config sweep):

| Run | Learning Rate | Batch Size | Effective Batch | Accuracy | F1-macro | TC7 (Time) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| PhoBERT_run1 | 2e-5 | 2 | 16 | 0.9315 | 0.9277 | 743.5s |
| PhoBERT_run2 | 2e-5 | 2 | 16 | 0.9340 | 0.9311 | 756.0s |
| PhoBERT_run3 | 2e-5 | 2 | 16 | 0.9340 | 0.9311 | 786.8s |
| TC2_lr1e-05 | 1e-5 | 2 | 16 | 0.9292 | 0.9270 | 930.0s |
| TC2_lr2e-05 | 2e-5 | 2 | 16 | 0.9340 | 0.9311 | 932.7s |
| TC2_lr3e-05 | **3e-5** | **4** | **32** | **0.9340** | 0.9289 | **827.3s** |

**Summary:**
| Metric | Value |
| :--- | :---: |
| Best Accuracy | **93.40%** |
| Average TC7 (3 repeat) | **762.1s** (~12.7 min) |
| Dataset | UIT-VSFC (500 stratified samples, 3 classes) |
| Model | `vinai/phobert-base` |
| Platform | GCP VM e2-medium CPU (gradient_accumulation_steps=8) |

> **Analysis:** PhoBERT achieved 93.4% accuracy on Vietnamese sentiment classification. The model showed high stability across repeat runs (±0.003). Learning rate 2e-5 proved optimal; lr=1e-5 slightly underperformed. Per-device batch size is 2 with gradient accumulation steps=8, giving effective batch size=16 to fit within 4GB RAM.

---

## Reproduction Guide

### UC1 (MNIST):
```bash
# On GCP VM:
cd uc1_mnist
python3 -u train_uc1.py --mode all
```

### UC2 (PhoBERT):
```bash
# On GCP VM e2-medium:
cd uc2_phobert
python3 -u train_uc2.py --mode all
```

### Environment setup:
👉 `configs/setup_mlflow_gcp.md`
