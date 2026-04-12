![Python](https://img.shields.io/badge/python-3.10-blue) ![MLflow](https://img.shields.io/badge/mlflow-2.19.0-important)
# MLOps Thesis — Experimental Evaluation of MLOps Platforms

##  Team Information
* **Doan Thanh Thao** – 23521466
* **Duong Thanh Huyen** – 23520659
* **Supervisor:** M.Sc. Le Anh Tuan

---

##  Experimental Environment
* **Platform:** Google Cloud Platform (GCP) - `e2-medium` (2 vCPU, 4GB RAM)
* **OS:** Ubuntu 22.04 LTS
* **Python:** 3.10
* **MLflow:** 2.19.0
* **PyTorch:** 2.11.0+cpu

---

##  Teck Stack & Tool
- **DVC (Data Version Control)**: Manages data versioning and prevents large datasets from being pushed to GitHub.
- **MLflow**: Tracks hyperparameters, metrics, and manages model artifacts/registry.
- **GCP (Google Cloud Platform)**: Remote environment for hosting the MLflow tracking server.
---

##  Project Structure

```text
mlops-project/
├── configs/                # System configurations and setup guides
│   ├── mlflow.service      # MLflow Server systemd service configuration
│   └── setup_mlflow_gcp.md # Documentation for GCP environment setup
├── results/                # Experimental artifacts and logs
│   ├── uc1_mlflow_run_log.txt      # Detailed pipeline execution logs
│   └── mlflow_final_backup.db.gz   # Compressed backup of MLflow SQLite database
├── saved_models/           # Local storage for trained model weights (.pth)
│   └── uc1_mnist/          # Use Case 1 (MNIST) model artifacts
├── uc1_mnist/              # Source code for Use Case 1
│   ├── data/               # Data directory (Managed by DVC)
│   │   └── MNIST.dvc       # DVC data pointer file
│   └── train_uc1.py        # Centralized training pipeline script
└── requirements.txt        # Project dependencies
```
<div align="center">
  <img src="https://github.com/user-attachments/assets/fd8484cf-9a94-45da-b70b-37149b5fe370" width="80%" alt="MLops Project Structure Tree" />
  <br>
  <em> mlops-project Stucture (Use Case 1)</em>
</div>

## UC1 MNIST Results — MLflow tracking
The results below were extracted from 9 automated runs (including hyperparameter sweeps and average time measurement) using train_uc1.py on GCP CPU:

| Model / Scenario | Accuracy | TC7 (Avg. Time) | Learning Rate | Batch Size |
| :--- | :---: | :---: | :---: | :---: |
| **SimpleNN** | 0.9348 | 93.0s | 0.01 | 64 |
| **CNN (Best)** | **0.9770** | 452.4s | 0.01 | 64 |

> **Analysis:** The CNN model achieved superior accuracy (97.70%). However, due to its architectural complexity and the lack of GPU acceleration on the GCP instance, its training time was approximately 4.8x longer than the SimpleNN model, highlighting a significant accuracy-performance trade-off.

## Reproduction Guide

For detailed instructions on how to set up the environment and reproduce these experiments, please refer to:

👉 configs/setup_mlflow_gcp.md
