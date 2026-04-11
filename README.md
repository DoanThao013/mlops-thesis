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

##  Project Structure

```text
mlops-project/
├── configs/
│   ├── mlflow.service
│   └── setup_mlflow_gcp.md
├── results/
│   └── mlflow_uc1_results.csv
├── saved_models/
│   └── uc1_mnist/
│       ├── cnn_architecture.pth
│       └── simplenn_architecture.pth
└── uc1_mnist/
    ├── train_mnist.py
    └── train_mnist_cnn.py
```
<div align="center">
  <img src="https://github.com/user-attachments/assets/9912c4f6-1af1-47ce-842c-e5b6e6175b53" width="80%" alt="MLops Project Structure Tree" />
  <br>
  <em> mlops-project Stucture (Use Case 1)</em>
</div>

## UC1 MNIST Results — MLflow tracking
| Model / Scenario | Accuracy | Avg. Time (s) | Learning Rate | Batch Size |
| :--- | :---: | :---: | :---: | :---: |
|  **SimpleNN** | 0.9364 | 57.4 | 0.01 | 64 |
|  **CNN** (Best) | **0.9737** | 267.9 | 0.01 | 64 |

## Reproduction Guide

For detailed instructions on how to set up the environment and reproduce these experiments, please refer to:

👉 configs/setup_mlflow_gcp.md
