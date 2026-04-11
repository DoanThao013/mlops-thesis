# MLOps Thesis — Đánh giá thực nghiệm các nền tảng MLOps

## Thông tin nhóm
- Đoàn Thanh Thảo – 23521466
- Dương Thanh Huyền – 23520659
- GVHD: ThS. Lê Anh Tuấn

## Môi trường thực nghiệm
- Platform: GCP e2-medium (2vCPU, 4GB RAM)
- OS: Ubuntu 22.04 LTS
- Python: 3.10
- MLflow: 2.19.0
- PyTorch: 2.11.0+cpu

## Cấu trúc thư mục
mlops-project/ ├── uc1_mnist/ │ ├── train_mnist.py # SimpleNN │ ├── train_mnist_cnn.py # CNN │ └── train_mnist_tc2.py # TC2 config sweep ├── uc2_phobert/ # TODO ├── results/ # CSV kết quả └── configs/ # File cấu hình môi trường

## Kết quả UC1 MNIST — MLflow
| Model | Accuracy TB | TC7 TB (giây) |
|---|---|---|
| SimpleNN | 0.9347 | 50.6 |
| CNN | 0.9721 | 281.1 |

## Cách tái tạo môi trường
Xem file configs/setup_mlflow_gcp.md
