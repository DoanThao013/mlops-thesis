# Hướng dẫn tái tạo môi trường MLflow trên GCP

## 1. Tạo VM
- Machine type: e2-medium (2vCPU, 4GB RAM)
- OS: Ubuntu 22.04 LTS
- Disk: 20GB Standard
- Firewall: HTTP, HTTPS, TCP 5000

## 2. Cài đặt
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv sqlite3 git
pip3 install mlflow==2.19.0
pip3 install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install scikit-learn pandas numpy
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
source ~/.bashrc
```

## 3. Cấu hình MLflow service
```bash
mkdir -p ~/mlflow-data/artifacts
# Tạo file /etc/systemd/system/mlflow.service
# (xem nội dung trong configs/mlflow.service)
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
```

## 4. Lỗi đã gặp và cách fix
- MLflow 3.x có --allowed-hosts nhưng 2.x không có → dùng 2.19.0
- PATH không tự thêm → thêm vào .bashrc
- UI filter mặc định sai → xóa filter thủ công
- e2-micro không đủ RAM → nâng lên e2-medium
