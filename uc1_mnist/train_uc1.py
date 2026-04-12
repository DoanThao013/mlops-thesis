"""
UC1 MNIST — MLflow Pipeline
Chạy: python3 train_uc1.py
Kết quả log tự động lên MLflow server
"""
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import argparse

# CONFIG

CONFIG = {
    "tracking_uri":  "http://localhost:5000",
    "experiment":    "UC1_MNIST_MLflow",
    "data_path":     "./data",
    "num_runs":      3,          # Số lần chạy lặp để lấy TB TC7

    # Các model cần chạy
    "models": ["SimpleNN", "CNN"],

    # TC2 — Config sweep
    "tc2_configs": [
        {"lr": 0.001, "batch_size": 32,  "epochs": 3},
        {"lr": 0.01,  "batch_size": 64,  "epochs": 3},
        {"lr": 0.05,  "batch_size": 128, "epochs": 3},
    ],

    # Hyperparameter mặc định
    "default": {
        "lr":         0.01,
        "batch_size": 64,
        "epochs":     3,
    }
}


# MODELS

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(64*5*5, 128)
        self.fc2   = nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

MODELS = {"SimpleNN": SimpleNN, "CNN": CNN}

# FUNCTIONS

def get_data(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = torchvision.datasets.MNIST(
        CONFIG["data_path"], train=True,
        download=True, transform=transform)
    test = torchvision.datasets.MNIST(
        CONFIG["data_path"], train=False, transform=transform)
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test,  batch_size=batch_size),
        len(test)
    )

def train_one_run(model_name, lr, batch_size, epochs, run_name):
    train_loader, test_loader, test_size = get_data(batch_size)
    model     = MODELS[model_name]()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name=run_name):
        start_time = time.time()

        # Log params
        mlflow.log_param("model",      model_name)
        mlflow.log_param("lr",         lr)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs",     epochs)
        mlflow.log_param("platform",   "MLflow-GCP")

        # Train
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        # Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in test_loader:
                correct += (model(X).argmax(1) == y).sum().item()
        accuracy = correct / test_size

        # Log metrics
        pipeline_time = time.time() - start_time
        mlflow.log_metric("test_accuracy",          accuracy)
        mlflow.log_metric("pipeline_time_seconds",  pipeline_time)
        mlflow.pytorch.log_model(model, "model")

        print(f"     Accuracy={accuracy:.4f} | TC7={pipeline_time:.1f}s")
        return accuracy, pipeline_time

# MAIN

def main(mode="all"):
    mlflow.set_tracking_uri(CONFIG["tracking_uri"])
    mlflow.set_experiment(CONFIG["experiment"])
    d = CONFIG["default"]

    if mode in ("all", "repeat"):
        # Chạy lặp num_runs lần để lấy TB TC7
        print("\n PHẦN 1: Chạy lặp để lấy TB TC7")
        for model_name in CONFIG["models"]:
            for i in range(1, CONFIG["num_runs"] + 1):
                print(f"\n  [{model_name}] Lần {i}/{CONFIG['num_runs']}")
                train_one_run(
                    model_name = model_name,
                    lr         = d["lr"],
                    batch_size = d["batch_size"],
                    epochs     = d["epochs"],
                    run_name   = f"{model_name}_run{i}"
                )

    if mode in ("all", "tc2"):
        # TC2 — Config sweep
        print("\n PHẦN 2: TC2 Config Sweep")
        for cfg in CONFIG["tc2_configs"]:
            run_name = f"TC2_lr{cfg['lr']}_batch{cfg['batch_size']}"
            print(f"\n  Config: {run_name}")
            train_one_run(
                model_name = "SimpleNN",
                run_name   = run_name,
                **cfg
            )

    print("\n Hoàn thành! Xem kết quả tại:", CONFIG["tracking_uri"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "repeat", "tc2"],
        default="all",
        help="all=chạy tất cả | repeat=chỉ lặp TC7 | tc2=chỉ config sweep"
    )
    args = parser.parse_args()
    main(args.mode)
