import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("UC1_MNIST_MLflow")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data  = torchvision.datasets.MNIST('./data', train=False, transform=transform)

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

configs = [
    {"lr": 0.001, "batch_size": 32,  "epochs": 3, "run_name": "TC2_config1_lr0.001_batch32"},
    {"lr": 0.01,  "batch_size": 64,  "epochs": 3, "run_name": "TC2_config2_lr0.01_batch64"},
    {"lr": 0.05,  "batch_size": 128, "epochs": 3, "run_name": "TC2_config3_lr0.05_batch128"},
]

for cfg in configs:
    print(f"\n Run config: {cfg['run_name']}")
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg["batch_size"])

    with mlflow.start_run(run_name=cfg["run_name"]):
        start_time = time.time()
        model = SimpleNN()
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
        criterion = nn.CrossEntropyLoss()

        # Log of all params — TC2 
        mlflow.log_param("model", "SimpleNN")
        mlflow.log_param("lr", cfg["lr"])
        mlflow.log_param("batch_size", cfg["batch_size"])
        mlflow.log_param("epochs", cfg["epochs"])

        for epoch in range(cfg["epochs"]):
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
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in test_loader:
                correct += (model(X).argmax(1) == y).sum().item()
        accuracy = correct / len(test_data)

        pipeline_time = time.time() - start_time
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("pipeline_time_seconds", pipeline_time)

        print(f"   Accuracy: {accuracy:.4f} | TC7: {pipeline_time:.1f}s")
