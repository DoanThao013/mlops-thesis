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
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=64)

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

with mlflow.start_run(run_name="CNN_UC1"):
    start_time = time.time()
    model = CNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    mlflow.log_param("model", "CNN")
    mlflow.log_param("lr", 0.01)
    mlflow.log_param("epochs", 3)
    mlflow.log_param("batch_size", 64)

    for epoch in range(3):
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
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            correct += (model(X).argmax(1) == y).sum().item()
    accuracy = correct / len(test_data)

    pipeline_time = time.time() - start_time
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("pipeline_time_seconds", pipeline_time)
    mlflow.pytorch.log_model(model, "model")

    print(f"\n  CNN Accuracy: {accuracy:.4f}")
    print(f" Pipeline time (TC7): {pipeline_time:.1f} giây")
