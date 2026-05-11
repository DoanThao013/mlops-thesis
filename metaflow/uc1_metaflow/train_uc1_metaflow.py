"""
UC1 MNIST — Metaflow Pipeline (Local Mode trên GCP VM)

Chạy:
  python3 train_uc1_metaflow.py run
  python3 train_uc1_metaflow.py run --model CNN
  python3 train_uc1_metaflow.py run --lr 0.05 --batch_size 128
  python3 train_uc1_metaflow.py run --model SimpleNN --lr 0.001 --batch_size 32
"""
import os
os.environ["METAFLOW_DEFAULT_DATASTORE"] = "local"
os.environ["METAFLOW_DEFAULT_METADATA"] = "local"

from metaflow import FlowSpec, step, Parameter
import time


class MNISTFlow(FlowSpec):
    """Pipeline UC1 MNIST — Metaflow"""

    model_name = Parameter('model', default='SimpleNN',
                           help='Model: SimpleNN or CNN')
    lr = Parameter('lr', default=0.01, type=float,
                   help='Learning rate')
    batch_size = Parameter('batch_size', default=64, type=int,
                           help='Batch size')
    epochs = Parameter('epochs', default=3, type=int,
                       help='Number of epochs')

    @step
    def start(self):
        self.start_time = time.time()
        print(f"\n{'='*50}")
        print(f"  UC1 MNIST — Metaflow (GCP VM)")
        print(f"  Model: {self.model_name}")
        print(f"  Config: lr={self.lr}, batch={self.batch_size}, epochs={self.epochs}")
        print(f"{'='*50}")
        self.next(self.train_and_evaluate)

    @step
    def train_and_evaluate(self):
        """Load data + Train + Evaluate (gộp 1 step tránh pickle error)"""
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.transforms as transforms

        # --- Load Data ---
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(
            './data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            './data', train=False, transform=transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size)

        print(f"  Data: {len(train_dataset)} train, {len(test_dataset)} test")

        # --- Define Models ---
        class SimpleNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 128)
                self.fc2 = nn.Linear(128, 10)
            def forward(self, x):
                x = x.view(-1, 784)
                return self.fc2(torch.relu(self.fc1(x)))

        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.pool = nn.MaxPool2d(2)
                self.fc1 = nn.Linear(64*5*5, 128)
                self.fc2 = nn.Linear(128, 10)
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(-1, 64*5*5)
                return self.fc2(torch.relu(self.fc1(x)))

        models = {"SimpleNN": SimpleNN, "CNN": CNN}
        model = models[self.model_name]()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        # --- Train ---
        print(f"  Training {self.model_name}...")
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.4f}")

        # --- Evaluate ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in test_loader:
                correct += (model(X).argmax(1) == y).sum().item()

        self.accuracy = correct / len(test_dataset)
        self.pipeline_time = time.time() - self.start_time

        print(f"\n  ✅ Accuracy: {self.accuracy:.4f}")
        print(f"  ✅ TC7 (pipeline time): {self.pipeline_time:.1f}s")
        self.next(self.end)

    @step
    def end(self):
        print(f"\n{'='*50}")
        print(f"  FINAL RESULT")
        print(f"  Model: {self.model_name}")
        print(f"  Accuracy: {self.accuracy:.4f}")
        print(f"  TC7: {self.pipeline_time:.1f}s")
        print(f"  Config: lr={self.lr}, batch={self.batch_size}, epochs={self.epochs}")
        print(f"  Platform: Metaflow Local Mode (GCP VM)")
        print(f"{'='*50}")


if __name__ == '__main__':
    MNISTFlow()
