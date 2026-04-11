import torch
import torch.nn as nn

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

# Lưu architecture (dùng lại không cần train lại)
torch.save(SimpleNN(), 'saved_models/uc1_mnist/simplenn_architecture.pth')
torch.save(CNN(),      'saved_models/uc1_mnist/cnn_architecture.pth')
print(" Đã lưu model architecture")
