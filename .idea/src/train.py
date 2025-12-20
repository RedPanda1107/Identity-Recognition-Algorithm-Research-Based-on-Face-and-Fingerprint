import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. 造一批假数据：10000 样本，特征 20 类，目标 2 类
X = torch.randn(10000, 20)
y = torch.randint(0, 2, (10000, ))
train_loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

# 2. 极简 MLP
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3. 训练 5 个 epoch
for epoch in range(5):
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch}: loss={loss.item():.4f}")

print("GPU 显存占用:", torch.cuda.memory_allocated()/1024**2, "MB")