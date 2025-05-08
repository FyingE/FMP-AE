import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from processing import (
    create_windows_multivariate,
    combined_loss_function,
    calculate_batch_similarity_matrix,
    calculate_optimized_matrix_profile,
)
from mymodel import FMP_TAE_Model

# === 训练数据加载 ===

data = torch.randn(1000, 5)  # 示例：1000 个时间步，每步 5 维
labels = torch.zeros(1000)  # 示例：全为正常点

# === 参数设置 ===
window_size = 64
step_size = 8
batch_size = 32
epochs = 500
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 数据窗口化处理 ===
windows, _ = create_windows_multivariate(
    data.numpy(), labels.numpy(), window_size, step_size
)
windows = torch.tensor(windows, dtype=torch.float32).to(device)
dataset = TensorDataset(windows, windows)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === 模型初始化 ===
model = FMP_TAE_Model(input_dim=5, window_size=window_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === 训练过程 ===
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch_data, _ in data_loader:
        optimizer.zero_grad()

        # CNN特征提取 & TAE重构
        cnn_features, segment_features, reconstruction = model(batch_data)

        # 相似度与MP计算
        similarity_matrix = calculate_batch_similarity_matrix(cnn_features)
        matrix_profile = calculate_optimized_matrix_profile(
            similarity_matrix, window_size=5
        )

        # 损失函数
        loss = combined_loss_function(
            reconstructed=reconstruction,
            original=batch_data,
            matrix_profile=matrix_profile,
            lambda_m=0.3,
            segment_length=10,
            lambda_internal_mp=0.1,
        )

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_loader):.4f}")

# === 模型保存 ===
torch.save(model.state_dict(), "fmp_tae_model.pth")
