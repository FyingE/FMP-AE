import random
import numpy as np
import torch
import torch.optim as optim
import stumpy  # Matrix Profile计算库
import matplotlib.pyplot as plt  # 可视化库
from torch.utils.data import DataLoader, TensorDataset
from model import create_windows, TimeSeriesCNN, Autoencoder, plot_loss_values, \
    calculate_batch_similarity_matrix, calculate_optimized_matrix_profile, \
    read_data_with_anomalies, weights_init, expand_anomalies, \
    combined_loss_function

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 超参数设置
window_size = 100
learning_rate = 0.001
batch_size = 64
num_epochs = 50

# 读取示例文件，获取时间序列数据
file_path_example = 'UCR_Anomaly_FullData/116_UCR_Anomaly_CIMIS44AirTemperature4_4000_5549_5597.txt'
data, train_val, anomaly_start, anomaly_end = read_data_with_anomalies(file_path_example)
# 标准化数据到 0-1 范围
data = np.array(data)
data = (data - data.min()) / (data.max() - data.min())
length = len(data)

# 根据异常起始和结束点生成标签数组
true_labels = np.zeros(length)
true_labels[anomaly_start:anomaly_end] = 1

# 确保训练数据中没有异常点
train_data = data[:train_val]
train_data_windows = create_windows(train_data, window_size=window_size)
train_data = torch.tensor(train_data_windows, dtype=torch.float32).unsqueeze(1)

test_data = data[train_val:]
test_data_windows = create_windows(test_data, window_size=window_size)
test_data = torch.tensor(test_data_windows, dtype=torch.float32).unsqueeze(1)
test_labels = np.zeros(len(test_data_windows))
test_labels[(anomaly_start - train_val):(anomaly_end - train_val)] = 1  # 异常标记

test_labels = torch.tensor(test_labels, dtype=torch.float32)

# 初始化模型
cnn_encoder = TimeSeriesCNN(window_size=window_size)
autoencoder = Autoencoder(input_dim=window_size)

cnn_encoder.apply(weights_init)
autoencoder.apply(weights_init)

# 定义优化器
optimizer = optim.Adam(list(cnn_encoder.parameters()) + list(autoencoder.parameters()), lr=learning_rate)

# 训练模型
train_dataset = TensorDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

loss_values = []

for epoch in range(num_epochs):
    lambda_m = 0.01 * (epoch / num_epochs)
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch[0]
        features = cnn_encoder(batch)
        reconstructed = autoencoder(batch)
        features_np = features.detach().numpy()
        sim_matrix = calculate_batch_similarity_matrix(features_np)
        matrix_profile = calculate_optimized_matrix_profile(sim_matrix, window_size=window_size)
        loss = combined_loss_function(reconstructed, batch,
                                      torch.tensor(matrix_profile, dtype=torch.float32).clone().detach(),
                                      lambda_m=lambda_m)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(cnn_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()
    loss_values.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# 使用训练好的模型进行异常检测
cnn_encoder.eval()
autoencoder.eval()

with torch.no_grad():
    test_features = cnn_encoder(test_data)  # 提取测试数据的特征表示（feature map）
    reconstructed_test_data = autoencoder(test_data)
    recon_error = torch.mean((reconstructed_test_data - test_data) ** 2, dim=(1, 2)).detach().numpy()

# 计算相似度矩阵和优化后的Matrix Profile
test_features_np = test_features.detach().numpy()
test_sim_matrix = calculate_batch_similarity_matrix(test_features_np)
test_matrix_profile = calculate_optimized_matrix_profile(test_sim_matrix, window_size=window_size)

# 使用stumpy计算Matrix Profile
stumpy_matrix_profile = stumpy.stump(data, m=window_size)[:, 0]  # 取出Matrix Profile的第一列

# 绘制三个子图
fig, axs = plt.subplots(3, figsize=(14, 12), sharex=True)

# 绘制原始时间序列
axs[0].plot(data, label='Original Time Series', color='black')
axs[0].fill_between(range(length), 0, 1, where=true_labels==1, color='red', alpha=0.5, label='True Anomalies')
axs[0].set_title('Original Time Series with True Anomalies')
axs[0].legend(loc='upper left')  # 设置图例在左上角

# 绘制stumpy计算的Matrix Profile
axs[1].plot(stumpy_matrix_profile, label='Stumpy Matrix Profile', color='blue')
axs[1].set_title('Stumpy Matrix Profile')
axs[1].legend(loc='upper left')

# 绘制优化模型计算的Matrix Profile（从第200个位置开始）
start_index = 300
axs[2].plot(range(start_index, len(test_matrix_profile)), test_matrix_profile[start_index:], label='Optimized Matrix Profile', color='orange')
axs[2].set_title('Optimized Matrix Profile (From 1D-CNN)')
axs[2].legend(loc='upper left')

plt.xlabel('Index')
plt.tight_layout()
plt.show()
