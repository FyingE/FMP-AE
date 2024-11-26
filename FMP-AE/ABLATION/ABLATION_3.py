#移除AE的检测
import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model import create_windows, TimeSeriesCNN, plot_loss_values, \
    calculate_batch_similarity_matrix, calculate_optimized_matrix_profile, \
    read_data_with_anomalies, weights_init, plot_original_series_with_anomalies, \
    plot_detected_anomalies, expand_anomalies, plot_matrix_profile, plot_roc_curve, plot_pr_curve

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 超参数设置
window_size = 150
learning_rate = 0.0001
batch_size = 32
num_epochs = 50

# 读取示例文件，获取时间序列数据
file_path_example = 'UCR_Anomaly_FullData/020_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z2_5000_7175_7388.txt'
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
test_labels[(anomaly_start - train_val) : (anomaly_end - train_val) ] = 1  # 异常标记

test_labels = torch.tensor(test_labels, dtype=torch.float32)

# 初始化CNN模型
cnn_encoder = TimeSeriesCNN(window_size=window_size)
cnn_encoder.apply(weights_init)

# 定义优化器
optimizer = optim.Adam(cnn_encoder.parameters(), lr=learning_rate)

# 训练模型
train_dataset = TensorDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

loss_values = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch[0]
        features = cnn_encoder(batch)
        features_np = features.detach().numpy()
        sim_matrix = calculate_batch_similarity_matrix(features_np)
        matrix_profile = calculate_optimized_matrix_profile(sim_matrix, window_size=window_size)
        # Convert matrix_profile to a tensor with requires_grad=True
        matrix_profile_loss = torch.tensor(matrix_profile, dtype=torch.float32, requires_grad=True)
        loss = matrix_profile_loss.mean()  # Ensuring loss is a scalar for backward()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(cnn_encoder.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()
    loss_values.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


# 绘制损失变化
plot_loss_values(loss_values, "Epochs", "Loss")

# 使用训练好的CNN模型进行异常检测
cnn_encoder.eval()

with torch.no_grad():
    test_features = cnn_encoder(test_data)
    test_features_np = test_features.detach().numpy()
    test_sim_matrix = calculate_batch_similarity_matrix(test_features_np)
    test_matrix_profile = calculate_optimized_matrix_profile(test_sim_matrix, window_size=window_size)

# 基于Matrix Profile进行异常检测
threshold_matrix_profile = np.percentile(test_matrix_profile, 95)
predicted_labels_matrix_profile = (test_matrix_profile > threshold_matrix_profile).astype(int)

# 扩展预测和真实标签
expanded_true_labels = expand_anomalies(test_labels.numpy(), window_size=window_size)
expanded_predicted_labels = expand_anomalies(predicted_labels_matrix_profile, window_size=window_size)

# 计算性能指标
accuracy = accuracy_score(expanded_true_labels, expanded_predicted_labels)
precision = precision_score(expanded_true_labels, expanded_predicted_labels)
recall = recall_score(expanded_true_labels, expanded_predicted_labels)
f1 = f1_score(expanded_true_labels, expanded_predicted_labels)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")



# 绘制ROC曲线和PR曲线
plot_roc_curve(expanded_true_labels, test_matrix_profile)
plot_pr_curve(expanded_true_labels, test_matrix_profile)
