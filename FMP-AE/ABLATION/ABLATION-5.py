#1D-cnn 变为MLP
import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import create_windows, Autoencoder, plot_loss_values, \
    calculate_batch_similarity_matrix, calculate_optimized_matrix_profile, read_data_with_anomalies, \
    weights_init, expand_anomalies, combined_loss_function, plot_original_series_with_anomalies, \
    plot_detected_anomalies, plot_matrix_profile, plot_reconstruct_error, plot_roc_curve, plot_pr_curve,TimeSeriesMLP

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
file_path_example = 'UCR_Anomaly_FullData/020_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z2_5000_7175_7388.txt'
data, train_val, anomaly_start, anomaly_end = read_data_with_anomalies(file_path_example)
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
test_labels[(anomaly_start - train_val): (anomaly_end - train_val)] = 1
test_labels = torch.tensor(test_labels, dtype=torch.float32)


encoder = TimeSeriesMLP(input_dim=window_size)

autoencoder = Autoencoder(input_dim=window_size)
encoder.apply(weights_init)
autoencoder.apply(weights_init)

# 定义优化器
optimizer = optim.Adam(list(encoder.parameters()) + list(autoencoder.parameters()), lr=learning_rate)

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
        features = encoder(batch)
        reconstructed = autoencoder(batch)
        features_np = features.detach().numpy()
        sim_matrix = calculate_batch_similarity_matrix(features_np)
        matrix_profile = calculate_optimized_matrix_profile(sim_matrix, window_size=window_size)
        loss = combined_loss_function(reconstructed, batch,
                                      torch.tensor(matrix_profile, dtype=torch.float32).clone().detach(),
                                      lambda_m=lambda_m)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()
    loss_values.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

plot_loss_values(loss_values, "Epochs", "Loss")

# 使用训练好的模型进行异常检测
encoder.eval()
autoencoder.eval()

with torch.no_grad():
    test_features = encoder(test_data)
    reconstructed_test_data = autoencoder(test_data)
    recon_error = torch.mean((reconstructed_test_data - test_data) ** 2, dim=(1, 2)).detach().numpy()

test_features_np = test_features.detach().numpy()
test_sim_matrix = calculate_batch_similarity_matrix(test_features_np)
test_matrix_profile = calculate_optimized_matrix_profile(test_sim_matrix, window_size=window_size)

threshold_recon_error = np.percentile(recon_error, 95)
threshold_matrix_profile = np.percentile(test_matrix_profile, 95)

predicted_labels_recon_error = (recon_error > threshold_recon_error).astype(int)
predicted_labels_matrix_profile = (test_matrix_profile > threshold_matrix_profile).astype(int)
predicted_labels = predicted_labels_recon_error & predicted_labels_matrix_profile

expanded_true_labels = expand_anomalies(test_labels.numpy(), window_size=window_size)
expanded_predicted_labels = expand_anomalies(predicted_labels, window_size=window_size)

accuracy = accuracy_score(expanded_true_labels, expanded_predicted_labels)
precision = precision_score(expanded_true_labels, expanded_predicted_labels)
recall = recall_score(expanded_true_labels, expanded_predicted_labels)
f1 = f1_score(expanded_true_labels, expanded_predicted_labels)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

plot_original_series_with_anomalies(data, true_labels, title="Original Time Series with True Anomalies")
plot_detected_anomalies(data[train_val:], expanded_predicted_labels, title="Detected Anomalies in Test Data")
plot_matrix_profile(test_matrix_profile, title="Matrix Profile for Test Data")
plot_reconstruct_error(recon_error, title="Reconstruct Error for Test Data")
plot_roc_curve(expanded_true_labels, recon_error)
plot_pr_curve(expanded_true_labels, recon_error)
