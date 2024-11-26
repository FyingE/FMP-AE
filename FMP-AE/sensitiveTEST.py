import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 数据准备函数
def read_data_with_anomalies(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    data = [float(line.strip()) for line in data if line.strip()]
    file_name = os.path.basename(file_path)
    parts = re.split('_|\.', file_name)
    train_val = int(parts[4])
    anomaly_start = int(parts[5])
    anomaly_end = int(parts[6])
    return data, train_val, anomaly_start, anomaly_end


def create_windows(time_series, window_size):
    if len(time_series) < window_size:
        raise ValueError("Time series length must be greater than or equal to the window size.")
    windows = []
    for i in range(0, len(time_series) - window_size + 1):
        windows.append(time_series[i:i + window_size])
    return np.array(windows)


def generate_anomaly_versions(data, num_versions, anomaly_density):
    versions = []
    length = len(data)
    for _ in range(num_versions):
        num_anomalies = int(length * anomaly_density)
        anomalies = np.zeros(length)
        if num_anomalies > 0:
            anomaly_indices = np.random.choice(length, num_anomalies, replace=False)
            anomalies[anomaly_indices] = 1
        version_data = data.copy()
        # 修正索引问题
        anomaly_indices = np.where(anomalies == 1)[0]
        version_data[anomaly_indices] += np.random.uniform(-0.5, 0.5, size=num_anomalies)
        versions.append((version_data, anomalies))
    return versions



# 定义联合损失函数
def combined_loss_function(reconstructed, original, matrix_profile, lambda_m=0.3):
    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    matrix_profile_loss = torch.mean(matrix_profile)
    reconstruction_loss = reconstruction_loss / torch.mean(reconstructed)
    matrix_profile_loss = matrix_profile_loss / torch.mean(matrix_profile)
    return reconstruction_loss + lambda_m * matrix_profile_loss


class TimeSeriesCNN(nn.Module):
    def __init__(self, window_size):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        fc_input_dim = self._get_fc_input_dim(window_size)
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)

    def _get_fc_input_dim(self, window_size):
        x = torch.zeros(1, 1, window_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        return x.numel()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, -1)
        return x


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.kaiming_normal_(m.weight)


def calculate_batch_similarity_matrix(features, batch_size=1000):
    num_features = features.shape[0]
    sim_matrix = np.zeros((num_features, num_features))
    for start_idx in range(0, num_features, batch_size):
        stop_idx = min(start_idx + batch_size, num_features)
        batch = features[start_idx:stop_idx]
        sim_matrix[start_idx:stop_idx] = np.linalg.norm(batch[:, np.newaxis] - features, axis=2)
    return sim_matrix


def calculate_optimized_matrix_profile(similarity_matrix, window_size):
    n = similarity_matrix.shape[0]
    matrix_profile = np.full(n, np.inf)
    for i in range(n):
        matrix_profile[i] = np.min(similarity_matrix[i, :max(1, i - window_size + 1)])
    return matrix_profile


def evaluate_on_anomaly_versions(versions, model_cnn, model_ae, criterion, lambda_m, device):
    model_cnn.to(device)
    model_ae.to(device)
    model_cnn.eval()
    model_ae.eval()

    loss_values = []

    with torch.no_grad():
        for data, true_labels in versions:
            test_data = torch.tensor(data, dtype=torch.float32).unsqueeze(1).to(device)
            test_features = model_cnn(test_data)
            reconstructed_test_data = model_ae(test_data)

            recon_error = torch.mean((reconstructed_test_data - test_data) ** 2, dim=(1, 2))
            test_features_np = test_features.cpu().numpy()
            test_sim_matrix = calculate_batch_similarity_matrix(test_features_np)
            test_matrix_profile = calculate_optimized_matrix_profile(test_sim_matrix, window_size=window_size)

            # 计算损失
            test_labels = torch.tensor(true_labels, dtype=torch.float32).unsqueeze(1).to(device)
            loss = combined_loss_function(reconstructed_test_data, test_data,
                                          torch.tensor(test_matrix_profile, dtype=torch.float32).clone().detach(),
                                          lambda_m=lambda_m)

            loss_values.append(loss.item())

    return loss_values


def compute_gradients(data, model_cnn, model_ae, criterion, lambda_m, device):
    model_cnn.to(device)
    model_ae.to(device)
    model_cnn.train()
    model_ae.train()

    data = torch.tensor(data, dtype=torch.float32).unsqueeze(1).to(device)
    optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=learning_rate)
    optimizer_ae = optim.Adam(model_ae.parameters(), lr=learning_rate)

    optimizer_cnn.zero_grad()
    optimizer_ae.zero_grad()

    test_features = model_cnn(data)
    reconstructed_test_data = model_ae(data)

    recon_error = torch.mean((reconstructed_test_data - data) ** 2, dim=(1, 2))
    test_features_np = test_features.cpu().numpy()
    test_sim_matrix = calculate_batch_similarity_matrix(test_features_np)
    test_matrix_profile = calculate_optimized_matrix_profile(test_sim_matrix, window_size=window_size)

    # 计算损失
    loss = combined_loss_function(reconstructed_test_data, data,
                                  torch.tensor(test_matrix_profile, dtype=torch.float32).clone().detach(),
                                  lambda_m=lambda_m)

    # 计算梯度
    loss.backward()

    # 提取梯度
    cnn_gradients = [param.grad.clone().cpu().numpy() for param in model_cnn.parameters()]
    ae_gradients = [param.grad.clone().cpu().numpy() for param in model_ae.parameters()]

    return cnn_gradients, ae_gradients


def sensitivity_test(data, model_cnn, model_ae, criterion, lambda_m, device, densities):
    loss_by_density = []
    for density in densities:
        versions = generate_anomaly_versions(data, num_versions=1, anomaly_density=density)
        loss_values = evaluate_on_anomaly_versions(versions, model_cnn, model_ae, criterion, lambda_m, device)
        loss_by_density.append(np.mean(loss_values))
    return loss_by_density


def plot_losses_and_gradients(loss_values, gradient_values, densities):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(densities, loss_values, marker='o')
    plt.title('Loss vs Anomaly Density')
    plt.xlabel('Anomaly Density')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(densities, [np.mean(np.abs(g)) for g in gradient_values], marker='o')
    plt.title('Gradient Magnitude vs Anomaly Density')
    plt.xlabel('Anomaly Density')
    plt.ylabel('Gradient Magnitude')

    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 配置
    window_size = 50
    learning_rate = 0.001
    lambda_m = 0.3
    device = 'cpu'

    # 读取数据
    file_path = 'UCR_Anomaly_FullData/008_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature4_4000_5549_5597.txt'  # 替换为你的数据文件路径
    data, train_val, anomaly_start, anomaly_end = read_data_with_anomalies(file_path)

    # 创建训练和测试数据
    time_series_windows = create_windows(data, window_size)
    model_cnn = TimeSeriesCNN(window_size)
    model_ae = Autoencoder(window_size)

    # 初始化模型权重
    model_cnn.apply(weights_init)
    model_ae.apply(weights_init)

    # 训练模型（这里仅作为示例，实际需要你自己的训练代码）
    # 例如：
    # optimizer_cnn = optim.Adam(model_cnn.parameters(), lr=learning_rate)
    # optimizer_ae = optim.Adam(model_ae.parameters(), lr=learning_rate)
    # ... 进行训练

    # 测试模型
    densities = [0.01, 0.05, 0.1]  # 异常点密度示例
    loss_values = sensitivity_test(data, model_cnn, model_ae, combined_loss_function, lambda_m, device, densities)

    # 可视化结果
    plot_losses_and_gradients(loss_values, [], densities)  # 这里只绘制了损失函数图，没有计算梯度图
