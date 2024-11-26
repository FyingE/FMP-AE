import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
import random
from model import create_windows, plot_loss_values, \
    calculate_batch_similarity_matrix, \
    calculate_optimized_matrix_profile, read_data_with_anomalies, \
    weights_init, plot_original_series_with_anomalies, plot_detected_anomalies, expand_anomalies, plot_matrix_profile, \
    plot_reconstruct_error

# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def normalize(matrix):
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())


def smooth(matrix, smoothing_factor=0.9):
    smoothed = np.zeros_like(matrix)
    smoothed[0] = matrix[0]
    for i in range(1, len(matrix)):
        smoothed[i] = smoothing_factor * smoothed[i - 1] + (1 - smoothing_factor) * matrix[i]
    return smoothed


class TimeSeriesCNN(nn.Module):
    def __init__(self, window_size):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        fc_input_dim = self._get_fc_input_dim(window_size)
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)

        # 二分类层
        self.fc3 = nn.Linear(64, 1)

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
        features = x
        x = self.fc3(x)
        return features, x


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


def combined_loss_function(reconstructed, original, matrix_profile, predicted, true, lambda_m=0, lambda_c=1):
    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    matrix_profile_loss = torch.mean(matrix_profile)
    classification_loss = nn.BCEWithLogitsLoss()(predicted, true)
    return reconstruction_loss + lambda_m * matrix_profile_loss + lambda_c * classification_loss


# 超参数设置
param_grid = {
    'window_size': [100, 200, 300],
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [32, 64, 128],
    'num_epochs': [50]
}

grid = list(ParameterGrid(param_grid))


def train_and_evaluate(params):
    window_size = params['window_size']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']

    # 读取示例文件，获取时间序列数据
    file_path_example = 'UCR_Anomaly_FullData/004_UCR_Anomaly_DISTORTEDBIDMC1_2500_5400_5600.txt'
    data, train_val, anomaly_start, anomaly_end = read_data_with_anomalies(file_path_example)
    data = np.array(data)
    data = (data - data.min()) / (data.max() - data.min())

    length = len(data)
    true_labels = np.zeros(length)
    true_labels[anomaly_start:anomaly_end] = 1

    train_data = data[:train_val]
    train_data_windows = create_windows(train_data, window_size=window_size)
    train_data = torch.tensor(train_data_windows, dtype=torch.float32).unsqueeze(1)

    test_data = data[train_val:]
    test_data_windows = create_windows(test_data, window_size=window_size)
    test_data = torch.tensor(test_data_windows, dtype=torch.float32).unsqueeze(1)
    test_labels = np.zeros(len(test_data_windows))
    test_labels[anomaly_start - train_val - window_size + 1:anomaly_end - train_val - window_size + 1] = 1
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    # 初始化模型
    cnn_encoder = TimeSeriesCNN(window_size=window_size)
    autoencoder = Autoencoder(input_dim=window_size)
    cnn_encoder.apply(weights_init)
    autoencoder.apply(weights_init)
    optimizer = optim.Adam(list(cnn_encoder.parameters()) + list(autoencoder.parameters()), lr=learning_rate)

    # 训练模型
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        lambda_m = 0.001 * (epoch / num_epochs)
        lambda_c = 1
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch[0]
            features, predicted = cnn_encoder(batch)
            reconstructed = autoencoder(batch)
            features_np = features.detach().numpy()
            sim_matrix = calculate_batch_similarity_matrix(features_np)
            matrix_profile = calculate_optimized_matrix_profile(sim_matrix, window_size=window_size)
            matrix_profile = normalize(matrix_profile)
            matrix_profile = smooth(matrix_profile)
            true_labels_batch = torch.zeros(predicted.size(0), 1, dtype=torch.float32)  # 调整形状
            loss = combined_loss_function(reconstructed, batch,
                                          torch.tensor(matrix_profile, dtype=torch.float32).clone().detach(), predicted,
                                          true_labels_batch, lambda_m=lambda_m, lambda_c=lambda_c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

    # 使用训练好的模型进行异常检测
    cnn_encoder.eval()
    autoencoder.eval()
    with torch.no_grad():
        test_features, predicted_test = cnn_encoder(test_data)
        reconstructed_test_data = autoencoder(test_data)
        recon_error = torch.mean((reconstructed_test_data - test_data) ** 2, dim=(1, 2)).detach().numpy()
        test_features_np = test_features.detach().numpy()
        test_sim_matrix = calculate_batch_similarity_matrix(test_features_np)
        test_matrix_profile = calculate_optimized_matrix_profile(test_sim_matrix, window_size=window_size)
        test_matrix_profile = normalize(test_matrix_profile)
        test_matrix_profile = smooth(test_matrix_profile)

    threshold_recon_error = np.percentile(recon_error, 95)  # 调整阈值百分位数
    threshold_matrix_profile = np.percentile(test_matrix_profile, 95)  # 调整阈值百分位数
    predicted_labels_recon_error = (recon_error > threshold_recon_error).astype(int)
    predicted_labels_matrix_profile = (test_matrix_profile > threshold_matrix_profile).astype(int)
    predicted_labels = predicted_labels_recon_error & predicted_labels_matrix_profile
    expanded_true_labels = expand_anomalies(test_labels.numpy(), window_size=window_size)
    expanded_predicted_labels = expand_anomalies(predicted_labels, window_size=window_size)

    accuracy = accuracy_score(expanded_true_labels, expanded_predicted_labels)
    precision = precision_score(expanded_true_labels, expanded_predicted_labels, zero_division=1)
    recall = recall_score(expanded_true_labels, expanded_predicted_labels, zero_division=1)
    f1 = f1_score(expanded_true_labels, expanded_predicted_labels)

    return accuracy, precision, recall, f1


# 执行网格搜索
best_params = None
best_score = 0
results = []

for params in grid:
    accuracy, precision, recall, f1 = train_and_evaluate(params)
    score = f1  # 或者根据你的具体需求选择其他指标
    results.append((params, accuracy, precision, recall, f1))
    if score > best_score:
        best_score = score
        best_params = params

print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")

# 打印所有结果
for result in results:
    print(f"Params: {result[0]}, Accuracy: {result[1]}, Precision: {result[2]}, Recall: {result[3]}, F1: {result[4]}")