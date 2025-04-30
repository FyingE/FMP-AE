import numpy as np
import os
import re
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve
# 定义联合损失函数

def combined_loss_function(reconstructed, original, matrix_profile, lambda_m=0.3):
    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    matrix_profile_loss = torch.mean(matrix_profile)
    # 标准化损失
    reconstruction_loss = reconstruction_loss / torch.mean(reconstructed)
    matrix_profile_loss = matrix_profile_loss / torch.mean(matrix_profile)
    return reconstruction_loss + lambda_m * matrix_profile_loss


def inter_window_mp_loss(features):
    """
    features: shape (N, D) - N个窗口的特征表示
    返回：平均的 inter-window MP loss
    """
    # 计算所有样本间的 pairwise cosine 相似度
    sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
    
    # 避免与自身比对，将对角线元素设为最大值（1），以便后续取最小
    sim_matrix.fill_diagonal_(1.0)
    
    # 对每个样本，取与其他样本的最小相似度（即最不像的样本）
    min_sim = torch.min(sim_matrix, dim=1)[0]

    # 损失越小越好，因此希望最小相似度越低
    return torch.mean(min_sim)

def intra_window_mp_loss(intra_features):
    """
    intra_features: shape (N, m, D) - N个窗口，每个窗口有m个子段特征
    返回：平均的 intra-window MP loss（即平均的1 - cosine相似度）
    """
    N, m, D = intra_features.shape
    loss_sum = 0.0
    count = 0

    for i in range(N):
        # 每个窗口内的 m 个子段特征
        segments = intra_features[i]  # shape (m, D)
        for p in range(m):
            for q in range(p + 1, m):
                # 计算 dissimilarity: 1 - cosine similarity
                sim = F.cosine_similarity(segments[p], segments[q], dim=0)
                dissim = 1 - sim
                loss_sum += dissim
                count += 1

    return loss_sum / (count + 1e-8)

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

    def forward(self, x, visualize=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        if visualize:
            self.visualize_feature_map(x, 'conv1')
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        if visualize:
            self.visualize_feature_map(x, 'conv2')
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        if visualize:
            self.visualize_feature_map(x, 'conv3')
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def visualize_feature_map(self, feature_map, layer_name):
        # 将特征图从 GPU 移动到 CPU 并转化为 numpy 数组
        feature_map = feature_map.detach().cpu().numpy()

        # 绘制特征图
        num_channels = feature_map.shape[1]
        plt.figure(figsize=(20, num_channels))
        for i in range(num_channels):
            plt.subplot(1, num_channels, i + 1)
            plt.imshow(feature_map[0, i, :], aspect='auto', cmap='viridis')
            plt.axis('off')
            plt.title(f'{layer_name} - Filter {i + 1}')
        plt.show()


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.kaiming_normal_(m.weight)

#x向autoencoder传人window_size参数
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
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, -1)  # 恢复原始形状
        return x




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

def create_windows(data, window_size, padding="reflect"):
    """
    根据给定的窗口大小创建滑动窗口。
    增加对时间序列的填充来减少边界效应。

    Args:
    - data (np.array): 输入的时间序列数据。
    - window_size (int): 窗口大小。
    - padding (str): 填充方式。默认使用"reflect"，也可以选择"zero"填充。

    Returns:
    - windows (np.array): 创建的窗口。
    """
    if padding == "reflect":
        # 反射填充，反射窗口大小的一半
        pad_size = window_size // 2
        padded_data = np.pad(data, (pad_size, pad_size), mode="reflect")
    elif padding == "zero":
        # Zero 填充
        pad_size = window_size // 2
        padded_data = np.pad(data, (pad_size, pad_size), mode="constant", constant_values=0)
    else:
        raise ValueError(f"Unknown padding type: {padding}")

    windows = []
    for i in range(len(data)):
        windows.append(padded_data[i:i + window_size])
    return np.array(windows)

#可视化

def plot_loss_values(loss_values, xlabel, ylabel):
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values)
    plt.title("Loss Change with Epochs")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_original_series_with_anomalies(data, true_labels, title="Original Time Series with Anomalies"):
    plt.figure(figsize=(12, 6))
    plt.plot(data, color='steelblue',label='Time Series')
    anomaly_indices = np.where(true_labels == 1)[0]
    plt.scatter(anomaly_indices, np.array(data)[anomaly_indices], color='red', label='True Anomalies', marker='x')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def plot_detected_anomalies(data, predicted_labels, title="Detected Anomalies"):
    plt.figure(figsize=(12, 6))
    plt.plot(data, color='teal',label='Time Series')
    detected_anomaly_indices = np.where(predicted_labels == 1)[0]
    plt.scatter(detected_anomaly_indices, np.array(data)[detected_anomaly_indices], color='orange', label='Detected Anomalies', marker='x')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def plot_matrix_profile(matrix_profile, title="Matrix Profile"):
    plt.figure(figsize=(12, 6))
    plt.plot(matrix_profile, label='Matrix Profile',color='blue')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Matrix Profile Value")
    plt.legend()
    plt.show()

def plot_reconstruct_error(recon_error, title="reconstruct_error"):
    plt.figure(figsize=(12, 6))
    plt.plot(recon_error, label='Reconstruct Error',color='red')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Reconstruct Error")
    plt.legend()
    plt.show()

# 使用从features数组（即时间序列窗口的特征向量）计算得到的 相似度矩阵 来计算优化后的 Matrix Profile
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

# 设置容错区域
def expand_anomalies(labels, window_size):
    expanded_labels = np.zeros_like(labels)
    half_window = window_size // 2
    for i, label in enumerate(labels):
        if label == 1:
            start = max(0, i - half_window)
            end = min(len(labels), i + half_window + 1)
            expanded_labels[start:end] = 1
    return expanded_labels

def calculate_metrics(conf_mat):
    tp = conf_mat[0, 0]
    fp = conf_mat[0, 1]
    fn = conf_mat[1, 0]
    tn = conf_mat[1, 1]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score}

class TimeSeriesMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(TimeSeriesMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)

    def forward(self, x):
        x = x.squeeze(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def plot_pr_curve(true_labels, predicted_scores):
    precision, recall, _ = precision_recall_curve(true_labels, predicted_scores)
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.show()


