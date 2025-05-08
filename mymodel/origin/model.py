import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# === 联合损失函数 ===
def combined_loss_function(reconstructed, original, matrix_profile, lambda_m=0.3):
    # 重构损失
    reconstruction_loss = nn.MSELoss()(reconstructed, original)  # original 是原始的 windows
    matrix_profile_loss = torch.mean(matrix_profile)  # 矩阵谱损失（如果有）
    
    # 标准化损失
    reconstruction_loss = reconstruction_loss / torch.mean(reconstructed)
    matrix_profile_loss = matrix_profile_loss / torch.mean(matrix_profile)
    
    # 返回联合损失
    return reconstruction_loss + lambda_m * matrix_profile_loss

# === 数据窗口化函数 ===

def create_windows_multivariate(data, labels, window_size, step_size):
    """
    创建包含标签的滑动窗口，每个窗口的标签由其中是否包含异常值决定。
    
    参数:
    - data: 原始时间序列数据（样本数，特征数）
    - labels: 对应的标签（样本数）
    - window_size: 窗口大小
    - step_size: 步长
    
    返回:
    - windows: 创建的窗口数据
    - window_labels: 每个窗口对应的标签
    """
    windows = []
    window_labels = []

    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window_data = data[start:end]
        window_label = 1 if np.any(labels[start:end] == 1) else 0  # 如果窗口内有异常值，标记为异常

        windows.append(window_data)
        window_labels.append(window_label)

    return np.array(windows), np.array(window_labels)


# === 模型权重初始化 ===
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# === 相似度矩阵计算 ===
def calculate_batch_similarity_matrix(features):
    """
    计算特征表示的相似度矩阵
    """
    norms = torch.norm(features, dim=1)  # 计算每个特征的范数
    similarity_matrix = torch.matmul(features, features.T)  # 计算点积
    similarity_matrix /= norms.unsqueeze(1)  # 按照列归一化
    similarity_matrix /= norms.unsqueeze(0)  # 按照行归一化
    return similarity_matrix



# === 优化后的矩阵谱计算 ===
def calculate_optimized_matrix_profile(similarity_matrix, window_size):
    if similarity_matrix.shape[0] <= window_size:
        #print(f"Warning: similarity_matrix.shape[0] ({similarity_matrix.shape[0]}) <= window_size ({window_size})")
        return torch.zeros(1).to(similarity_matrix.device)  # 返回零张量以避免错误

    matrix_profile = []
    for i in range(similarity_matrix.shape[0] - window_size):
        profile = np.mean(similarity_matrix[i:i + window_size, i:i + window_size])
        matrix_profile.append(profile)

    # 正确创建 PyTorch 张量
    return torch.tensor(matrix_profile, dtype=torch.float32).clone().detach().to(similarity_matrix.device)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path='checkpoint.pth'):
        """
        初始化早停机制
        :param patience: 容忍验证损失未下降的epoch数量
        :param delta: 最小变化量
        :param verbose: 是否打印信息
        :param path: 保存模型的路径
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        # 初始化时设置最优损失
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        # 验证损失是否有显著下降
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """保存当前最优模型"""
        if self.verbose:
            print("Validation loss decreased. Saving model...")
        torch.save(model.state_dict(), self.path)
# === 3. 模型定义 ===

class TimeSeriesCNN(nn.Module):
    def __init__(self, window_size, num_variables, step_size=1):
        super(TimeSeriesCNN, self).__init__()
        self.window_size = window_size
        self.num_variables = num_variables
        self.step_size = step_size  # 滑窗步长

        # 定义三层卷积层
        self.conv1 = nn.Conv1d(in_channels=num_variables, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # 定义最大池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout=nn.Dropout(p=0.3)

        # 动态计算卷积层的输出尺寸
        effective_window_size = (window_size - step_size + step_size) // step_size
        # Conv1 输出尺寸
        conv1_output_size = (effective_window_size - 3 + 2 * 1) // 1 + 1  # kernel_size=3, padding=1
        pool1_output_size = conv1_output_size // 2  # MaxPool kernel_size=2

        # Conv2 输出尺寸
        conv2_output_size = (pool1_output_size - 3 + 2 * 1) // 1 + 1
        pool2_output_size = conv2_output_size // 2  # MaxPool kernel_size=2

        # Conv3 输出尺寸
        conv3_output_size = (pool2_output_size - 3 + 2 * 1) // 1 + 1
        pool3_output_size = conv3_output_size // 2  # MaxPool kernel_size=2

        # 动态设置全连接层输入尺寸
        self.fc_input_size = 64 * pool3_output_size  # 64是conv3输出的通道数

        # 根据window_size动态计算全连接层输出维度
        self.fc_output_size = self.fc_input_size  # 你可以根据实际需求设置它

        # 全连接层
        self.fc = nn.Linear(self.fc_input_size, self.fc_output_size)  # 输出维度为动态计算的 fc_output_size

    def forward(self, x):
        # 三层卷积和池化
        x = self.pool(F.relu(self.conv1(x)))  # 第一次卷积和池化
        x=self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))  # 第二次卷积和池化
        x=self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))  # 第三次卷积和池化

        x = x.view(x.size(0), -1)  # Flatten操作，将特征图展开
        return self.fc(x)  # 输出根据动态计算的fc_output_size

# === 2. Autoencoder 定义 ===
class Autoencoder(nn.Module):
    def __init__(self, input_dim, output_dim=38000):  # input_dim 根据 CNN 的输出维度
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),  # 输入的维度由 CNN 输出决定
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),  # 输出维度与原始数据一致
            nn.Sigmoid()  # 输出归一化
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# === 3. ADModel 定义 ===
class ADModel(nn.Module):
    def __init__(self, window_size, num_variables, step_size=1):
        super(ADModel, self).__init__()
        self.feature_extractor = TimeSeriesCNN(window_size, num_variables, step_size=step_size)

        # 自动调整 Autoencoder 输入维度
        cnn_output_dim = self.feature_extractor.fc_output_size  # 直接使用 TimeSeriesCNN 计算的输出维度
        print(f"Feature extractor output dimension: {cnn_output_dim}")  # 打印 CNN 输出维度
        self.autoencoder = Autoencoder(input_dim=cnn_output_dim)

    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        flat_input = features.view(features.size(0), -1)  # Flatten 操作
        reconstructed = self.autoencoder(flat_input)
        return features, reconstructed, None


# === 可视化工具 ===
def plot_loss_values(loss_values, x_label, y_label, title="Loss over Epochs"):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label="Loss")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_roc_curve(true_labels, scores):
    fpr, tpr, _ = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def plot_pr_curve(true_labels, scores):
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(10, 5))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()
