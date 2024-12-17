import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from model import TimeSeriesCNN, AttentionAutoencoder, combined_loss_function, \
    calculate_batch_similarity_matrix, calculate_optimized_matrix_profile,  weights_init,calculate_internal_matrix_profile

def set_random_seed(seed):
    """
    设置随机种子以确保结果可复现
    Args:
        seed: 整数值，随机种子
    """
    random.seed(seed)              # 设置 Python 的随机数种子
    np.random.seed(seed)           # 设置 numpy 的随机数种子
    torch.manual_seed(seed)        # 设置 PyTorch 的随机数种子（CPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)           # 设置 PyTorch 的随机数种子（当前 GPU）
        torch.cuda.manual_seed_all(seed)       # 设置 PyTorch 的随机数种子（所有 GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积算法选择一致
    torch.backends.cudnn.benchmark = False     # 关闭自动优化以确保结果一致


# === Early Stopping 类 ===
class EarlyStopping:
    def __init__(self, patience=5, delta=0, save_path="best_model.pth"):
        """
        Args:
            patience (int): How long to wait after the last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, cnn_encoder, autoencoder):
        if val_loss < self.best_loss - self.delta:
            # Improvement detected, reset counter
            self.best_loss = val_loss
            self.counter = 0
            # Save best model
            torch.save({
                'cnn_encoder': cnn_encoder.state_dict(),
                'autoencoder': autoencoder.state_dict(),
            }, self.save_path)
        else:
            # No improvement, increase counter
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



def read_psm_data(train_file, test_file, labels_file):
    """
    读取 PSM 多变量时间序列数据，去除时间步列，并处理缺失值或非法值
    Args:
        train_file: 训练文件路径
        test_file: 测试文件路径
        labels_file: 标签文件路径
    Returns:
        train_data: 训练数据 (不含时间步)
        test_data: 测试数据 (不含时间步)
        labels: 测试标签
    """
    # 用 pandas 读取数据，处理空值
    train_data = pd.read_csv(train_file, delimiter=',').iloc[:, 1:]  # 跳过时间步列
    test_data = pd.read_csv(test_file, delimiter=',').iloc[:, 1:]
    labels = pd.read_csv(labels_file, delimiter=',', header=None)  # 标签文件

    # 转换为 numpy 数组
    train_data = train_data.replace('', np.nan).astype(float).to_numpy()
    test_data = test_data.replace('', np.nan).astype(float).to_numpy()
    labels = labels.to_numpy()

    # 检查并处理缺失值
    if np.any(np.isnan(train_data)):
        print("Found missing values in training data, filling with column mean.")
        train_data = np.nan_to_num(train_data, nan=np.nanmean(train_data, axis=0))
    if np.any(np.isnan(test_data)):
        print("Found missing values in test data, filling with column mean.")
        test_data = np.nan_to_num(test_data, nan=np.nanmean(test_data, axis=0))

    return train_data, test_data, labels

def apply_point_adjustment(labels, predictions):
    """
    应用 Point Adjustment 技巧，将异常序列调整为连续的异常段。

    参数：
    - labels: 真实标签
    - predictions: 模型的预测标签

    返回：
    - 调整后的预测标签
    """
    anomaly_state = False  # 标记是否处于异常段
    for i in range(len(labels)):
        # 当真实标签和预测标签均为异常时，开始调整
        if labels[i] == 1 and predictions[i] == 1 and not anomaly_state:
            anomaly_state = True

            for j in range(i, 0, -1):
                if labels[j] == 0:
                    break
                else:
                    if predictions[j] == 0:
                        predictions[j] = 1

            for j in range(i, len(labels)):
                if labels[j] == 0:
                    break
                else:
                    if predictions[j] == 0:
                        predictions[j] = 1

        elif labels[i] == 0:
            anomaly_state = False


        if anomaly_state:
            predictions[i] = 1

    return predictions
    
def create_windows_multivariate(data, labels=None, window_size=100, step_size=50):
    """
    创建包含标签的滑动窗口，每个窗口的标签由其中是否包含异常值决定。
    
    参数:
    - data: 原始时间序列数据（样本数，特征数）
    - labels: 对应的标签（样本数，可选）
    - window_size: 窗口大小
    - step_size: 步长
    
    返回:
    - windows: 创建的窗口数据
    - window_labels: 每个窗口对应的标签（如果未提供 labels，则返回 None）
    """
    windows = []
    window_labels = []

    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window_data = data[start:end]
        windows.append(window_data)

        # 如果提供了标签，则生成窗口标签
        if labels is not None:
            window_label = 1 if np.any(labels[start:end] == 1) else 0  # 如果窗口内有异常值，标记为异常
            window_labels.append(window_label)

    return np.array(windows), np.array(window_labels) if labels is not None else None



def train_model(cnn_encoder, autoencoder, train_loader, val_loader, optimizer, num_epochs=50, lambda_m=0.3, segment_length=10, lambda_internal_mp=0.1, patience=5, window_size=100):
    cnn_encoder.train()
    autoencoder.train()
    train_loss_history = []
    val_loss_history = []

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, save_path="best_model.pth")

    for epoch in range(num_epochs):
        # Training loop
        cnn_encoder.train()
        autoencoder.train()
        train_loss = 0.0
        for batch_idx, (data) in enumerate(train_loader):
            data = data[0].float()
            data = data.transpose(1, 2)

            optimizer.zero_grad()
            features = cnn_encoder(data)
            reconstructed = autoencoder(features)
            similarity_matrix = calculate_batch_similarity_matrix(features)
            matrix_profile = calculate_optimized_matrix_profile(similarity_matrix, window_size)
            
            # 计算损失
            loss = combined_loss_function(reconstructed, data.reshape(data.size(0), -1), matrix_profile, lambda_m, segment_length, lambda_internal_mp)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation loop
        cnn_encoder.eval()
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data) in enumerate(val_loader):
                data = data[0].float()
                data = data.transpose(1, 2)
                features = cnn_encoder(data)
                reconstructed = autoencoder(features)
                similarity_matrix = calculate_batch_similarity_matrix(features)
                matrix_profile = calculate_optimized_matrix_profile(similarity_matrix, window_size)
                
                # 计算损失
                loss = combined_loss_function(reconstructed, data.reshape(data.size(0), -1), matrix_profile, lambda_m, segment_length, lambda_internal_mp)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Check early stopping
        early_stopping(avg_val_loss, cnn_encoder, autoencoder)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    return train_loss_history, val_loss_history

# === 测试函数 ===
def test_model(cnn_encoder, autoencoder, test_loader, labels, window_size, segment_length, lambda_m=0.3, lambda_internal_mp=0.1):
    autoencoder.eval()
    cnn_encoder.eval()
    reconstruction_errors = []
    mploss_values = []
    combined_scores = []
    predictions = []

    with torch.no_grad():
        for data in test_loader:
            data = data[0]
            data = data.transpose(1, 2)  # 转换数据维度，符合 CNN 输入要求
            features = cnn_encoder(data)  # 从 CNN 编码器提取特征
            reconstructed = autoencoder(features.reshape(features.size(0), -1))  # 重构输入数据
            
            # 计算重构误差
            errors = torch.abs(reconstructed - data.reshape(data.size(0), -1))  # 计算重构误差
            mean_errors = torch.mean(errors, dim=1)  # 对每个样本计算均值重构误差

            # 计算窗口间矩阵谱损失
            similarity_matrix = calculate_batch_similarity_matrix(features)
            mploss_inter = calculate_optimized_matrix_profile(similarity_matrix, window_size)  # 计算窗口间的Matrix Profile损失

            # 计算窗口内部相似度损失
            internal_mp_loss = 0.0
            for i in range(reconstructed.size(0)):  # 遍历每个样本
                window_data = reconstructed[i].view(-1, reconstructed.size(1))  # 展平后作为窗口数据
                internal_matrix_profile = calculate_internal_matrix_profile(window_data, segment_length)  # 计算窗口内部的Matrix Profile
                internal_mp_loss += internal_matrix_profile.mean()  # 计算平均Matrix Profile损失

            # 计算窗口间和窗口内的矩阵谱损失的权重 (Softmax计算)
            weights_inter = torch.softmax(mploss_inter, dim=0)  # 对窗口间矩阵谱损失应用Softmax
            weights_intra = torch.softmax(internal_mp_loss, dim=0)  # 对窗口内矩阵谱损失应用Softmax
            
            # 计算最终的异常分数
            anomaly_scores = lambda_m * (weights_inter * mploss_inter) + lambda_internal_mp * (weights_intra * internal_mp_loss) + mean_errors
            
            # 保存结果
            reconstruction_errors.append(mean_errors.detach().cpu().tolist())
            mploss_values.append(mploss_inter.detach().cpu().tolist())
            combined_scores.append(anomaly_scores.detach().cpu().tolist())

            # 根据异常分数来预测异常 (大于阈值的认为是异常)
            predictions.append((anomaly_scores > torch.quantile(anomaly_scores, 0.90)).int())  # 使用异常分数的90百分位数作为阈值

    # 转换为 NumPy 数组
    combined_scores = np.concatenate(combined_scores)
    predictions = np.concatenate(predictions)

    # 应用 Point Adjustment (PA) 技巧
    predictions = apply_point_adjustment(labels, predictions)
    print(labels.shape, combined_scores.shape)

    # 计算评估指标
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    # 计算 AUC 分数
    roc_auc = roc_auc_score(labels, combined_scores)
    precision_vals, recall_vals, _ = precision_recall_curve(labels, combined_scores)
    pr_auc = auc(recall_vals, precision_vals)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

    return combined_scores, predictions



if __name__ == "__main__":
    # 设置随机种子
    seed = 42
    set_random_seed(seed)

    # 参数设置
    train_file = "PSM/train.csv"
    test_file = "PSM/test.csv"
    labels_file = "PSM/test_label.csv"

    window_size = 100
    step_size = 10
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    lambda_m = 0.3  # 矩阵谱损失的权重
    lambda_internal_mp = 0.3  # 新增：窗口内部矩阵谱损失的权重
    segment_length = 20  # 第二步窗口化的片段长度超参数，方便调参

    # 读取数据
    train_data, test_data, labels = read_psm_data(train_file, test_file, labels_file)

    # === 创建窗口数据 ===
    print("Generating windows with labels...")

    # 生成训练数据窗口，无需提供标签
    train_windows, _ = create_windows_multivariate(train_data, labels=None, window_size=window_size, step_size=step_size)
    print(f"Shape of train windows: {train_windows.shape}")

    # 生成测试数据窗口，包含真实标签
    test_windows, test_labels = create_windows_multivariate(test_data, labels=labels, window_size=window_size, step_size=step_size)
    print(f"Shape of test windows: {test_windows.shape}, Shape of test labels: {test_labels.shape}")

   # 划分训练集和验证集
    train_size = int(0.8 * len(train_windows))
    val_size = len(train_windows) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        TensorDataset(torch.tensor(train_windows, dtype=torch.float32)),
        [train_size, val_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 测试数据的 DataLoader，测试标签仅用于评估
    # 测试数据的 DataLoader，确保同时包含数据和标签用于评估
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_windows, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

    # 模型初始化
    num_variables = train_data.shape[1]
    cnn_encoder = TimeSeriesCNN(window_size, num_variables)
    fc_input_size = cnn_encoder.fc_input_size
    
    # 使用带有注意力机制的自编码器替代原自编码器
    autoencoder = AttentionAutoencoder(input_dim=fc_input_size, output_dim=window_size * num_variables)

    cnn_encoder.apply(weights_init)
    autoencoder.apply(weights_init)

    # 优化器
    optimizer = torch.optim.Adam(list(cnn_encoder.parameters()) + list(autoencoder.parameters()), lr=learning_rate, weight_decay=1e-5)

    # 训练模型
    print("Starting training...")
    train_loss, val_loss = train_model(
        cnn_encoder, 
        autoencoder, 
        train_loader, 
        val_loader, 
        optimizer, 
        window_size=window_size, 
        segment_length=segment_length,  # 传递 segment_length 参数
        num_epochs=num_epochs, 
        lambda_m=lambda_m, 
        lambda_internal_mp=lambda_internal_mp,  # 传递 lambda_internal_mp 参数
        patience=5
    )

    # 测试模型
    print("Testing model...")
    combined_scores, predictions = test_model(
        cnn_encoder, 
        autoencoder, 
        test_loader, 
        test_labels, 
        window_size=window_size, 
        segment_length=segment_length,  # 传递 segment_length 参数
        lambda_m=lambda_m, 
        lambda_internal_mp=lambda_internal_mp  # 传递 lambda_internal_mp 参数
    )

    # 应用 Point Adjustment 技巧
    predictions = apply_point_adjustment(test_labels, predictions)


