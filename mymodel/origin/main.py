import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from model import TimeSeriesCNN, Autoencoder, combined_loss_function, create_windows_multivariate, \
    calculate_batch_similarity_matrix, calculate_optimized_matrix_profile, plot_roc_curve, plot_pr_curve, weights_init, plot_loss_values

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


def read_smd_data(train_file, test_file, labels_file):
    """
    读取 SMD 多变量时间序列数据
    Args:
        train_file: 训练文件路径
        test_file: 测试文件路径
        labels_file: 标签文件路径
    Returns:
        train_data: 训练数据
        test_data: 测试数据
        labels: 测试标签
    """
    train_data = np.loadtxt(train_file, delimiter=',')
    test_data = np.loadtxt(test_file, delimiter=',')
    labels = np.loadtxt(labels_file)  # 标签是单列，0表示正常，1表示异常

    # 检查 NaN 值并进行均值填充
    for i in range(train_data.shape[1]):  # 遍历每一列
        if np.any(np.isnan(train_data[:, i])):  # 如果该列有 NaN
            col_mean = np.nanmean(train_data[:, i])  # 计算该列均值，忽略 NaN
            train_data[np.isnan(train_data[:, i]), i] = col_mean  # 填充 NaN 值为均值

    for i in range(test_data.shape[1]):  # 遍历每一列
        if np.any(np.isnan(test_data[:, i])):  # 如果该列有 NaN
            col_mean = np.nanmean(test_data[:, i])  # 计算该列均值，忽略 NaN
            test_data[np.isnan(test_data[:, i]), i] = col_mean  # 填充 NaN 值为均值

    return train_data, test_data, labels

# === 训练函数 ===
def train_model(cnn_encoder, autoencoder, train_loader, optimizer, num_epochs=50, lambda_m=0.3):
    cnn_encoder.train()
    autoencoder.train()
    total_loss = 0.0
    train_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (data) in enumerate(train_loader):
            data = data[0].float()
            data = data.transpose(1, 2)

            optimizer.zero_grad()
            features = cnn_encoder(data)
            reconstructed = autoencoder(features)
            reconstruction_loss = nn.MSELoss()(reconstructed, data.reshape(data.size(0), -1))
            similarity_matrix = calculate_batch_similarity_matrix(features)
            matrix_profile = calculate_optimized_matrix_profile(similarity_matrix, window_size)
            matrix_profile_loss = torch.mean(matrix_profile)
            loss = reconstruction_loss + lambda_m * matrix_profile_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_loss.append(avg_epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")
    
    return train_loss

# === 测试函数 ===
def test_model(cnn_encoder, autoencoder, test_loader):
    autoencoder.eval()
    cnn_encoder.eval()
    reconstruction_errors = []
    predictions = []

    with torch.no_grad():
        for data in test_loader:
            data = data[0]
            data = data.transpose(1, 2)
            features = cnn_encoder(data)
            reconstructed = autoencoder(features.reshape(features.size(0), -1))
            errors = torch.abs(reconstructed - data.reshape(data.size(0), -1))
            mean_errors = torch.mean(errors, dim=1)
            threshold = torch.quantile(mean_errors, 0.95)
            batch_predictions = (mean_errors > threshold).int()
            reconstruction_errors.append(mean_errors.detach().cpu().tolist())
            predictions.append(batch_predictions.detach().cpu().tolist())

    return np.concatenate(reconstruction_errors), np.concatenate(predictions)

# === 主程序入口 ===
if __name__ == "__main__":
    # 设置随机种子
    seed = 42
    set_random_seed(seed)

    # 参数设置
    train_file = "SMD/train/machine-1-1.txt"
    test_file = "SMD/test/machine-1-1.txt"
    labels_file = "SMD/test_label/machine-1-1.txt"

    window_size = 100
    step_size = window_size // 2

    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    lambda_m = 0.3

    # 读取数据
    train_data, test_data, labels = read_smd_data(train_file, test_file, labels_file)

    # 创建窗口数据
    print("Generating windows with labels...")
    train_windows, train_labels = create_windows_multivariate(train_data, labels, window_size, step_size)
    print(f"Shape of train windows: {train_windows.shape}, Shape of train labels: {train_labels.shape}")
    test_windows, test_labels = create_windows_multivariate(test_data, labels, window_size, step_size)
    print(f"Shape of test windows: {test_windows.shape}, Shape of test labels: {test_labels.shape}")

    # 创建 DataLoader
    train_dataset = TensorDataset(torch.tensor(train_windows, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_windows, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_variables = train_data.shape[1]
    cnn_encoder = TimeSeriesCNN(window_size, num_variables)
    fc_input_size = cnn_encoder.fc_input_size
    autoencoder = Autoencoder(input_dim=fc_input_size, output_dim=window_size * num_variables)

    cnn_encoder.apply(weights_init)
    autoencoder.apply(weights_init)

    optimizer = torch.optim.Adam(list(cnn_encoder.parameters()) + list(autoencoder.parameters()), lr=learning_rate,weight_decay=1e-5)

    # 训练模型
    print("Starting training...")
    train_loss = train_model(cnn_encoder, autoencoder, train_loader, optimizer, num_epochs=num_epochs, lambda_m=lambda_m)
    #plot_loss_values(train_loss, x_label='epochs', y_label='loss')

    # 测试模型
    print("Testing model...")
    reconstruction_errors, predictions = test_model(cnn_encoder, autoencoder, test_loader)

    # 评估
    print("Evaluating anomaly detection...")
    threshold = np.percentile(reconstruction_errors, 95)
    predictions = (reconstruction_errors > threshold).astype(int)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    # 计算 AUC 分数
    roc_auc = roc_auc_score(test_labels, reconstruction_errors)
    precision_vals, recall_vals, _ = precision_recall_curve(test_labels, reconstruction_errors)
    pr_auc = auc(recall_vals, precision_vals)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f},ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    

    # 绘制曲线
    #plot_roc_curve(test_labels, reconstruction_errors)
    #plot_pr_curve(test_labels, reconstruction_errors)