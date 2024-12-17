import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from model import TimeSeriesCNN, combined_loss_function, create_windows_multivariate, \
    calculate_batch_similarity_matrix, calculate_optimized_matrix_profile,  weights_init, calculate_internal_matrix_profile
from torch.utils.data import DataLoader, TensorDataset
from model import TimeSeriesCNN, AttentionAutoencoder 



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
    labels = np.loadtxt(labels_file)  

  
    for i in range(train_data.shape[1]):  
        if np.any(np.isnan(train_data[:, i])):  
            col_mean = np.nanmean(train_data[:, i])  
            train_data[np.isnan(train_data[:, i]), i] = col_mean  #


    return train_data, test_data, labels


# === Early Stopping Helper Class ===
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
           
            self.best_loss = val_loss
            self.counter = 0
           
            torch.save({
                'cnn_encoder': cnn_encoder.state_dict(),
                'autoencoder': autoencoder.state_dict(),
            }, self.save_path)
        else:
           
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def apply_point_adjustment(labels, predictions):

    anomaly_state = False  
    for i in range(len(labels)):
        
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


# === 训练函数 ===
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
       
            loss = combined_loss_function(reconstructed, data.reshape(data.size(0), -1), matrix_profile, lambda_m, segment_length, lambda_internal_mp)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)


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
def test_model(cnn_encoder, autoencoder, test_loader, labels, window_size, segment_length, lambda_1=0.3, lambda_2=0.3, lambda_3=0.4, tau=0.5, step_size=1):
    """
    测试模型并计算各种性能指标。
    
    参数：
    - cnn_encoder: 训练好的 CNN 编码器模型
    - autoencoder: 训练好的自编码器模型
    - test_loader: 测试数据加载器
    - labels: 测试数据的真实标签
    - window_size: 窗口大小
    - segment_length: 切片长度，用于计算矩阵剖面
    - lambda_1, lambda_2, lambda_3: 权重超参数
    - tau: 阈值，用于判定异常
    - step_size: 滑动窗口的步长
    
    返回：
    - combined_scores: 计算的异常分数
    - predictions: 预测的标签（正常/异常）
    """
    cnn_encoder.eval()
    autoencoder.eval()

    reconstruction_errors = []
    mploss_values = []
    combined_scores = []
    predictions = []

    with torch.no_grad():
        for data in test_loader:
            data = data[0]
            data = data.transpose(1, 2)

    
            features = cnn_encoder(data)

            reconstructed = autoencoder(features.reshape(features.size(0), -1))

   
            errors = torch.abs(reconstructed - data.reshape(data.size(0), -1))
            mean_errors = torch.mean(errors, dim=1)

     
            similarity_matrix = calculate_batch_similarity_matrix(features)
            mploss_inter = calculate_optimized_matrix_profile(similarity_matrix, window_size)

            internal_mp_loss = 0.0
            for i in range(reconstructed.size(0)):  
                window_data = reconstructed[i].view(-1, reconstructed.size(1))
                internal_matrix_profile = calculate_internal_matrix_profile(window_data, segment_length)
                internal_mp_loss += internal_matrix_profile.mean()

       
            weights_inter = torch.softmax(mploss_inter, dim=0)
            weights_intra = torch.softmax(internal_mp_loss, dim=0)

            # 使用 calculate_anomaly_score 函数计算异常分数
            anomaly_scores, predicted_labels = calculate_anomaly_score(
                data, labels, model=cnn_encoder, 
                window_size=window_size, step_size=step_size,
                lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, tau=tau
            )

            
            reconstruction_errors.append(mean_errors.detach().cpu().tolist())
            mploss_values.append(mploss_inter.detach().cpu().tolist())
            combined_scores.append(anomaly_scores.detach().cpu().tolist())

            # 预测标签 (基于阈值判定异常)
            predictions.append((anomaly_scores > torch.quantile(anomaly_scores, 0.90)).int())  # 90% 分位点为阈值


    combined_scores = np.concatenate(combined_scores)
    predictions = np.concatenate(predictions)

  
    predictions = apply_point_adjustment(labels, predictions)

    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

  
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
    train_file = "SMD/train/machine-1-1.txt"
    test_file = "SMD/test/machine-1-1.txt"
    labels_file = "SMD/test_label/machine-1-1.txt"

    window_size = 100
    step_size = 5
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    lambda_m = 0.3  
    lambda_internal_mp = 0.3  
    segment_length = 20  

   
    train_data, test_data, labels = read_smd_data(train_file, test_file, labels_file)

  
    print("Generating train windows...")
    train_windows = create_windows_multivariate(train_data, None, window_size, step_size)
    print(f"Shape of train windows: {train_windows.shape}")

 
    print("Generating test windows with labels...")
    test_windows, test_labels = create_windows_multivariate(test_data, labels, window_size, step_size)
    print(f"Shape of test windows: {test_windows.shape}, Shape of test labels: {test_labels.shape}")

    train_size = int(0.8 * len(train_windows))
    val_size = len(train_windows) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        TensorDataset(torch.tensor(train_windows, dtype=torch.float32)),
        [train_size, val_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_windows, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )

 
    num_variables = train_data.shape[1]
    cnn_encoder = TimeSeriesCNN(window_size, num_variables)
    fc_input_size = cnn_encoder.fc_input_size
    
    
    autoencoder = AttentionAutoencoder(input_dim=fc_input_size, output_dim=window_size * num_variables)

    cnn_encoder.apply(weights_init)
    autoencoder.apply(weights_init)

    
    optimizer = torch.optim.Adam(list(cnn_encoder.parameters()) + list(autoencoder.parameters()), lr=learning_rate, weight_decay=1e-5)

  
    print("Starting training...")
    train_loss, val_loss = train_model(
        cnn_encoder, 
        autoencoder, 
        train_loader, 
        val_loader, 
        optimizer, 
        window_size=window_size, 
        segment_length=segment_length,  
        num_epochs=num_epochs, 
        lambda_m=lambda_m, 
        lambda_internal_mp=lambda_internal_mp,  
        patience=5
    )

    
    print("Testing model...")
    combined_scores, predictions = test_model(
        cnn_encoder, 
        autoencoder, 
        test_loader, 
        test_labels, 
        window_size=window_size, 
        segment_length=segment_length,  
        lambda_m=lambda_m, 
        lambda_internal_mp=lambda_internal_mp  
    )


    predictions = apply_point_adjustment(test_labels, predictions)
