import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from model import TimeSeriesCNN, AttentionAutoencoder, combined_loss_function, \
    calculate_batch_similarity_matrix, calculate_optimized_matrix_profile,  weights_init,calculate_internal_matrix_profile



def read_smap_data(train_file, test_file):
    """
    读取 SMAP 多变量时间序列数据
    Args:
        train_file: 训练文件路径
        test_file: 测试文件路径
    Returns:
        train_data: 训练数据
        test_data: 测试数据
        test_labels: 测试标签
    """
 
    train_data = np.genfromtxt(train_file, delimiter=',', skip_header=1)
    test_data = np.genfromtxt(test_file, delimiter=',', skip_header=1)

   
    train_data = train_data[:, 1:]
    test_data = test_data[:, 1:]

    test_labels = test_data[:, -1]  
    test_data = test_data[:, :-1]  

    # 检查 NaN 值并进行均值填充
    for i in range(train_data.shape[1]):  
        if np.any(np.isnan(train_data[:, i])):  
            col_mean = np.nanmean(train_data[:, i])  
            train_data[np.isnan(train_data[:, i]), i] = col_mean  

    for i in range(test_data.shape[1]): 
        if np.any(np.isnan(test_data[:, i])): 
            col_mean = np.nanmean(test_data[:, i])  
            test_data[np.isnan(test_data[:, i]), i] = col_mean  

    return train_data, test_data, test_labels

def create_windows_multivariate(data, labels, window_size, step_size):
    """
    创建包含标签的滑动窗口，每个窗口的标签由其中是否包含异常值决定。
    
    参数:
    - data: 原始时间序列数据（样本数，特征数）
    - labels: 对应的标签（样本数）或 None
    - window_size: 窗口大小
    - step_size: 步长
    
    返回:
    - windows: 创建的窗口数据
    - window_labels: 每个窗口对应的标签（如果 labels 不为 None）
    """
    windows = []
    window_labels = []

    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window_data = data[start:end] 
        
        
        if labels is not None:
            window_label = 1 if np.any(labels[start:end] == 1) else 0 
            window_labels.append(window_label)
        
        windows.append(window_data)

    
    windows = np.array(windows)
 
    if labels is not None:
        window_labels = np.array(window_labels)
        return windows, window_labels
    else:
        return windows, None


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
            data = data.transpose(1, 2) 
            features = cnn_encoder(data)  
            reconstructed = autoencoder(features.reshape(features.size(0), -1)) 
            
            # 计算重构误差
            errors = torch.abs(reconstructed - data.reshape(data.size(0), -1))  
            mean_errors = torch.mean(errors, dim=1) 

            # 计算窗口间矩阵谱损失
            similarity_matrix = calculate_batch_similarity_matrix(features)
            mploss_inter = calculate_optimized_matrix_profile(similarity_matrix, window_size)  

            # 计算窗口内部相似度损失
            internal_mp_loss = 0.0
            for i in range(reconstructed.size(0)):  
                window_data = reconstructed[i].view(-1, reconstructed.size(1))  
                internal_matrix_profile = calculate_internal_matrix_profile(window_data, segment_length)  
                internal_mp_loss += internal_matrix_profile.mean()  

            # 计算窗口间和窗口内的矩阵谱损失的权重 (Softmax计算)
            weights_inter = torch.softmax(mploss_inter, dim=0)  
            weights_intra = torch.softmax(internal_mp_loss, dim=0) 
            
            # 计算最终的异常分数
            anomaly_scores = lambda_m * (weights_inter * mploss_inter) + lambda_internal_mp * (weights_intra * internal_mp_loss) + mean_errors
            
            # 保存结果
            reconstruction_errors.append(mean_errors.detach().cpu().tolist())
            mploss_values.append(mploss_inter.detach().cpu().tolist())
            combined_scores.append(anomaly_scores.detach().cpu().tolist())

            # 根据异常分数来预测异常 (大于阈值的认为是异常)
            predictions.append((anomaly_scores > torch.quantile(anomaly_scores, 0.90)).int()) 

 
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

    seed = 42
    set_random_seed(seed)


    train_file = "processed_csv/MSL_train.csv"
    test_file = "processed_csv/MSL_test.csv"

    window_size = 100
    step_size = 5
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.0001
    lambda_m = 0.5 
    lambda_internal_mp = 0.5 
    segment_length = 20 


    train_data, test_data, test_labels = read_smap_data(train_file, test_file)

    print("Generating windows with labels...")


    train_windows, _ = create_windows_multivariate(train_data, None, window_size, step_size)
    print(f"Shape of train windows: {train_windows.shape}")
    
 
    test_windows, test_labels = create_windows_multivariate(test_data, test_labels, window_size, step_size)
    print(f"Shape of test windows: {test_windows.shape}, Shape of test labels: {test_labels.shape}")


    val_size = int(0.2 * len(train_windows))
    val_windows = train_windows[:val_size]
    train_windows = train_windows[val_size:]

  
    train_dataset = TensorDataset(torch.tensor(train_windows, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_windows, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_windows, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        segment_length=segment_length,  # 传递 segment_length 参数
        num_epochs=num_epochs, 
        lambda_m=lambda_m, 
        lambda_internal_mp=lambda_internal_mp,  # 传递 lambda_internal_mp 参数
        patience=5
    )

 
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


    predictions = apply_point_adjustment(test_labels, predictions)
