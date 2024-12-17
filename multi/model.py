import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# === 数据窗口化函数 ===

def create_windows_multivariate(data, labels, window_size, step_size):
    windows = []
    window_labels = []

    # 滑动窗口切分
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window_data = data[start:end]

        # 对应窗口的标签，若窗口内有任何异常值，则标记为异常（1），否则为正常（0）
        if labels is not None:  # 如果提供了标签（用于测试阶段）
            window_label = 1 if np.any(labels[start:end] == 1) else 0
            window_labels.append(window_label)

        windows.append(window_data)

    if labels is not None:
        return np.array(windows), np.array(window_labels)  # 返回数据和标签（用于测试阶段）
    else:
        return np.array(windows)  # 仅返回数据（用于训练阶段）


# === 联合损失函数 ===
def combined_loss_function(reconstructed, original, matrix_profile, lambda_m=0.3, segment_length=10, lambda_internal_mp=0.1):
    """
    计算综合损失，包括重构损失、外部矩阵谱损失和窗口内部的矩阵谱损失。
    """
    # 1. 重构损失
    reconstruction_loss = nn.MSELoss()(reconstructed, original)  # 原始数据的重构损失

    matrix_profile_loss = torch.mean(matrix_profile)  # 外部矩阵谱损失
    
    # 2. 窗口内部矩阵谱损失
    internal_mp_loss = 0.0
    for i in range(reconstructed.size(0)):  # 遍历每个样本
        window_data = reconstructed[i].view(-1, reconstructed.size(1))  # 展平后作为窗口数据
        internal_matrix_profile = calculate_internal_matrix_profile(window_data, segment_length)  # 计算窗口内部的Matrix Profile
        internal_mp_loss += internal_matrix_profile.mean()  # 计算平均Matrix Profile损失

    # 3. 联合损失 (加权方式)
    total_loss = reconstruction_loss + lambda_m * matrix_profile_loss + lambda_internal_mp * internal_mp_loss
    return total_loss




def calculate_batch_similarity_matrix(features):
    """
    计算特征表示的相似度矩阵
    """
    norms = torch.norm(features, dim=1)  
    similarity_matrix = torch.matmul(features, features.T)  
    similarity_matrix /= norms.unsqueeze(1)  
    similarity_matrix /= norms.unsqueeze(0)  
    return similarity_matrix

def calculate_optimized_matrix_profile(similarity_matrix, window_size):
    if similarity_matrix.shape[0] <= window_size:
        #print(f"Warning: similarity_matrix.shape[0] ({similarity_matrix.shape[0]}) <= window_size ({window_size})")
        return torch.zeros(1).to(similarity_matrix.device)  

    matrix_profile = []
    for i in range(similarity_matrix.shape[0] - window_size):
        profile = np.mean(similarity_matrix[i:i + window_size, i:i + window_size])
        matrix_profile.append(profile)

    # 正确创建 PyTorch 张量
    return torch.tensor(matrix_profile, dtype=torch.float32).clone().detach().to(similarity_matrix.device)

def calculate_internal_matrix_profile(window_data, segment_length):
    """
    计算窗口内部切片的 Matrix Profile，相似度基于片段之间的欧几里得距离。
    假设窗口是一个 [num_segments, segment_length] 的矩阵。
    """
    num_segments = window_data.shape[0]
    
    segments = []
    for i in range(0, window_data.shape[1], segment_length):  
        segment = window_data[:, i:i + segment_length]  
        segments.append(segment)
    
    segments = torch.stack(segments, dim=0)  

    similarity_matrix = torch.cdist(segments, segments, p=2)  
    internal_matrix_profile = torch.mean(similarity_matrix, dim=1)  

    return internal_matrix_profile




def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class TimeSeriesCNN(nn.Module):
    def __init__(self, window_size, num_variables, step_size=1):
        super(TimeSeriesCNN, self).__init__()
        self.window_size = window_size
        self.num_variables = num_variables
        self.step_size = step_size  

        
        self.conv1 = nn.Conv1d(in_channels=num_variables, out_channels=16, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm([16, window_size])  
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm([32, window_size // 2])  
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm([64, window_size // 4])  
        
      
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)

        
        effective_window_size = (window_size - step_size + step_size) // step_size
        conv1_output_size = (effective_window_size - 3 + 2 * 1) // 1 + 1
        pool1_output_size = conv1_output_size // 2  # MaxPool kernel_size=2
        conv2_output_size = (pool1_output_size - 3 + 2 * 1) // 1 + 1
        pool2_output_size = conv2_output_size // 2
        conv3_output_size = (pool2_output_size - 3 + 2 * 1) // 1 + 1
        pool3_output_size = conv3_output_size // 2

       
        self.fc_input_size = 64 * pool3_output_size 
        self.fc_output_size = self.fc_input_size  

        self.fc = nn.Linear(self.fc_input_size, self.fc_output_size)  

    def forward(self, x):
        
        x = self.pool(F.relu(self.norm1(self.conv1(x))))  
        x = self.dropout(x)
        x = self.pool(F.relu(self.norm2(self.conv2(x))))  
        x = self.dropout(x)
        x = self.pool(F.relu(self.norm3(self.conv3(x))))  

        x = x.view(x.size(0), -1) 
        return self.fc(x)  



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        
        attn_output, _ = self.attention(x, x, x)
        return attn_output



class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, output_dim=38000):
        super(AttentionAutoencoder, self).__init__()
        
    
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            MultiHeadSelfAttention(embed_dim, num_heads),  
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU()
        )
        
     
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim),
            nn.ReLU(),
            MultiHeadSelfAttention(embed_dim, num_heads),  
            nn.Linear(embed_dim, output_dim),
            nn.Sigmoid()  
        )

    def forward(self, x):
      
        latent = self.encoder(x)
        
        
        reconstructed = self.decoder(latent)
        
        return reconstructed


class TimeSeriesModel(nn.Module):
    def __init__(self, window_size, num_variables, step_size=1, output_dim=38000):
        super(TimeSeriesModel, self).__init__()
        
       
        self.cnn = TimeSeriesCNN(window_size, num_variables, step_size)
        
        
        self.autoencoder = AttentionAutoencoder(input_dim=self.cnn.fc_output_size, output_dim=output_dim)

    def forward(self, x):
      
        features = self.cnn(x)  
       
        reconstructed = self.autoencoder(features)
        
        return reconstructed


import torch
import torch.nn.functional as F
import numpy as np

def calculate_anomaly_score(data, labels, model, window_size, step_size, lambda_1=0.3, lambda_2=0.3, lambda_3=0.4, tau=0.5):
    """
    计算异常分数，包括矩阵剖面和重构误差的加权和。
    
    参数：
    - data: 时序数据，形状为 (num_samples, num_variables, data_length)
    - labels: 标签，1 表示异常，0 表示正常
    - model: 训练好的模型，包括 CNN 和自编码器
    - window_size: 窗口大小
    - step_size: 步长
    - lambda_1, lambda_2, lambda_3: 权重超参数
    - tau: 阈值，用于确定是否为异常
    
    返回：
    - anomaly_scores: 异常分数，形状为 (num_samples,)
    - labels: 预测的标签，1 表示异常，0 表示正常
    """

    windows, _ = create_windows_multivariate(data, labels, window_size, step_size)
  
    anomaly_scores = []
    
    for i in range(len(windows)):
        window_data = torch.tensor(windows[i], dtype=torch.float32).unsqueeze(0) 
        
      
        features = model.cnn(window_data)  
        
        
        reconstructed = model.autoencoder(features) 
        
        reconstruction_error = torch.norm(window_data - reconstructed, p=2).item()  
        
       
        similarity_matrix = calculate_batch_similarity_matrix(features)  
        inter_matrix_profile = calculate_optimized_matrix_profile(similarity_matrix, window_size)
        
        
        intra_matrix_profile = calculate_internal_matrix_profile(window_data, segment_length=10)
       
        w_inter = torch.exp(inter_matrix_profile) / torch.sum(torch.exp(inter_matrix_profile))
        w_intra = torch.exp(intra_matrix_profile) / torch.sum(torch.exp(intra_matrix_profile))
        
       
        weighted_inter_mp = torch.sum(w_inter * inter_matrix_profile)
        weighted_intra_mp = torch.sum(w_intra * intra_matrix_profile)
        
        anomaly_score = lambda_1 * weighted_inter_mp + lambda_2 * weighted_intra_mp + lambda_3 * reconstruction_error
        
        anomaly_scores.append(anomaly_score)
    

    anomaly_scores = np.array(anomaly_scores)
    predicted_labels = (anomaly_scores > tau).astype(int) 
    
    return anomaly_scores, predicted_labels
