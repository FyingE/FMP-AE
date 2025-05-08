# === processing.py ===

import numpy as np
import torch
import torch.nn as nn


# === 数据窗口化函数 ===
def create_windows_multivariate(data, labels, window_size, step_size):
    windows = []
    window_labels = []

    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window_data = data[start:end]

        if labels is not None:
            window_label = 1 if np.any(labels[start:end] == 1) else 0
            window_labels.append(window_label)

        windows.append(window_data)

    if labels is not None:
        return np.array(windows), np.array(window_labels)
    else:
        return np.array(windows)


# === 联合损失函数 ===
def combined_loss_function(
    reconstructed,
    original,
    matrix_profile,
    lambda_m=0.3,
    segment_length=10,
    lambda_internal_mp=0.1,
):
    reconstruction_loss = nn.MSELoss()(reconstructed, original)
    matrix_profile_loss = torch.mean(matrix_profile)

    internal_mp_loss = 0.0
    for i in range(reconstructed.size(0)):
        window_data = reconstructed[i].view(-1, reconstructed.size(1))
        internal_matrix_profile = calculate_internal_matrix_profile(
            window_data, segment_length
        )
        internal_mp_loss += internal_matrix_profile.mean()

    total_loss = (
        reconstruction_loss
        + lambda_m * matrix_profile_loss
        + lambda_internal_mp * internal_mp_loss
    )
    return total_loss


# === 相似度计算 ===
def calculate_batch_similarity_matrix(features):
    norms = torch.norm(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    similarity_matrix /= norms.unsqueeze(1)
    similarity_matrix /= norms.unsqueeze(0)
    return similarity_matrix


# === 外部Matrix Profile计算 ===
def calculate_optimized_matrix_profile(similarity_matrix, window_size):
    if similarity_matrix.shape[0] <= window_size:
        return torch.zeros(1).to(similarity_matrix.device)

    matrix_profile = []
    for i in range(similarity_matrix.shape[0] - window_size):
        profile = torch.mean(
            similarity_matrix[i : i + window_size, i : i + window_size]
        )
        matrix_profile.append(profile)

    return (
        torch.tensor(matrix_profile, dtype=torch.float32)
        .clone()
        .detach()
        .to(similarity_matrix.device)
    )


# === 内部Matrix Profile计算 ===
def calculate_internal_matrix_profile(window_data, segment_length):
    num_segments = window_data.shape[0]

    segments = []
    for i in range(0, window_data.shape[1], segment_length):
        segment = window_data[:, i : i + segment_length]
        if segment.shape[1] == segment_length:
            segments.append(segment)

    if len(segments) == 0:
        return torch.zeros(1).to(window_data.device)

    segments = torch.stack(segments, dim=0)
    similarity_matrix = torch.cdist(segments, segments, p=2)
    internal_matrix_profile = torch.mean(similarity_matrix, dim=1)

    return internal_matrix_profile
