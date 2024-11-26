import numpy as np
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from model import create_windows, Autoencoder, plot_loss_values, \
    calculate_batch_similarity_matrix, calculate_optimized_matrix_profile, read_data_with_anomalies, \
    weights_init, plot_original_series_with_anomalies, plot_detected_anomalies, expand_anomalies, \
    plot_matrix_profile, plot_reconstruct_error, combined_loss_function, plot_roc_curve, plot_pr_curve


# 定义绘制特征图的函数
def plot_feature_map(feature_map, title="Feature Map"):
    num_channels = feature_map.shape[1]
    feature_map = feature_map[0].detach().cpu().numpy()

    plt.figure(figsize=(20, 8))
    for i in range(num_channels):
        plt.subplot(1, num_channels, i + 1)
        plt.imshow(feature_map[i], cmap='viridis')
        plt.axis('off')
        plt.title(f'Channel {i + 1}')
    plt.suptitle(title)
    plt.show()


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
        feature_map = feature_map.detach().cpu().numpy()
        num_channels = feature_map.shape[1]

        fig, axs = plt.subplots(2, num_channels // 2, figsize=(20, 8))  # 使用子图

        for i in range(num_channels):
            ax = axs[i // (num_channels // 2), i % (num_channels // 2)]  # 定位每个子图
            ax.plot(feature_map[0, i, :], color='red')  # 绘制红色线条
            ax.set_title(f'{layer_name} - Channel {i + 1}')

            # 设置边框颜色为黑色
            ax.spines['top'].set_color('black')
            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['right'].set_color('black')

            # 添加坐标轴
            ax.set_xlim([0, feature_map.shape[2] - 1])  # 设置x轴范围

            # 调整y轴范围，缩小显示范围来突出特征
            y_min, y_max = feature_map[0, i, :].min(), feature_map[0, i, :].max()
            y_range = y_max - y_min
            ax.set_ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])  # 缩小y轴范围以增加特征的突出性

        plt.tight_layout()
        plt.show()


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    window_size = 100
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 50

    file_path_example = 'UCR_Anomaly_FullData/020_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z2_5000_7175_7388.txt'
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
    test_labels[(anomaly_start - train_val):(anomaly_end - train_val)] = 1
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    cnn_encoder = TimeSeriesCNN(window_size=window_size)
    autoencoder = Autoencoder(input_dim=window_size)

    cnn_encoder.apply(weights_init)
    autoencoder.apply(weights_init)

    optimizer = optim.Adam(list(cnn_encoder.parameters()) + list(autoencoder.parameters()), lr=learning_rate)

    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    loss_values = []
    for epoch in range(num_epochs):
        cnn_encoder.train()
        autoencoder.train()

        lambda_m = 0.01 * (epoch / num_epochs)
        epoch_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            batch = batch[0]
            features = cnn_encoder(batch, visualize=(epoch % 10 == 0))
            reconstructed = autoencoder(batch)

            features_np = features.detach().cpu().numpy()
            sim_matrix = calculate_batch_similarity_matrix(features_np)
            matrix_profile = calculate_optimized_matrix_profile(sim_matrix, window_size=window_size)

            loss = combined_loss_function(reconstructed, batch, torch.tensor(matrix_profile, dtype=torch.float32).clone().detach(), lambda_m=lambda_m)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(cnn_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        loss_values.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    plot_loss_values(loss_values, "Epochs", "Loss")

    torch.save(cnn_encoder.state_dict(), 'cnn_encoder.pth')
    torch.save(autoencoder.state_dict(), 'autoencoder.pth')

    cnn_encoder.eval()
    autoencoder.eval()

    with torch.no_grad():
        test_features = cnn_encoder(test_data)
        reconstructed_test_data = autoencoder(test_data)
        recon_error = torch.mean((reconstructed_test_data - test_data) ** 2, dim=(1, 2)).detach().numpy()

        plot_feature_map(test_features, title="CNN Extracted Feature Map")

    test_features_np = test_features.detach().cpu().numpy()
    test_sim_matrix = calculate_batch_similarity_matrix(test_features_np)
    test_matrix_profile = calculate_optimized_matrix_profile(test_sim_matrix, window_size=window_size)

    threshold_recon_error = np.percentile(recon_error, 95)
    threshold_matrix_profile = np.percentile(test_matrix_profile, 95)

    predicted_labels_recon_error = (recon_error > threshold_recon_error).astype(int)
    predicted_labels_matrix_profile = (test_matrix_profile > threshold_matrix_profile).astype(int)

    predicted_labels = predicted_labels_recon_error & predicted_labels_matrix_profile

    expanded_true_labels = expand_anomalies(test_labels.numpy(), window_size=window_size)
    expanded_predicted_labels = expand_anomalies(predicted_labels, window_size=window_size)

    accuracy = accuracy_score(expanded_true_labels, expanded_predicted_labels)
    precision = precision_score(expanded_true_labels, expanded_predicted_labels)
    recall = recall_score(expanded_true_labels, expanded_predicted_labels)
    f1 = f1_score(expanded_true_labels, expanded_predicted_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == '__main__':
    main()
