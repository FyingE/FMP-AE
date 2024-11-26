# FMP-AE: Unsupervised Anomaly Detection in Time Series using Feature Map Matrix Profile and Autoencoder

## Overview

**FMP-AE** (Feature map Matrix Profile with an AutoEncoder) is a hybrid method for unsupervised anomaly detection in univariate time series data. It combines the traditional **Matrix Profile (MP)** technique with deep learning methods to achieve improved anomaly detection accuracy, robustness, and computational efficiency. The model uses a **1D Convolutional Neural Network (1D-CNN)** for feature extraction and computes the **Matrix Profile** for identifying anomalous subsequences in the time series.

This method is specifically designed to handle the challenges of **label scarcity**, **data imbalance**, and **computational inefficiency** that are common in time series anomaly detection tasks.

## Key Features

- **Hybrid Model**: Combines Matrix Profile techniques with deep learning (1D-CNN and Autoencoder) for enhanced anomaly detection.
- **Sliding Window Technique**: Improves sensitivity to sparse anomalies by processing smaller subsequences.
- **Matrix Profile Loss**: A novel loss function that integrates Matrix Profile loss with the Autoencoder's reconstruction loss, improving anomaly detection accuracy.
- **Efficient and Scalable**: Optimized for large-scale time series data, improving both computational efficiency and model generalization across domains.

## Methodology

### 1. **Data Preprocessing**

The time series data is first preprocessed using a **sliding window technique**, where each window represents a subsequence of fixed length. This approach addresses the issue of **data imbalance** and **label scarcity** by independently analyzing each subsequence for potential anomalies. If any point in a window is identified as anomalous, the entire window is marked as anomalous, which helps capture sparse anomalies.

### 2. **Feature Extraction using 1D-CNN**

Each subsequence is passed through a **1D-CNN** to extract local features. The CNN architecture consists of:

- Three convolutional layers with **Batch Normalization** and **ReLU activation**.
- **Max Pooling** layers to reduce computational complexity and highlight prominent features.

### 3. **Matrix Profile Computation**

After feature extraction, the model computes the **Matrix Profile** by:

- Calculating the **Euclidean distances** between the feature maps of different subsequences.
- Using these distances to create an **optimized distance profile**, which reflects the similarity between segments based on their feature representations.
- Computing the **Matrix Profile** by selecting the minimum distance for each subsequence, helping identify anomalous subsequences (those with larger distances from their most similar counterparts).

### 4. **Loss Function**

The model introduces a novel **Matrix Profile loss function**, which is combined with the **Autoencoder’s reconstruction loss**. The **Matrix Profile loss** focuses on capturing internal similarities between subsequences, while the **reconstruction loss** ensures the model can accurately reconstruct the input time series. This hybrid loss function enhances anomaly detection and improves the model’s ability to handle sparse anomalies.

### 5. **Training and Evaluation**

The model is trained using the combined loss function. We perform extensive experiments on benchmark datasets, including the **UCR250** dataset, to evaluate performance across multiple metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **AUC-ROC**.

## Experimental Results

The experimental results on the **UCR250 benchmark datasets** demonstrate the effectiveness of **FMP-AE** in comparison to traditional anomaly detection methods. Our model outperforms existing methods in terms of:

- **Precision**, **Recall**, **F1-score**, and **AUC-ROC**.
- **Computational efficiency** for large-scale datasets.
- **Generalization** across diverse time series domains.

## Installation

To use the **FMP-AE** model, you need to have the following dependencies:

- Python 3.x
- PyTorch (1.x or higher)
- NumPy
- SciPy
- Matplotlib (optional, for visualization)

