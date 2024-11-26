
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 数据
models = [
    'THOC', 'OC-SVM', 'IForest', 'LOF', 'ARIMA',
    'LSTM-VAE', 'LSTM-VAE(p)', 'OmniAnomaly', 'BeatGAN',
    'ADTransformer', 'USAD', 'GDN', 'InterFusion', 'Deep-SVDD',
    'Our Model'
]
precision = [52.30, 41.14, 40.77, 41.47, 13.59,
             65.73, 62.08, 64.21, 45.20, 71.35,
             23.75, 32.46, 60.74, 47.08, 81.03]
recall = [82.95, 90.04, 93.60, 98.80, 85.71,
          89.45, 95.54, 86.93, 88.42, 97.60,
          95.60, 98.60, 95.20, 88.91, 94.30]
f1_score = [64.33, 57.23, 56.07, 58.42, 39.75,
             75.73, 75.89, 73.86, 59.82, 82.37,
             47.86, 46.59, 74.16, 61.56, 86.79]

# 设置柱状图的位置和宽度
x = np.arange(len(models))  # 模型数量
width = 0.25  # 柱宽

fig, ax = plt.subplots(figsize=(14, 8))

# 绘制每个指标的柱状图
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# 添加文本标签、标题和自定义 x 轴标签
ax.set_xlabel('Model')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Models on Precision, Recall, and F1-Score')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=90)

# 设置图例
legend = ax.legend(frameon=True, loc='upper left', fontsize='small')  # 缩小字体

# 调整图例边框的大小和样式
legend.get_frame().set_edgecolor('black')  # 设置边框颜色
legend.get_frame().set_linewidth(0.5)      # 设置边框宽度

# 添加柱子上的标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

fig.tight_layout()

plt.show()

colors = sns.color_palette("dark", len(models))
