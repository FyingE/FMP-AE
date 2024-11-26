import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(0)
days = np.arange(1, 31)  # 用1到30表示日期
normal_values = np.random.normal(100, 10, size=len(days))
anomalous_group = np.random.normal(150, 10, size=5)  # 假设第15-19天为异常群组


# 合并数据，将异常群组插入到第15-19天
data = np.copy(normal_values)
data[15:20] = anomalous_group

# 绘制数据
plt.figure(figsize=(10, 5))
plt.plot(days, data, linestyle='-', color='r', label='Value')  # 将线条改为红色

# 标记异常群组
anomaly_dates = days[15:20]
plt.scatter(anomaly_dates, data[15:20], color='y', label='Anomaly Group')  # 将异常点标记为红色

# 标记异常区域，用红色背景色高亮
plt.axvspan(days[15], days[19], color='yellow', alpha=0.3)

# 添加标题和标签
plt.title('Time series with Group Anomaly')
plt.xlabel('Time series')
plt.ylabel('Value')
plt.grid(False)

# 添加图例
plt.legend()

# 显示图表
plt.show()
