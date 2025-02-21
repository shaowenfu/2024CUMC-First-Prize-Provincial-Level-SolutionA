import pandas as pd
import numpy as np
# 定义用于绘制阿基米德螺线的 theta 范围
theta_range = np.linspace(0, 40 * np.pi, 1000)
# 计算对应的 r 值
r_values = a + b * theta_range

# 创建图表
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_range, r_values, 'b', label='Archimedean Spiral')