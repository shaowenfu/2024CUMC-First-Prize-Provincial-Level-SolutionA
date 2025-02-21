import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
from calculate_head_p import calculate_position  # 导入计算位置的函数
from matplotlib.animation import FuncAnimation
from calculate_v import calculate_v2  # 导入计算速度的函数
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # 忽略运行时警告
matplotlib.use('TkAgg')  # 使用 TkAgg 后端绘图

# 参数设置
d = 2.86  # 常量 d
a = 0  # 常量 a
b = 0.55 / (2 * np.pi)  # 常量 b


# 定义隐式函数
def implicit_fun_2024_A_1(theta_2, d, a, b, theta_1):
    """
    隐式函数用于计算给定参数下的方程值
    :param theta_2: 角度 theta_2
    :param d: 常量 d
    :param a: 常量 a
    :param b: 常量 b
    :param theta_1: 角度 theta_1
    :return: 方程值
    """
    return 2 * (a ** 2) + 2 * a * b * (theta_1 + theta_2) + (b ** 2) * (theta_1 ** 2 + theta_2 ** 2) - \
           2 * (a + b * theta_1) * (a + b * theta_2) * np.cos(theta_2 - theta_1) - d ** 2


def calculate_next_position(theta_1, d, a, b):
    """
    计算下一个位置的角度 theta_2
    :param theta_1: 当前角度 theta_1
    :param d: 常量 d
    :param a: 常量 a
    :param b: 常量 b
    :return: 计算出的角度 theta_2
    """
    initial_guess = theta_1 + 1  # 初始猜测 theta_2
    # 使用 fsolve 求解隐式方程
    theta_2_solution = fsolve(lambda theta_2: implicit_fun_2024_A_1(theta_2, d, a, b, theta_1), initial_guess)
    return theta_2_solution[0]


# 定义用于绘制阿基米德螺线的 theta 范围
theta_range = np.linspace(0, 40 * np.pi, 1000)  # 从 0 到 40*pi，共 1000 个点
# 计算对应的 r 值
r_values = a + b * theta_range

# 创建图表
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})  # 创建极坐标图
ax.plot(theta_range, r_values, color=(47 / 255, 127 / 255, 193 / 255), label='Spiral')  # 绘制阿基米德螺线
points, = ax.plot([], [], 'o', color=(174 / 255, 32 / 255, 18 / 255), label='Points on Spiral')  # 点的图例
lines, = ax.plot([], [], 'g-', color=(238 / 255, 155 / 255, 000 / 255), label='Connecting Lines')  # 连接线的图例
first_point, = ax.plot([], [], 'o', color=(0, 0, 1), markersize=8, label='First Point')  # 第一个点的图例
first_line, = ax.plot([], [], '-', color=(254 / 255, 220 / 255, 94 / 255), linewidth=4, label='First Line')  # 第一条线的图例
ax.legend()  # 显示图例

# 创建用于保存数据的 DataFrame
result_table = pd.DataFrame()
result_table_xy = pd.DataFrame()
result_velocity = pd.DataFrame()


# 更新螺线上的点和连接线的函数
def update(t):
    """
    更新图表中的点和线
    :param t: 当前时间帧
    :return: 更新后的点和线
    """
    global result_table  # 使用全局变量
    global result_table_xy
    global result_velocity
    theta_list = []  # 存储角度的列表
    figure_list = []  # 存储角度的列表
    xy_list = []  # 存储坐标的列表
    velocity = []  # 存储速度的列表

    # 计算龙头前把手的位置
    theta_head = calculate_position(t, a, b)
    # 更新点的位置
    r_pos_values_t = b * np.array(theta_head)
    first_point.set_data([theta_head], [r_pos_values_t])  # 设置第一个点的位置

    # 龙头的速度
    v1 = 1
    velocity.append(1)

    # 龙头的位置
    x_values = b * (theta_head) * np.cos(theta_head)
    y_values = b * (theta_head) * np.sin(theta_head)
    xy_list.append(x_values)
    xy_list.append(y_values)

    # 计算龙头后把手的位置
    theta_next = calculate_next_position(theta_head, 2.86, 0, b)
    theta_list.append(theta_head)
    theta_list.append(theta_next)
    figure_list.append(theta_next)
    x_values = b * (theta_next) * np.cos(theta_next)
    y_values = b * (theta_next) * np.sin(theta_next)
    xy_list.append(x_values)
    xy_list.append(y_values)
    first_line.set_data([theta_head, theta_next], [r_pos_values_t, b * np.array(theta_next)])  # 设置第一条线的位置

    # 龙头后把手的速度
    v2 = calculate_v2(b, theta_head, theta_next, v1)
    velocity.append(v2)

    num = 2
    # 计算所有板凳的位置和速度
    while num < 224:
        num += 1
        theta_temp = theta_next
        theta_next = calculate_next_position(theta_next, 1.65, 0, b)
        v2 = calculate_v2(b, theta_temp, theta_next, v2)
        velocity.append(v2)
        theta_list.append(theta_next)
        x_values = b * (theta_next) * np.cos(theta_next)
        y_values = b * (theta_next) * np.sin(theta_next)
        xy_list.append(x_values)
        xy_list.append(y_values)
        if theta_next < 32 * np.pi:
            figure_list.append(theta_next)

    # 更新点的位置
    r_pos_values = b * np.array(figure_list)
    points.set_data(figure_list, r_pos_values)  # 设置点的位置

    # 更新连接点的线
    lines.set_data(figure_list, r_pos_values)  # 设置连接线的位置

    # 动态更新标题
    ax.set_title(f'Spiral with WoodBoards (t={t}s)', fontsize=12)

    # 调整径向标签的位置
    ax.set_rlabel_position(-22.5)

    # 更新图例
    ax.legend()

    # 保存数据到 DataFrame
    df = pd.DataFrame([theta_list], columns=[f'theta_{i}' for i in range(len(theta_list))])
    df_xy = pd.DataFrame([xy_list], columns=[f'{i}' for i in range(len(xy_list))])
    df_velocity = pd.DataFrame([velocity], columns=[f'velocity{i + 1}' for i in range(len(velocity))])
    result_table = pd.concat([result_table, df], ignore_index=True)
    result_table_xy = pd.concat([result_table_xy, df_xy], ignore_index=True)
    result_velocity = pd.concat([result_velocity, df_velocity], ignore_index=True)

    # 如果到达最后一帧，保存数据为 CSV 文件
    if t == 399:
        result_table.to_csv("spiral_positions2.csv", index=False)
        result_table_xy.to_csv("cordinate_positions2.csv", index=False)
        result_velocity.to_csv("velocity2.csv", index=False)
        print("Data saved to spiral_positions.csv")

    return points, lines, first_point, first_line


# 创建动画
ani = FuncAnimation(fig, update, frames=np.arange(0, 500), interval=100, blit=True, repeat=False)
# points, lines = update(300)
# plt.title('Spiral with handle position points and Connecting Lines(300s)')
from matplotlib.animation import PillowWriter

ani.save("E:\\all_workspace\\ML\\PyCharm1\\mathematic_A\\results\\problem1_animation.gif", writer=PillowWriter(fps=10))  # fps 是帧率

plt.show()
