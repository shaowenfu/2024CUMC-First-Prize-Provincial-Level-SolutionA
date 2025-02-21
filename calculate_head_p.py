# 计算任意时刻龙头前把手的位置
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
matplotlib.use('TkAgg')  # 使用 TkAgg 后端绘图

# 定义常量
p = 0.55  # 螺距 55 cm = 0.55 m
speed = 1.0  # 速度 1 m/s
theta0 = 16 * 2 * np.pi  # 初始第16圈的角度
a = 0
b = p / (2 * np.pi)  # b 是通过螺距 p 和 2π 计算的

def arc_length_integrand(theta, a, b):
    """
    计算给定 theta 下的弧长积分函数
    :param theta: 当前角度
    :param a: 常量 a
    :param b: 常量 b
    :return: 弧长积分函数值
    """
    return np.sqrt(b**2 + (a + b * theta)**2)

def find_theta2(L, theta1, a, b):
    """
    通过给定的弧长 L 和初始角度 theta1 计算对应的角度 theta2
    :param L: 走过的弧长
    :param theta1: 初始角度
    :param a: 常量 a
    :param b: 常量 b
    :return: 计算出的角度 theta2
    """
    # 定义需要求解的方程
    def equation(theta2):
        # 计算 theta1 和 theta2 之间的弧长
        integral, _ = quad(arc_length_integrand, theta2, theta1, args=(a, b))
        # 计算计算得到的弧长与给定弧长 L 的差值
        return integral - L

    # theta2 的初始猜测值
    initial_guess = theta1
    # 求解 theta2
    theta2_solution, = fsolve(equation, initial_guess)
    return theta2_solution

def calculate_position(t , a, b):
    """
    计算在时间 t 后龙头把手的角度
    :param t: 时间
    :param a: 常量 a
    :param b: 常量 b
    :return: 计算出的角度 theta2
    """
    L = speed * t  # t 时间后走过的弧长
    theta1 = theta0  # 初始角度
    theta2 = find_theta2(L, theta1, a, b)
    return theta2

if __name__ == "__main__":
    a = 0
    b = 0.55 / (2 * np.pi)
    position_list = []  # 存储每个时间步的角度

    # 计算 0 到 399 时间步下的角度
    for i in range(400):
        position = calculate_position(i, a, b)
        position_list.append(position)

    # 定义用于绘制阿基米德螺线的 theta 范围
    theta_range = np.linspace(0, 40 * np.pi, 1000)

    # 计算对应的 r 值
    r_values = b * theta_range

    # 创建图表并绘制阿基米德螺线
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_range, r_values, 'b', label='Archimedean Spiral')  # 绘制阿基米德螺线

    # 提取 position_list 中的角度和半径值
    theta_values = position_list  # 角度值
    r_pos_values = b * np.array(position_list)  # 半径值
    # 在图中绘制 position_list 中的点
    ax.scatter(theta_values, r_pos_values, color='red', label='Points on Spiral')  # 绘制点

    # 显示图形
    plt.legend()  # 显示图例
    plt.title('Archimedean Spiral with Dynamic Points and Connecting Line')  # 设置标题
    plt.show()  # 展示图形
