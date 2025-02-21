import numpy as np
from scipy.optimize import fsolve, minimize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 常量 b
b = 1.7 / (2 * np.pi)

# 定义用于求解 beta_1, beta_2 和 R 的超越方程
def equations(vars, theta_1, theta_2):
    R, OC1, OC2, ang_O = vars

    # L 的平方
    L2 = b ** 2 * ((theta_1 * np.cos(theta_1) - theta_2 * np.cos(theta_2)) ** 2 +
                   (theta_1 * np.sin(theta_1) - theta_2 * np.sin(theta_2)) ** 2)

    # 位置向量
    T_1 = np.array([b * np.cos(theta_1) - b * theta_1 * np.sin(theta_1),
                    b * np.sin(theta_1) + b * theta_1 * np.cos(theta_1)])
    T_2 = np.array([b * np.cos(theta_2) - b * theta_2 * np.sin(theta_2),
                    b * np.sin(theta_2) + b * theta_2 * np.cos(theta_2)])

    M1M2 = np.array([b * (theta_2 * np.cos(theta_2) - theta_1 * np.cos(theta_1)),
                     b * (theta_2 * np.sin(theta_2) - theta_1 * np.sin(theta_1))])

    # 角度 alpha_1 和 alpha_2
    alpha_1 = np.arcsin(np.dot(M1M2, T_1) / (np.linalg.norm(M1M2) * np.linalg.norm(T_1)))
    alpha_2 = np.arcsin(np.dot(M1M2, T_2) / (np.linalg.norm(M1M2) * np.linalg.norm(T_2)))

    # 定义非线性方程
    eq1 = np.cos(ang_O) - ((OC1 + 2 * R) ** 2 + (OC2 + R) ** 2 - L2) / (2 * (OC1 + 2 * R) * (OC2 + R))
    eq2 = np.sin(alpha_1) / np.sin(alpha_2) - (OC1 + 2 * R) / (OC2 + R)
    eq3 = np.arcsin((np.sin(ang_O) * OC2) / (3 * R)) - alpha_1
    eq4 = np.arcsin((np.sin(ang_O) * OC1) / (3 * R)) - alpha_2

    return [eq1, eq2, eq3, eq4]

# 给定 theta_1 和 theta_2，求解 R, OC1, OC2, 和 ang_O
def solve_beta_R(theta_1, theta_2):
    # 初始猜测值
    initial_guess = [1, 1, 1, np.pi / 4]
    # 调整容忍度和最大迭代次数
    solution = fsolve(equations, initial_guess, args=(theta_1, theta_2), xtol=1e-8, maxfev=1000)

    R = solution[0]
    beta_1 = np.arcsin((np.sin(solution[3]) * solution[2]) / (3 * R))
    beta_2 = np.arcsin((np.sin(solution[3]) * solution[1]) / (3 * R))
    return beta_1, beta_2, R, solution[1], solution[2], solution[3]  # 返回 OC1, OC2 和 ang_O

# 定义目标函数
def objective_2(x):
    theta_1, theta_2 = x
    beta_1, beta_2, R, OC1, OC2, ang_O = solve_beta_R(theta_1, theta_2)
    # 目标函数
    obj = (np.pi - beta_1) * 2 * R + (np.pi + beta_2) * R
    return obj

# 优化 theta_1 和 theta_2
initial_guess = [10*np.pi/1.7, -12*np.pi/1.7]  # 初始猜测 theta_1 和 theta_2

# 定义 theta_1 和 theta_2 的边界
bounds = [(0, 9 * np.pi / 1.7), (-9 * np.pi / 1.7, 0)]

# 使用 L-BFGS-B 算法进行优化，带有边界约束
result = minimize(objective_2, initial_guess, method='L-BFGS-B', bounds=bounds)

# 打印优化后的 theta_1, theta_2 和对应的 R
optimal_theta_1, optimal_theta_2 = result.x
beta_1, beta_2, optimal_R, OC1, OC2, ang_O = solve_beta_R(optimal_theta_1, optimal_theta_2)

# 计算切线向量 T_1 和 T_2
T_1 = np.array([b * np.cos(optimal_theta_1) - b * optimal_theta_1 * np.sin(optimal_theta_1),
                b * np.sin(optimal_theta_1) + b * optimal_theta_1 * np.cos(optimal_theta_1)])
T_2 = np.array([b * np.cos(optimal_theta_2) - b * optimal_theta_2 * np.sin(optimal_theta_2),
                b * np.sin(optimal_theta_2) + b * optimal_theta_2 * np.cos(optimal_theta_2)])

# 计算方向向量
M1 = np.array([b * (optimal_theta_2 * np.cos(optimal_theta_2) - optimal_theta_1 * np.cos(optimal_theta_1)),
               b * (optimal_theta_2 * np.sin(optimal_theta_2) - optimal_theta_1 * np.sin(optimal_theta_1))])
M2 = np.array([b * (optimal_theta_2 * np.cos(optimal_theta_2) - optimal_theta_1 * np.cos(optimal_theta_1)),
               b * (optimal_theta_2 * np.sin(optimal_theta_2) - optimal_theta_1 * np.sin(optimal_theta_1))])

direction_M1_C1 = np.array([OC1 - M1[0], OC1 - M1[1]])
direction_M2_C2 = np.array([OC2 - M2[0], OC2 - M2[1]])

print(f"优化后的 theta_1: {optimal_theta_1}")
print(f"优化后的 theta_2: {optimal_theta_2}")
print(f"优化后的 R: {optimal_R}")
print(f"T_1: {T_1}")
print(f"T_2: {T_2}")
print(f"方向向量 M1 到圆心 C1: {direction_M1_C1}")
print(f"方向向量 M2 到圆心 C2: {direction_M2_C2}")


theta_1 = optimal_theta_1
theta_2 = optimal_theta_2

x_M1 = b * (theta_1) * np.cos(theta_1)
y_M1 = b * (theta_1) * np.sin(theta_1)
x_M2 = b * (theta_2) * np.cos(theta_2)
y_M2 = b * (theta_2) * np.sin(theta_2)


from circle_equation import *

# 示例输入
M1 = (x_M1, y_M1)
M2 = (x_M2, y_M2)
R1 = 2*optimal_R
R2 = optimal_R
dir_C1M1 = -direction_M1_C1  # 向量 C1M1 的方向
dir_C2M2 = -direction_M2_C2  # 向量 C2M2 的方向
T1 = T_1  # 切线方向 T1
T2 = T_2 # 切线方向 T2

