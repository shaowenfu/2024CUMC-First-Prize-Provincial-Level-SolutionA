import numpy as np
from scipy.optimize import fsolve

b = 1.7 / (2 * np.pi)
def equations(vars, theta_1, theta_2):
    R, OC1, OC2, ang_O = vars


    L2 = b ** 2 * ((theta_1 * np.cos(theta_1) - theta_2 * np.cos(theta_2)) ** 2 +
                   (theta_1 * np.sin(theta_1) - theta_2 * np.sin(theta_2)) ** 2)


    T_1 = np.array([b * np.cos(theta_1) - b * theta_1 * np.sin(theta_1),
                    b * np.sin(theta_1) + b * theta_1 * np.cos(theta_1)])
    T_2 = np.array([b * np.cos(theta_2) - b * theta_2 * np.sin(theta_2),
                    b * np.sin(theta_2) + b * theta_2 * np.cos(theta_2)])

    M1M2 = np.array([b * (theta_2 * np.cos(theta_2) - theta_1 * np.cos(theta_1)),
                     b * (theta_2 * np.sin(theta_2) - theta_1 * np.sin(theta_1))])


    alpha_1 = np.arcsin(np.dot(M1M2, T_1) / (np.linalg.norm(M1M2) * np.linalg.norm(T_1)))
    alpha_2 = np.arcsin(np.dot(M1M2, T_2) / (np.linalg.norm(M1M2) * np.linalg.norm(T_2)))


    eq1 = np.cos(ang_O) - ((OC1 + 2 * R) ** 2 + (OC2 + R) ** 2 - L2) / (2 * (OC1 + 2 * R) * (OC2 + R))
    eq2 = np.sin(alpha_1) / np.sin(alpha_2) - (OC1 + 2 * R) / (OC2 + R)
    eq3 = np.arcsin((np.sin(ang_O) * OC2) / (3 * R)) - alpha_1
    eq4 = np.arcsin((np.sin(ang_O) * OC1) / (3 * R)) - alpha_2

    return [eq1, eq2, eq3, eq4]


def solve_beta_R(theta_1, theta_2):

    initial_guess = [1, 1, 1, np.pi / 4]

    solution = fsolve(equations, initial_guess, args=(theta_1, theta_2), xtol=1e-8, maxfev=1000)

    R = solution[0]
    beta_1 = np.arcsin((np.sin(solution[3]) * solution[2]) / (3 * R))
    beta_2 = np.arcsin((np.sin(solution[3]) * solution[1]) / (3 * R))
    return beta_1, beta_2, R, solution[1], solution[2], solution[3]


def objective_2(x):
    theta_1, theta_2 = x
    beta_1, beta_2, R, OC1, OC2, ang_O = solve_beta_R(theta_1, theta_2)

    obj = (np.pi - beta_1) * 2 * R + (np.pi + beta_2) * R
    return obj
def calculate_curvature(points):
    """计算路径上每一点的曲率"""
    curvatures = []
    for i in range(1, len(points) - 1):
        p0, p1, p2 = points[i-1], points[i], points[i+1]

        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        curvature = 4 * area / (a * b * c)
        curvatures.append(curvature)
    return curvatures

def calculate_max_head_speed(max_body_speed, path_points):
    """计算龙头的最大速度"""
    curvatures = calculate_curvature(np.array(path_points))
    max_curvature = max(curvatures)
    max_head_speed = max_body_speed / max_curvature
    return max_head_speed


path_points = [
    np.array([0, 0]),
    np.array([1, 0]),
    np.array([2, 1]),
    np.array([3, 1]),
    np.array([4, 0])
]

max_body_speed = 2


max_head_speed = calculate_max_head_speed(max_body_speed, path_points)
print(f"龙头的最大行进速度：{max_head_speed:.2f} m/s")
