import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def plot_circle(ax, center, radius, color='b'):
    circle = plt.Circle(center, radius, color=color, fill=False, linestyle='--', linewidth=1.5)
    ax.add_patch(circle)


def plot_trajectory(ax, trajectory):
    # 画圆C1
    plot_circle(ax, trajectory['entry_to_C1']['circle_C1']['center'],
                trajectory['entry_to_C1']['circle_C1']['radius'], 'b')
    # 画圆C2
    plot_circle(ax, trajectory['entry_to_C2']['circle_C2']['center'],
                trajectory['entry_to_C2']['circle_C2']['radius'], 'r')

    # 绘制轨迹
    entry_to_C1 = trajectory['entry_to_C1']
    entry_to_C2 = trajectory['entry_to_C2']
    exit_from_C2 = trajectory['exit_from_C2']

    # 从M1沿T1进入C1
    ax.plot([entry_to_C1['entry_point'][0], entry_to_C1['circle_C1']['tangent_point'][0]],
            [entry_to_C1['entry_point'][1], entry_to_C1['circle_C1']['tangent_point'][1]],
            'g--', label='M1 to C1 Tangent')

    # C1上到C1和C2切点
    # 注意：C1和C2的切点计算可以更复杂，示例中仅为直线段表示
    ax.plot([entry_to_C1['circle_C1']['tangent_point'][0], entry_to_C2['entry_point'][0]],
            [entry_to_C1['circle_C1']['tangent_point'][1], entry_to_C2['entry_point'][1]],
            'b-', label='C1 to C2 Path')

    # 从C2沿T2离开C2到M2
    ax.plot([entry_to_C2['entry_point'][0], exit_from_C2['exit_point'][0]],
            [entry_to_C2['entry_point'][1], exit_from_C2['exit_point'][1]],
            'r--', label='C2 to M2 Tangent')

    # 标注点
    ax.plot(*entry_to_C1['entry_point'], 'go', label='M1')
    ax.plot(*entry_to_C2['entry_point'], 'ro', label='C2 Entry')
    ax.plot(*exit_from_C2['exit_point'], 'mo', label='M2')
    ax.plot(*entry_to_C1['circle_C1']['tangent_point'], 'bo', label='C1-C2 Tangent Point')

    # 设置图形属性
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()
    plt.grid(True)
    plt.title('Trajectory Visualization')
    plt.show()


def calculate_circle_trajectory(M1, M2, R1, R2, dir_C1M1, dir_C2M2, T1, T2):
    # 提取点坐标和方向向量
    x1_m, y1_m = M1
    x2_m, y2_m = M2

    # 单位方向向量
    dir_C1M1 = np.array(dir_C1M1) / np.linalg.norm(dir_C1M1)
    dir_C2M2 = np.array(dir_C2M2) / np.linalg.norm(dir_C2M2)
    T1 = np.array(T1) / np.linalg.norm(T1)
    T2 = np.array(T2) / np.linalg.norm(T2)

    # 计算圆心坐标
    C1 = np.array([x1_m, y1_m]) + R1 * dir_C1M1
    C2 = np.array([x2_m, y2_m]) + R2 * dir_C2M2

    # 计算圆C1和C2的切点
    def compute_tangent_point(C1, C2, R1, R2):
        d = np.linalg.norm(C2 - C1)
        r1 = R1
        r2 = R2
        tangent_point = C1 + (r1 / d) * (C2 - C1)
        return tangent_point

    tangent_point = compute_tangent_point(C1, C2, R1, R2)

    # 描述轨迹
    trajectory_description = {
        "entry_to_C1": {
            "entry_point": M1,
            "entry_direction": T1,
            "circle_C1": {
                "center": C1,
                "radius": R1,
                "tangent_point": tangent_point
            }
        },
        "entry_to_C2": {
            "entry_point": tangent_point,
            "entry_direction": np.cross([0, 0, 1], [T1[0], T1[1], 0])[:2],  # 计算C1到C2的切线方向
            "circle_C2": {
                "center": C2,
                "radius": R2,
                "exit_point": M2
            }
        },
        "exit_from_C2": {
            "exit_point": M2,
            "exit_direction": T2
        }
    }

    return trajectory_description

if __name__ == "__main__":
    # 示例输入
    M1 = (1, 1)
    M2 = (6, 6)
    R1 = 5
    R2 = 3
    dir_C1M1 = (1, 1)  # 向量 C1M1 的方向
    dir_C2M2 = (-1, 1)  # 向量 C2M2 的方向
    T1 = (-1, 1)  # 切线方向 T1
    T2 = (1, 1)  # 切线方向 T2

    trajectory = calculate_circle_trajectory(M1, M2, R1, R2, dir_C1M1, dir_C2M2, T1, T2)

    # 绘制轨迹
    fig, ax = plt.subplots()
    plot_trajectory(ax, trajectory)
