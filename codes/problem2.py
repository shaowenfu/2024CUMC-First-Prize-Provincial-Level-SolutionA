import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from calculate_head_p import calculate_position
from calculate_next_p import calculate_next_position
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # 忽略运行时警告
matplotlib.use('TkAgg')  # 使用 TkAgg 后端绘图

def check_overlap_AABB(board1, board2):
    """
    检查两个木板是否在轴对齐包围盒（AABB）范围内重叠
    :param board1: 第一个木板对象
    :param board2: 第二个木板对象
    :return: 如果重叠则返回 True，否则返回 False
    """
    x1_min, x1_max, y1_min, y1_max = board1.get_bounding_box()
    x2_min, x2_max, y2_min, y2_max = board2.get_bounding_box()

    if x1_max < x2_min or x2_max < x1_min:
        return False  # x 轴上没有重叠
    if y1_max < y2_min or y2_max < y1_min:
        return False  # y 轴上没有重叠
    return True  # 存在重叠

def project_polygon(vertices, axis):
    """
    计算多边形在给定轴上的投影范围
    :param vertices: 多边形的顶点坐标
    :param axis: 投影轴
    :return: 投影的最小值和最大值
    """
    projections = np.dot(vertices, axis)
    return np.min(projections), np.max(projections)

def check_overlap_SAT(board1, board2):
    """
    使用分离轴定理（SAT）检查两个木板是否重叠
    :param board1: 第一个木板对象
    :param board2: 第二个木板对象
    :return: 如果重叠则返回 True，否则返回 False
    """
    vertices1 = board1.vertices
    vertices2 = board2.vertices

    edges1 = np.diff(vertices1, axis=0, append=vertices1[0:1])
    edges2 = np.diff(vertices2, axis=0, append=vertices2[0:1])

    for edge in np.concatenate((edges1, edges2)):
        axis = np.array([-edge[1], edge[0]])  # 获取边的法线作为分离轴
        axis /= np.linalg.norm(axis)  # 单位化分离轴

        min1, max1 = project_polygon(vertices1, axis)
        min2, max2 = project_polygon(vertices2, axis)

        if max1 < min2 or max2 < min1:
            return False  # 如果在某个轴上没有重叠，木板不重叠

    return True  # 在所有轴上都存在重叠

class WoodBoard:
    def __init__(self, theta1, theta2, b):
        """
        初始化木板对象
        :param theta1: 第一个孔的角度
        :param theta2: 第二个孔的角度
        :param b: 螺距相关常量
        """
        self.width = 0.3  # 木板的宽度
        self.dist = 0.275  # 木板上的孔之间的距离
        self.theta1 = theta1
        self.theta2 = theta2

        # 将极坐标转换为笛卡尔坐标以用于绘图
        r1 = b * theta1
        r2 = b * theta2
        self.hole1 = np.array([r1 * np.cos(theta1), r1 * np.sin(theta1)])  # 第一个孔的位置
        self.hole2 = np.array([r2 * np.cos(theta2), r2 * np.sin(theta2)])  # 第二个孔的位置

        # 计算木板的顶点
        self.vertices = self.get_vertices()

    def get_vertices(self):
        """
        计算木板的四个顶点
        :return: 木板的顶点坐标
        """
        # 计算孔之间的方向向量
        direction = self.hole1 - self.hole2
        unit_direction = direction / np.linalg.norm(direction)

        # 计算垂直方向的单位向量
        perp_direction = np.array([-unit_direction[1], unit_direction[0]])
        unit_perp_direction = perp_direction / np.linalg.norm(unit_direction)

        # 计算木板的四个顶点
        top_left = self.hole1 + unit_perp_direction * (self.width / 2) + unit_direction * self.dist
        top_right = self.hole2 + unit_perp_direction * (self.width / 2) - unit_direction * self.dist
        bottom_left = self.hole1 - unit_perp_direction * (self.width / 2) + unit_direction * self.dist
        bottom_right = self.hole2 - unit_perp_direction * (self.width / 2) - unit_direction * self.dist

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def get_bounding_box(self):
        """
        获取木板的轴对齐包围盒（AABB）
        :return: 木板的 x 和 y 轴的最小值和最大值
        """
        x_min = np.min(self.vertices[:, 0])
        x_max = np.max(self.vertices[:, 0])
        y_min = np.min(self.vertices[:, 1])
        y_max = np.max(self.vertices[:, 1])
        return x_min, x_max, y_min, y_max

def plot_woodboard(ax, board, label, color, show_legend):
    """
    在图上绘制木板
    :param ax: matplotlib 的轴对象
    :param board: 木板对象
    :param label: 图例标签
    :param color: 木板的颜色
    :param show_legend: 是否显示图例
    """
    vertices = board.vertices
    vertices = np.vstack([vertices, vertices[0]])  # 关闭多边形
    if color == '1':
        ax.plot(vertices[:, 0], vertices[:, 1], color='orange', label=label if show_legend else None)
    else:
        ax.plot(vertices[:, 0], vertices[:, 1], color=(254/255, 220/255, 94/255), label=label if show_legend else None)
    # 绘制孔的位置
    ax.plot(board.hole1[0], board.hole1[1], 'ko', markersize=3)  # 黑色圆点，尺寸为3
    ax.plot(board.hole2[0], board.hole2[1], 'ko', markersize=3)

def update(t):
    """
    更新动画的每一帧
    :param t: 当前时间
    """
    global theta_next, board_list, anim

    # 清除之前的图形
    ax.clear()

    # 更新螺旋图
    ax.plot(r_values * np.cos(theta_range), r_values * np.sin(theta_range), color=(47/255, 127/255, 193/255), label='Spiral')

    # 重置木板列表
    board_list = []

    # 重新计算木板的位置
    theta_head = calculate_position(t, a, b)
    theta_next = calculate_next_position(theta_head, 2.86, 0, b)
    board1 = WoodBoard(theta_head, theta_next, b=b)
    board_list.append(board1)

    num = 1
    while num < 224:
        num += 1
        theta_temp = theta_next
        theta_next = calculate_next_position(theta_temp, 1.65, 0, b)
        board2 = WoodBoard(theta_temp, theta_next, b=b)
        board_list.append(board2)

    # 绘制第一个木板（龙头）为红色，并显示图例
    plot_woodboard(ax, board_list[0], f'head of loong', '1', True)
    plot_woodboard(ax, board_list[1], f'body of loong', 'dsa', True)

    # 绘制其余的木板为橙色，不显示图例
    for i in range(2, len(board_list) - 1):
        plot_woodboard(ax, board_list[i], f'Wood_Board {i + 1}', 'orange', False)

    # 绘制最后一个木板（龙尾）为绿色，并显示图例
    plot_woodboard(ax, board_list[-1], f'tail of loong', 'red', True)

    # 设置坐标轴标签和标题
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title(f'Spiral and Wood Board position in (t={t}s)')
    ax.legend()
    plt.grid(True)

    # 检查木板重叠情况
    if t > 200:
        for j in range(20):
            for i in range(j + 2, min(len(board_list), 24)):  # 检查接下来的木板
                if not check_overlap_SAT(board_list[j], board_list[i]):
                    continue
                else:
                    if j == 0:
                        print(f"木板{j+1}(龙头)在t={t}s时与第{i + 1}个木板相撞")
                    else:
                        print(f"木板{j + 1}在t={t}s时与第{i + 1}个木板相撞")
                    sys.exit(0)  # 碰撞后退出程序
                    break

if __name__ == "__main__":
    a = 0
    b = 0.55 / (2 * np.pi)
    theta_range = np.linspace(0, 40 * np.pi, 1000)
    r_values = a + b * theta_range

    # 初始化绘图
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    # 初始化变量
    theta_head = 0
    theta_next = 0
    board_list = []

    # 搜索时间的区间
    t = [400, 420]
    # 搜索的时间精确度（秒）
    search_time_gap = 1
    # 创建动画，每1000毫秒（1秒）更新一次
    ani = FuncAnimation(fig, update, frames=np.arange(t[0], t[1], search_time_gap), interval=100)

    from matplotlib.animation import PillowWriter

    ani.save("E:\\all_workspace\\ML\\PyCharm1\\mathematic_A\\results\\problem2_animation.gif", writer=PillowWriter(fps=10))  # fps 是帧率
    # 显示动画
    plt.show()
