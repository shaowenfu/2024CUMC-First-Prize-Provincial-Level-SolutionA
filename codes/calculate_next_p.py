import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from calculate_head_p import calculate_position
matplotlib.use('TkAgg')  # 或 'Qt5Agg' 或 'Agg'
# Set parameters
d = 2.86
a = 0
b = 0.55 / (2 * np.pi)
# Define the implicit function
def implicit_fun_2024_A_1(theta_2, d, a, b, theta_1):
    return 2*(a**2) + 2*a*b*(theta_1 + theta_2) + (b**2) * (theta_1**2 + theta_2**2) - \
           2*(a + b*theta_1)*(a + b*theta_2)*np.cos(theta_2 - theta_1) - d**2

def calculate_next_position(theta_1,d,a,b):
    # Initial guess for theta_2
    initial_guess = theta_1+1

    # Solve using fsolve
    theta_2_solution = fsolve(lambda theta_2: implicit_fun_2024_A_1(theta_2, d, a, b, theta_1), initial_guess)
    return theta_2_solution[0]

if __name__ == "__main__":

    t = 1
    theta_list = []
    # 先计算龙头前把手的位置
    theta_head = calculate_position(t, a, b)
    # 计算龙头后把手的位置
    theta_next = calculate_next_position(theta_head, 2.86, 0, b)
    theta_list.append(theta_head)
    theta_list.append(theta_next)
    # 计算所有板凳的位置
    while (theta_next < 32 * np.pi):
        theta_next = calculate_next_position(theta_next, 1.65, 0, b)
        theta_list.append(theta_next)

    # Define theta range for plotting the Archimedean spiral
    theta_range = np.linspace(0, 40 * np.pi, 1000)

    # Compute the corresponding r values
    r_values = a + b * theta_range

    # Plot the Archimedean spiral
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta_range, r_values, 'b', label='Archimedean Spiral')

    # 提取 position_list 中的角度和半径值
    theta_values = [pos for pos in theta_list]
    r_pos_values = b * np.array(theta_values)
    # 在图中绘制 position_list 中的点
    ax.scatter(theta_values, r_pos_values, color='red', label='Points on Spiral')

    # Add legend and title
    ax.legend()
    plt.title('Archimedean Spiral with Points')
    plt.show()



