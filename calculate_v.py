import numpy as np


def calculate_v2(b, theta1, theta2, v1):
    x1 = b * (theta1) * np.cos(theta1)  # Example coordinates
    y1 = b * (theta1) * np.sin(theta1)
    x2 = b * (theta2) * np.cos(theta2)  # Example coordinates
    y2 = b * (theta2) * np.sin(theta2)
    # Calculate the direction vector v0
    v0 = np.array([x2 - x1, y2 - y1])

    # Compute the magnitudes of v0
    v0_norm = np.linalg.norm(v0)

    # Define the components of v1 and v2 based on the given formulas
    m1 = np.array([b * np.cos(theta1) - b * theta1 * np.sin(theta1),
                   b * np.sin(theta1) + b * theta1 * np.cos(theta1)])

    m2 = np.array([b * np.cos(theta2) - b * theta2 * np.sin(theta2),
                   b * np.sin(theta2) + b * theta2 * np.cos(theta2)])

    m1_norm = np.linalg.norm(m1)
    m2_norm = np.linalg.norm(m2)

    # Calculate cos(alpha1) and cos(alpha2)
    cos_alpha1 = np.dot(v0, m1) / (v0_norm * m1_norm)
    cos_alpha2 = np.dot(v0, m2) / (v0_norm * m2_norm)

    v2 = v1 * (cos_alpha1 / cos_alpha2)
    v2_norm = np.linalg.norm(v2)

    return v2_norm


if __name__ == "__main__":
    # Example usage
    # 参数设置
    d = 2.86
    a = 0
    b = 0.55 / (2 * np.pi)
    theta1 = np.pi / 4  # Example angle in radians
    theta2 = np.pi / 6  # Example angle in radians
