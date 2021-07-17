import numpy as np


def pcd(q1, q2, l1, l2):
    x1 = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y1 = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

    return x1.round(5), y1.round(5)


x, y = pcd(np.deg2rad(90), np.deg2rad(-90), 2, 1)

print('x:', x, 'y:', y)
