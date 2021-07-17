import numpy as np
import matplotlib.pyplot as plt


def pcd(q1, q2, l1, l2):
    x1 = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y1 = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

    return x1.round(5), y1.round(5)


def dibujar_trayectoria_pcd(q1s, q2s, l1, l2):
    for i in range(q1s.size):
        x, y = pcd(q1s[i], q2s[i], l1, l2)
        label = 'q1: ' + str(np.rad2deg(q1s[i]).round(3)) + \
                ', q2: ' + str(np.rad2deg(q2s[i]).round(3))
        plt.plot(x, y, 'o', label=label)
        
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_aspect('equal')
    fig.set_size_inches(10, 10)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim([- (l1 + l2 + 1), l1 + l2 + 1])
    plt.xlim([- (l1 + l2 + 1), l1 + l2 + 1])
    plt.legend()
    dibujar_robot(q1s[q1s.size - 1], q2s[q2s.size - 1], l1, l2)
    plt.show()


def dibujar_robot(q1, q2, l1, l2):
    x0, y0 = 0, 0
    x1, y1 = pcd(q1, 0, l1, 0)
    x2, y2 = pcd(q1, q2, l1, l2)
    x, y = [x0, x1, x2], [y0, y1, y2]
    plt.plot(x, y, 'k')
    plt.plot(x1, y1, 'k.')
    plt.plot(x0, y0, 'b.')


Vq1 = np.array([np.deg2rad(120), np.deg2rad(90),
                np.deg2rad(60), np.deg2rad(30), np.deg2rad(15)])
Vq2 = np.array([np.deg2rad(90), np.deg2rad(60),
                np.deg2rad(-10), np.deg2rad(30), np.deg2rad(15)])

dibujar_trayectoria_pcd(Vq1, Vq2, 1, 1)
