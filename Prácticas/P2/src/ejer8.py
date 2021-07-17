import numpy as np
import matplotlib.pyplot as plt


def pcd(q1, q2, l1, l2):
    x1 = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y1 = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

    return x1.round(5), y1.round(5)


def pci(xf, yf, l1, l2):
    cosq2 = (xf ** 2 + yf ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    sinq2 = np.sqrt(1 - cosq2 ** 2)
    q2 = np.arctan2(sinq2, cosq2)

    alpha = np.arctan2((l2 * sinq2), (l1 + l2 * cosq2))
    beta = np.arctan2(yf, xf)
    q1 = beta - alpha

    return q1, q2


def dibujar_trayectoria_pci(xs, ys, l1, l2):
    q1, q2 = pci(xs[len(xs) - 1], ys[len(ys) - 1], l1, l2)
    
    plt.plot(xs, ys)
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_aspect('equal')
    fig.set_size_inches(10, 10)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim([- (l1 + l2 + 1), l1 + l2 + 1])
    plt.xlim([- (l1 + l2 + 1), l1 + l2 + 1])
    dibujar_robot(q1, q2, l1, l2)
    plt.show()


def dibujar_robot(q1, q2, l1, l2):
    x0, y0 = 0, 0
    x1, y1 = pcd(q1, 0, l1, 0)
    x2, y2 = pcd(q1, q2, l1, l2)
    x, y = [x0, x1, x2], [y0, y1, y2]
    plt.plot(x, y, 'k')
    plt.plot(x1, y1, 'k.')
    plt.plot(x2, y2, 'r.')
    plt.plot(x0, y0, 'b.')


def animacion_trayectoria_pci(xs, ys, l1, l2):
    n = min(len(xs), len(ys))
    for i in range(1, n+1):
        plt.clf()
        dibujar_trayectoria_pci(xs[0:i], ys[0:i], l1, l2)
        plt.pause(0.001)


Vx = np.linspace(1, -1, 100)
Vy = np.linspace(0, 0, 100)

animacion_trayectoria_pci(Vx, Vy, 2, 2)
