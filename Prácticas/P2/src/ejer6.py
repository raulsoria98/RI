import numpy as np
import matplotlib.pyplot as plt


def pcd(q1, q2, l1, l2):
    x1 = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y1 = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

    return x1.round(5), y1.round(5)


def dibujar_trayectoria_pcd(q1s, q2s, l1, l2):
    vx, vy = [], []
    for i in range(q1s.size):
        x, y = pcd(q1s[i], q2s[i], l1, l2)
        vx.append([x])
        vy.append([y])
    
    plt.plot(vx, vy)
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_aspect('equal')
    fig.set_size_inches(8, 8)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim([- (l1 + l2 + 1), l1 + l2 + 1])
    plt.xlim([- (l1 + l2 + 1), l1 + l2 + 1])
    dibujar_robot(q1s[q1s.size - 1], q2s[q2s.size - 1], l1, l2)
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


def animacion_trayectoria_pcd(q1s, q2s, l1, l2):
    n = min(len(q1s), len(q2s))
    for i in range(1, n+1):
        plt.clf()
        dibujar_trayectoria_pcd(q1s[0:i], q2s[0:i], l1, l2)
        plt.pause(0.001)


Vq1 = np.linspace(0, np.pi, 101)
primera_mitad = np.linspace(0, 50 * np.pi / 100, 51)
segunda_mitad = primera_mitad[49::-1]
Vq2 = np.append(primera_mitad, segunda_mitad)

animacion_trayectoria_pcd(Vq1, Vq2, 2, 2)
