import numpy as np
import matplotlib.pyplot as plt


def pci(xf, yf, l1, l2):
    cosq2 = (xf ** 2 + yf ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    sinq2 = np.sqrt(1 - cosq2 ** 2)
    q2 = np.arctan2(sinq2, cosq2)

    alpha = np.arctan((l2 * sinq2) / (l1 + l2 * cosq2))
    beta = np.arctan2(yf, xf)
    q1 = beta - alpha

    return q1, q2


q1f, q2f = pci(1, 0.5, 1, 1)

print('q1:', np.rad2deg(q1f), 'q2:', np.rad2deg(q2f))
