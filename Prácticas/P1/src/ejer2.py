import numpy as np
import matplotlib.pyplot as plt

pxB = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10,
                10, 10, 10, 10, 10, 10, 11, 12, 13, 14, 14, 14, 14,
                14, 14, 14, 14, 14, 14, 14])
pyB = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,
                8, 9, 10, 10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
pzB = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
pB = np.array([pxB, pyB, pzB])


def rotation(eje, alpha):
    if eje.upper() == 'X':
        rotation = np.array([[1, 0, 0],
                             [0, np.cos(alpha), -np.sin(alpha)],
                             [0, np.sin(alpha), np.cos(alpha)]])
    elif eje.upper() == 'Y':
        rotation = np.array([[np.cos(alpha), 0, np.sin(alpha)],
                             [0, 1, 0],
                             [-np.sin(alpha), 0, np.cos(alpha)]])
    else:
        rotation = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                             [np.sin(alpha), np.cos(alpha), 0],
                             [0, 0, 1]])

    return rotation


t = rotation('X', np.deg2rad(60)).dot(rotation('Y', np.deg2rad(90)))\
    .dot(rotation('Z', np.deg2rad(30)))
pt = t.dot(pB)

t2 = rotation('Z', np.deg2rad(30)).dot(rotation('Y', np.deg2rad(90)))\
    .dot(rotation('X', np.deg2rad(60)))
pt2 = t2.dot(pB)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(121, projection='3d')
ax.plot(pt[0], pt[2], pt[1], 'mo', label='Rotación 60X-90Y-30Z')
ax.plot(pB[0], pB[2], pB[1], 'bo', label='Original')
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
plt.legend(loc='upper right')

ax = fig.add_subplot(122, projection='3d')
ax.plot(pt2[0], pt2[2], pt2[1], 'mo', label='Rotación 30Z-90Y-60X')
ax.plot(pB[0], pB[2], pB[1], 'bo', label='Original')
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
plt.legend(loc='upper right')

plt.show()
