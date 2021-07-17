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
        rotation = np.array([1, 0, 0,
                             0, np.cos(alpha), -np.sin(alpha),
                             0, np.sin(alpha), np.cos(alpha)])
    elif eje.upper() == 'Y':
        rotation = np.array([np.cos(alpha), 0, np.sin(alpha),
                             0, 1, 0,
                             -np.sin(alpha), 0, np.cos(alpha)])
    else:
        rotation = np.array([np.cos(alpha), -np.sin(alpha), 0,
                             np.sin(alpha), np.cos(alpha), 0,
                             0, 0, 1])

    rotation = rotation.reshape(3, 3)
    return rotation


pBX = rotation('X', np.deg2rad(90)).dot(pB)
xX = pBX[0]
yX = pBX[1]
zX = pBX[2]
pBY = rotation('Y', np.deg2rad(90)).dot(pB)
xY = pBY[0]
yY = pBY[1]
zY = pBY[2]
pBZ = rotation('Z', np.deg2rad(90)).dot(pB)
xZ = pBZ[0]
yZ = pBZ[1]
zZ = pBZ[2]

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(221, projection='3d')
ax.plot(pxB, pzB, pyB, 'bo', label='Original')
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
plt.legend()

ax = fig.add_subplot(222, projection='3d')
ax.plot(xX, zX, yX, 'ro', label='Rotación 90 X')
ax.plot(pxB, pzB, pyB, 'bo', label='Original')
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
plt.legend()

ax = fig.add_subplot(223, projection='3d')
ax.plot(xY, zY, yY, 'go', label='Rotación 90 Y')
ax.plot(pxB, pzB, pyB, 'bo', label='Original')
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
plt.legend()

ax = fig.add_subplot(224, projection='3d')
ax.plot(xZ, zZ, yZ, 'mo', label='Rotación 90 Z')
ax.plot(pxB, pzB, pyB, 'bo', label='Original')
ax.set_xlabel("X")
ax.set_ylabel("Z")
ax.set_zlabel("Y")
plt.legend()

plt.show()
