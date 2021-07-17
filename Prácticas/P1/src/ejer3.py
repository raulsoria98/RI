import numpy as np
from sympy import *


def dh(matriz):
    final = np.identity(4)
    for fila in matriz:
        rot_z = np.array([[cos(fila[0]), -sin(fila[0]), 0, 0],
                          [sin(fila[0]), cos(fila[0]), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        des_z = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, fila[1]],
                          [0, 0, 0, 1]])
        des_x = np.array([[1, 0, 0, fila[2]],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        rot_x = np.array([[1, 0, 0, 0],
                          [0, cos(fila[3]), -sin(fila[3]), 0],
                          [0, sin(fila[3]), cos(fila[3]), 0],
                          [0, 0, 0, 1]])

        final = final.dot(rot_z)
        final = final.dot(des_z)
        final = final.dot(des_x)
        final = final.dot(rot_x)

    return final


q1, q2, q3 = symbols('q1 q2 q3')
l1, l2, l3 = symbols('l1 l2 l3')
matriz_in = np.array([[q1, l1, 0, 0],
                      [np.deg2rad(90), q2, 0, np.deg2rad(90)],
                      [0, l3 + q3, 0, 0]])

mat = dh(matriz_in)

mat = nsimplify(mat, tolerance=1e-6)

pprint(mat)
