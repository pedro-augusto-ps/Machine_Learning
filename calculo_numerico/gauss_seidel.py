#Código feito pelo professor em aula

import numpy as np
from numpy.linalg import norm, inv

def Gauss_seide(matriz, independentes, tolerancia):
    n = np.size(matriz, 0)
    k = 0
    x = np.zeros(n)
    lower = np.zeros([n , n])
    diagonal = np.zeros([n , n])
    upper = np.zeros([n , n])
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                diagonal[i, j] = matriz[i, j]
            elif i>j:
                lower[i , j] = matriz[i, j]
            elif i<j:
                upper[i, j] = matriz[i, j]
    LDinv = inv(lower+diagonal)
    erro = 2*tolerancia

    while erro>tolerancia:
        x = np.matmul(LDinv,(independentes-np.matmul(upper, x)))
        erro = norm(np.matmul(matriz, x)-independentes)
        k += 1
    return x, k

matriz = np.array([[9.0, -0.1, -2.4],
              [0.1,  7.0, -5.0],
              [2.5, -0.2, 10.0]])

independentes = np.array([7.85, -19.3, 71.4])

tolerancia = 6 ** -6
Gauss_seide(matriz, independentes, tolerancia)