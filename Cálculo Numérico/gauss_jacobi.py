import numpy as np

matriz = np.array([
    [9.0, -0.1, -2.4],
    [0.1,  7.0, -5.0],
    [2.5, -0.2, 10.0]
])

independentes = np.array([7.85, -19.3, 71.4])
xs = [0, 0, 0]

while True:
    xs_novo = []
    for i in range(len(matriz)):
        soma = 0
        for j in range(len(matriz[i])):
            if i != j:
                peso = matriz[i][j]
                valor = xs[j]
                calculo = peso * valor
                soma += calculo
        xs_novo.append((independentes[i] - soma) / matriz[i][i]) 
    erro = 0
    for i in range(len(xs)):
        diferenca = abs(xs_novo[i] - xs[i])

        if diferenca > erro:
            erro = diferenca
    xs = xs_novo.copy()
    if erro < 10 ** -6:
        break

    print(xs)
    print("-----------")