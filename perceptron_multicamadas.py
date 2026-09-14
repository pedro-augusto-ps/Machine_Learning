import numpy as np
import math
#Conteúdo auxiliar: Site Kaggle
#-----------------IMPORTANTE-----------------
#MATRIZ 3x3, iniciado em 0
#COLUNA 0 e 1 = PESOS
#COLUNA 2 = BIAS
#-----------------IMPORTANTE-----------------
#A = SIGMOIDE
np.random.seed(10) #Travo a aleatoriedade
matriz = np.random.rand(3, 3) #Crio a matriz 3x3 com valores aleatórios
print(matriz)

taxa_aprendizado = 0.5

valores = np.array([[0.0, 0.0],[1.0, 1.0],[1.0, 0.0], [0.0, 1.0]]) #Valores da XOR
desejado = np.array([0.0, 0.0, 1.0, 1.0])  #Saida esperada da XOR
print(matriz[1][1])
def sigmoide(valor):
    A = 1 / (1 + math.exp(-valor))
    return A

def calcular_saida(valores, matriz, desejado):
    for posicao in range(len(valores)):
        H1 = (valores[posicao][0] * matriz[0][0]) + (valores[posicao][1] * matriz[0][1]) + matriz[0][2]
        A1 = sigmoide(H1)

        H2 = (valores[posicao][0] * matriz[1][0]) + (valores[posicao][1] * matriz[1][1]) + matriz[1][2]
        A2 = sigmoide(H2)

        H_saida = (A1 * matriz[2][0]) + (A2 * matriz[2][1]) + matriz[2][2]
        A3 = sigmoide(H_saida)

        #ERRO
        erro = desejado[posicao] - A3
        derivada_saida = A3 * (1 - A3) #Não sei derivar, tive que pegar o contexto na net

        #BACKPROPAGATION
        erro_saida = erro * derivada_saida
        erro_H1 = A1 * (1 - A1) * erro_saida * matriz[2][0]
        erro_H2 = A2 * (1 - A2) * erro_saida * matriz[2][1]

        #ATUALIZANDO OS PESOS
        #PESOS H1
        matriz[0][0] = matriz[0][0] + taxa_aprendizado * erro_H1 * valores[posicao][0]
        matriz[0][1] = matriz[0][1] + taxa_aprendizado * erro_H1 * valores[posicao][1]
        matriz[0][2] = matriz[0][2] + taxa_aprendizado * erro_H1 #BIAS

        #PESOS H2
        matriz[1][0] = matriz[1][0] + taxa_aprendizado * erro_H2 * valores[posicao][0]
        matriz[1][1] = matriz[1][1] + taxa_aprendizado * erro_H2 * valores[posicao][1]
        matriz[1][2] = matriz[1][2] + taxa_aprendizado * erro_H2 #BIAS

        #PESOS SAIDA
        # Pesos da saída
        matriz[2][0] = matriz[2][0] + taxa_aprendizado * erro_saida * A1
        matriz[2][1] = matriz[2][1] + taxa_aprendizado * erro_saida * A2
        matriz[2][2] = matriz[2][2] + taxa_aprendizado * erro_saida #BIAS

    print(f"ENTRADA: {valores[posicao]}")
    print(f"SAÍDA: {A3}")
    print(f"DESEJADO: {desejado[posicao]}")
    print("\n")

for epoca in range(1):
    calcular_saida(valores, matriz, desejado)