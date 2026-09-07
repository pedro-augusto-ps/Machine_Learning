import numpy as np
import matplotlib.pyplot as plt

#Treinamento da porta AND
# 0 - 1 = 0
# 1 - 0 = 0
# 0 - 0 = 0
# 1 - 1 = 1

#Função de ativação:
#saida > 0 = 1
#saida < 0 = 0

#PESOS INICIAIS 
w1 = 0.5
w2 = 0.5

entradas = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
esperado = np.array([0.0, 0.0, 0.0, 1.0])
bias = 0
taxa_aprendizado = 1

def ativacao(valor):
    if valor > 0:
        return 1
    else:
        return 0

for epoca in range(10):

    for posicao in range(len(entradas)):
        valor = (entradas[posicao][0] * w1) + (entradas[posicao][1] * w2) + bias

        saida = ativacao(valor)
        erro = esperado[posicao] - saida

        #Atualizando os pesos = wnovo = wantigo + learning_rate * (esperado - saida) * entrada
        w1 = w1 + taxa_aprendizado * (erro) * entradas[posicao][0]
        w2 = w2 + taxa_aprendizado * (erro) * entradas[posicao][1]

        #Atualizando o bias
        bias = bias + taxa_aprendizado * erro

    print(f"--------EPOCA: {epoca}--------")
    print(f"--------SAIDA: {saida}--------")
    print(f"--------ERRO: {erro}--------")
    print(f"--------ESPERADO: {esperado[posicao]}--------")
    print("\n")