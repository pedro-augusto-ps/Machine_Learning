#Tentativa de criar um perceptron multicamadas genérico
#IMPORTANTE:
# Utilizei IA para me ajudar no backpropagation
# Arquitetura da RNA, o tamanho deve corresponder a MATRIZ "VALORES"
# Código não tem tratamento de erros

import numpy as np
import math

qtd_entradas = int(input("Insira a quantia de entradas que irá ter: "))
qtd_oculta = int(input("Insira a quantia de camadas ocultas que irá ter: ")) #qtd de neuronios
qtd_saida = int(input("Insira a quantia de saídas que irá ter: "))


#valores = (exemplo: XOR, 2-2-1)
#linhas de "valores" precisa ser = qtd_entradas colunas,
#linhas de "desejado" precisa ser = qtd_saida colunas.
entradas = []
valores = np.array([[0.0, 0.0],[1.0, 1.0],[1.0, 0.0], [0.0, 1.0]]) 
desejado = np.array([[0.0], [0.0], [1.0], [1.0]])
taxa_aprendizado = 0.5

#Criação dos pesos:
#É possível fazer automático também
matriz_pesos = np.random.rand(qtd_oculta, qtd_entradas + 1) #+1 reserva o BIAS
matriz_pesos_saida = np.random.rand(qtd_saida, qtd_oculta + 1) #+1 reserva o BIAS

respostas_ocultos = np.zeros(qtd_oculta) #Guarda a resposta dos neuronios

def sigmoide(valor):
    A = 1 / (1 + math.exp(-valor))
    return A

def forward(entradas, matriz_pesos, matriz_pesos_saida):
    """Calcula cada neurônio oculto por vez
    Entradas = Valores fornecidos pelo usuário
    matriz_pesos = Matriz de tamanho escolhido pelo usuário porém com valores random
    matriz_pesos_saida = Matriz de pesos para saída, 0-0-s 
    OBS: Tem uma matriz para H e uma matriz para S"""
    for neuronio_atual in range(qtd_oculta):
        calculo = 0.0
        for entrada_atual in range(qtd_entradas):
            peso = matriz_pesos[neuronio_atual][entrada_atual]
            calculo += entradas[entrada_atual] * peso 
        calculo += matriz_pesos[neuronio_atual][qtd_entradas]#Bias fica na ultíma coluna
        #Faz = (X * W) + B e quarda nessa matriz      
        respostas_ocultos[neuronio_atual] = sigmoide(calculo)

    #SAÍDA
    respostas_saidas = np.zeros(qtd_saida) #Guarda a resposta das saídas
    
    for neuronio_saida in range(qtd_saida):
        calculo = 0.0
        for neuronio_oculto in range(qtd_oculta):
            peso = matriz_pesos_saida[neuronio_saida][neuronio_oculto]
            calculo += respostas_ocultos[neuronio_oculto] * peso
        calculo += matriz_pesos_saida[neuronio_saida][qtd_oculta] #bias
        #Faz (Ativação pesos * peso) + b e quarda nessa matriz
        respostas_saidas[neuronio_saida] = sigmoide(calculo)


    return respostas_ocultos, respostas_saidas 

def treinar(entrada, resposta_desejada, matriz_pesos, matriz_pesos_saida):
    #FORWARD
    respostas_ocultos, respostas_saidas = forward(entrada, matriz_pesos, matriz_pesos_saida)

    #ERRO SAÍDA
    matriz_erros_saida = np.zeros(qtd_saida) 
    for neuronio_saida in range(qtd_saida):
        saida = respostas_saidas[neuronio_saida]
        erro = resposta_desejada[neuronio_saida] - saida
        derivada = saida * (1 - saida)  #Não sei derivadar, pedi ajuda pra IA
        matriz_erros_saida[neuronio_saida] = erro * derivada
 
    #BACKPROPAGATION
    #IMPORTANTE:
    #Pelo que entendi, o backpropagation é a atualização do valores + aprendizagem * 
    #O erro desse mesmo valores * o valor atual

    matriz_erros_ocultos = np.zeros(qtd_oculta)
    for neuronio_oculto in range(qtd_oculta):
        calculo = 0.0
        for neuronio_saida in range(qtd_saida):
            peso = matriz_pesos_saida[neuronio_saida][neuronio_oculto]
            calculo += matriz_erros_saida[neuronio_saida] * peso
        derivada = respostas_ocultos[neuronio_oculto] * (1 - respostas_ocultos[neuronio_oculto])
        matriz_erros_ocultos[neuronio_oculto] = derivada * calculo
 
    #ATUALIZAÇÃO CAMADA OCULTA
    for neuronio_atual in range(qtd_oculta):
        for entrada_atual in range(qtd_entradas):
            matriz_pesos[neuronio_atual][entrada_atual] += (taxa_aprendizado * matriz_erros_ocultos[neuronio_atual] * entrada[entrada_atual])
        matriz_pesos[neuronio_atual][qtd_entradas] += taxa_aprendizado * matriz_erros_ocultos[neuronio_atual]  # bias
 
    #ATUALIZAÇÃO SAÍDA
    for neuronio_saida in range(qtd_saida):
        for neuronio_oculto in range(qtd_oculta):
            matriz_pesos_saida[neuronio_saida][neuronio_oculto] += taxa_aprendizado * matriz_erros_saida[neuronio_saida] * respostas_ocultos[neuronio_oculto]
        matriz_pesos_saida[neuronio_saida][qtd_oculta] += taxa_aprendizado * matriz_erros_saida[neuronio_saida]  # bias
 
    return respostas_saidas

qtd_epocas = int(input("Insira a quantia de épocas: "))

for epoca in range(qtd_epocas):
    for posicao in range(len(valores)):
        saida = treinar(valores[posicao], desejado[posicao], matriz_pesos, matriz_pesos_saida)
        if epoca % 1000 == 0:
            print(f"Época: {epoca}")
            print(f"Entrada: {valores[posicao]} Saída: {saida} Desejado: {desejado[posicao]}")
print(f"Treino concluído")