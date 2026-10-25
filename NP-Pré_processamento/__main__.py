import numpy as np
import pandas as pd

dados = pd.read_csv("wdbc.data", delimiter=",", header=list)
print(dados)
#------------------------LIMPANDO OS DADOS------------------------
#LIMPAGEM DOS DADOS
# matriz_filtrada = []
# for linha in matriz: #Numpy, percorre a linha inteira
#     for elemento in linha: #Segundo laço, estou no elemento específico
#         if np.isnan(elemento) == True: #Isnan == True, elemento é nan
#             break #essa linha não vai ser adicionada
#     else:
#         matriz_filtrada.append(linha)

# matriz_limpa = np.array(matriz_filtrada) #Python LIST -> np.array
# #------------------------LIMPANDO OS DADOS------------------------


# #------------------------NORMALIZAÇÃO------------------------
# matriz_normalizada = np.copy(matriz_limpa)

# minimo = np.min(matriz_normalizada, axis=0)
# maximo = np.max(matriz_normalizada, axis=0)

# #Não sabia que isso era possível, aprendi agora,
# #NUMPY usa C, então com essa sintaxe "percorro" todos elementos fazendo as contas
# matriz_normalizada = (matriz_normalizada - minimo) / (maximo - minimo)
# #------------------------NORMALIZAÇÃO------------------------


# #------------------------REPARTINDO O TRABALHO------------------------
# '''Sobre as variáveis de divisão:
# fim_treino = posição onde o treino termina (excluída): 70% do total.
# fim_validacao = posição onde a validação termina (excluída): fim_treino + 15% do total.
# O teste é o que sobra, de fim_validacao até o final 15%'''
# fatores_normalizacao = np.vstack((minimo, maximo))

# num_linhas = matriz_normalizada.shape[0] #.shape -> dimensões array [0]=Colunas

# indices_linhas = np.random.permutation(num_linhas) #Embaralha os índices
# fim_treino = int(num_linhas * 0.70) #Pare na linha 7
# fim_validacao = fim_treino + int(num_linhas * 0.15) #Fim do validação = fim_treino + 15% do tota

# treinamento = matriz_normalizada[indices_linhas[:fim_treino]] #0->6
# validacao = matriz_normalizada[indices_linhas[fim_treino:fim_validacao]] #7->8
# teste = matriz_normalizada[indices_linhas[fim_validacao:]] #8->Final
# #------------------------REPARTINDO O TRABALHO------------------------

# #4 - SALVAR OS DADOS
# np.savetxt('matriz_original.txt',matriz, delimiter=',')
# np.savetxt('matriz_normalizada.txt',matriz_normalizada, delimiter=',')
# np.savetxt('fatores_normalizacao.txt', fatores_normalizacao, delimiter=',')
# np.savetxt('treinamento.txt', treinamento, delimiter=',')
# np.savetxt('validacao.txt', validacao, delimiter=',')
# np.savetxt('teste.txt', teste, delimiter=',')



