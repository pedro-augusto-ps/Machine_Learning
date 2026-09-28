#Objetivo: Dobrar uma "imagem"(pixeis estão em forma de matriz)
#utilizando interpolação bilinear e gerar uma visualização dos dados.

import numpy as np
import matplotlib.pyplot as plt

with open("topografia.txt", "r") as arquivo:
    dados = arquivo.read() #Abre o arquivo para leitura

dados = dados.replace(",",".") #Troca os dados que possuem ,
linhas = dados.splitlines() #Fatia as linhas
matriz = np.loadtxt(linhas) #Gera uma matriz com as linhas

