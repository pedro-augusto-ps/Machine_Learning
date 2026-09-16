#Objetivo: Dobrar uma "imagem"(pixeis estão em forma de matriz)
#utilizando interpolação bilinear e gerar uma visualização dos dados.

import numpy as np
import matplotlib.pyplot as plt

with open("topografia.txt", "r") as arquivo:
    dados = arquivo.read()

dados = dados.replace(",",".")
linhas = dados.splitlines()
matriz = np.loadtxt(linhas)

print(matriz)