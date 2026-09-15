import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Passo 1: Leitura do arquivo texto original
dados_originais = np.loadtxt('Interpolacao-ArquivoAuxiliar-topografia.txt')
num_linhas, num_colunas = dados_originais.shape

# Passo 2: Construção dos eixos e da malha duplicada
x_orig = np.arange(num_colunas)
y_orig = np.arange(num_linhas)

x_novo = np.linspace(0, num_colunas - 1, num_colunas * 2)
y_novo = np.linspace(0, num_linhas - 1, num_linhas * 2)

# Passo 3: Interpolação Bilinear
interpolador = RegularGridInterpolator((y_orig, x_orig), dados_originais, method='linear')
grid_y, grid_x = np.meshgrid(y_novo, x_novo, indexing='ij')
pontos = np.array([grid_y.ravel(), grid_x.ravel()]).T

# (a) Matriz resultante com resolução duplicada
matriz_interpolada = interpolador(pontos).reshape(len(y_novo), len(x_novo))

# (b) Gravação do arquivo texto de saída
np.savetxt('topografia_dobro_resolucao.txt', matriz_interpolada, fmt='%.7e')

# (c) Geração do mapa visual/imagem
plt.figure(figsize=(10, 6))
plt.imshow(matriz_interpolada, cmap='terrain', origin='upper')
plt.colorbar(label='Elevação (m)')
plt.title('Modelo Digital de Elevação (Resolução Duplicada)')
plt.xlabel('Coluna')
plt.ylabel('Linha')
plt.savefig('topografia_interpolada.png', dpi=300, bbox_inches='tight')
plt.show()