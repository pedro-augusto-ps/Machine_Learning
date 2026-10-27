import math
import matplotlib.pyplot as plt

def funcao_passada(T, t, K=0.1, TA=20.0):
    """T = Temp. café
    TA = Temp. ambiente
    K = Constante"""
    return -K * (T - TA)
    #-K para representar uma perca (perda de calor)

def exato(t, T0=90.0, K=0.1, TA=20.0):
    return TA + (T0 - TA) * math.exp(-K * t)

def euler(funcao_passada, x0, t0, tf, N):
    """Resolve a EDO em passos pequenos
    funcao_passada = Você passa a função, aqui é a lei do resfriamento -K(T - TA)
    x0 = Valor inicial
    t0 e tf = Tempo inicial e Tempo final
    N = Número de passos"""
    h = (tf - t0) / N
    t, x = t0, x0
    ts, xs = [t], [x]
    for i in range(1, N + 1):
        x = x + h * funcao_passada(x, t)  # segue a tangente
        t = t0 + i * h
        ts.append(t)
        xs.append(x)
    return ts, xs

#Exercício 1:
# Com 01_euler_cafe.py, encontre o tempo em que o café atinge 60 °C para min e
# compare com o valor analítico .

valor_inicial = 90 # Café começa quente
tempo_inicial = 0 
tempo_final = 60 # Exercício pede para descobrir quando atinge 60C
Htamanho_passo = 60 #Vai dar 1 por minuto, pq é calculado na fórmula além disso, se quiser mais precisão, 
#aumente o tamanho do passo

resultado_tempo, resultado_temperatura = euler(funcao_passada, valor_inicial, tempo_inicial, tempo_final, Htamanho_passo)

#A função retorna a lista junta, com o tempo e a temperatura, desempacotando os dois, temos uma lista com
#tempo(resultado_tempo) e temperatura(resultado_temperatura)

for i in range(len(resultado_tempo)):
    print(f"Tempo: {resultado_tempo[i]}, Temperatura: {resultado_temperatura[i]}")
    if resultado_temperatura[i] <= 60: #Teste condicional para achar onde o tempo acha 60graus
        aux = resultado_tempo[i]
        break 

formula_dada = -math.log((60 - 20) / (90 - 20)) / 0.1 #Fórmula calcula o tempo
resultado_exato = exato(formula_dada)  
print(f"Nesse tempo exato, a temperatura do café é: {resultado_exato}°C")
print(f"Por euler, no tempo {aux} é <= 60 C")


exato_Ts = [exato(t) for t in resultado_tempo]
plt.plot(resultado_tempo, resultado_temperatura, "o-", label="Euler")
plt.plot(resultado_tempo, exato_Ts, "-", label="Analítica")
plt.xlim(0, 20) #"zoom" no gráfico
plt.legend()
plt.grid(True)

plt.show()