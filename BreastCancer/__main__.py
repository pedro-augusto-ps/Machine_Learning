#ATENÇÃO - CÓDIGO FORNECIDO PELO PROFESSOR

import numpy as np
import matplotlib.pyplot as plt
import csv
import copy as cp
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Função de treinamento e teste do perceptron multicamada
def multilayer_perceptron(dados):
    x = np.array(list(dados)).astype(float)
    output = x[:, 2]     # diagnóstico
    Input = x[:, 3:]     # 10 features
    bias = np.array([-1,-1,-1])
    learning_rate = float(input('Taxa de aprendizado [0,1]: '))
    epochs = int(input('Número de épocas: '))
        
    weights = np.random.rand(3,3)
    Input_train,Input_test,output_train,output_test = train_test_split(Input,output,random_state=0)
    
    for iter in range(0,epochs):
        out = np.zeros(np.size(Input_train,0))
        for j in range(0,np.size(Input_train,0)):
            # forward
            H1 = bias[0]*weights[0,0] + Input_train[j,0]*weights[0,1]+Input_train[j,1]*weights[0,2]
            X1 = 1/(1+np.exp(-H1))
            H2 = bias[1]*weights[1,0] + Input_train[j,0]*weights[1,1]+Input_train[j,1]*weights[1,2]
            X2 = 1/(1+np.exp(-H2))
            R = bias[2]*weights[2,0] + X1*weights[2,1]+X2*weights[2,2]
            out[j] = 1/(1 + np.exp(-R))
            
            # backward
            weights[2,0] = weights[2,0] + learning_rate*bias[2]*out[j]*(1-out[j])*(output_train[j]-out[j])
            weights[2,1] = weights[2,1] + learning_rate*X1*out[j]*(1-out[j])*(output_train[j]-out[j])
            weights[2,2] = weights[2,2] + learning_rate*X2*out[j]*(1-out[j])*(output_train[j]-out[j])        
            
            weights[0,0] = weights[0,0] + learning_rate*bias[0]*X1*(1-X1)*weights[2,1]*out[j]*(1-out[j])*(output_train[j]-out[j])
            weights[1,0] = weights[1,0] + learning_rate*bias[1]*X2*(1-X2)*weights[2,2]*out[j]*(1-out[j])*(output_train[j]-out[j])
                        
            weights[0,1] = weights[0,1] + learning_rate*Input_train[j,0]*X1*(1-X1)*weights[2,1]*out[j]*(1-out[j])*(output_train[j]-out[j])
            weights[1,1] = weights[1,1] + learning_rate*Input_train[j,0]*X2*(1-X2)*weights[2,2]*out[j]*(1-out[j])*(output_train[j]-out[j])
                       
            weights[0,2] = weights[0,2] + learning_rate*Input_train[j,1]*X1*(1-X1)*weights[2,1]*out[j]*(1-out[j])*(output_train[j]-out[j])
            weights[1,2] = weights[1,2] + learning_rate*Input_train[j,1]*X2*(1-X2)*weights[2,2]*out[j]*(1-out[j])*(output_train[j]-out[j])
    
    # Aplicação da rede ao conjunto teste
    out_test = np.zeros(np.size(Input_test,0))
    for j in range(0,np.size(Input_test,0)):
            # forward
            H1 = bias[0]*weights[0,0] + Input_test[j,0]*weights[0,1]+Input_test[j,1]*weights[0,2]
            X1 = 1/(1+np.exp(-H1))
            H2 = bias[1]*weights[1,0] + Input_test[j,0]*weights[1,1]+Input_test[j,1]*weights[1,2]
            X2 = 1/(1+np.exp(-H2))
            R = bias[2]*weights[2,0] + X1*weights[2,1]+X2*weights[2,2]
            out_test[j] = 1/(1 + np.exp(-R))
    
    # Confusion matrix    
    for i in range(0,np.size(out_test)):
        out_test[i] = round(out_test[i])
    out_test = out_test.astype(float)
    
    cm = confusion_matrix(output_test,out_test)

    fig, ax = plt.subplots()
    
    im = ax.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ['Benigno','Maligno']
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='Diagnóstico real',
           xlabel='Diagnóstico RNA')
    
    fmt = '.2f'
    thresh = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    # classificações
    tp = cm[1,1]
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    
    # métricas de desempenho
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision =  tp/(tp + fp)
    recall = tp/(tp + fn)
    F1 = 2*precision*recall/(precision + recall)
    print('')
    print('Métricas de Desempenho:')
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-Score: ', F1)
    return weights

dados = csv.reader(open("treinamento.txt"), delimiter=",")
multilayer_perceptron(dados)
plt.show()