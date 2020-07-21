#Exercicio utilizando o dataset que verifica os pesos em uma balança (classificação)
#a variável target dete dataset informa para qual lado a balança pendeu ou se ficou equilibrada
#será usado o otimizador 'amsgrad'
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation
arquivo = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                      names=['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'])

print(arquivo.head()) #verificando o dataset

#Definindo variáveis preditoras e variável target
y = arquivo['Class']
x = arquivo.drop('Class', axis=1)
print(y.value_counts()) #verificando a quantidade de informações em cada classe

#A coluna class é nossa variável target, com dados alfanuméricos, precisamos transformar estes dados em números
y.replace('L', 0, inplace=True)
y.replace('R', 1, inplace=True)
y.replace('B', 2, inplace=True)
y_convertido = np_utils.to_categorical(y)
#Visualizando os dados da coluna target convertidos em colunas
print(y_convertido)

#Separando os dados entre treino e teste, deixando 20% para teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y_convertido, test_size = 0.2)

#Criando o modelo
modelo = Sequential()
#criando três camadas ocultas com 50 neurônios
#podemos manipular estes valores, aumentando a quantidade da camadas ou quantidade de neurônios para verificar o resultado
modelo.add(Dense(50, input_dim=4, kernel_initializer='normal', activation='relu')) #primeira camada oculta
modelo.add(Dense(50, kernel_initializer='normal', activation='relu')) #segunda camada oculta
modelo.add(Dense(50, kernel_initializer='normal', activation='relu')) #terceira camada oculta
modelo.add(Dense(3, kernel_initializer='normal', activation='softmax')) #saída
#Usando como otimizador o 'amsgrad'
otimizador = Adam(amsgrad=True)
modelo.compile(loss='categorical_crossentropy', optimizer=otimizador, metrics=['acc'])
print(x_treino.shape) #Verificando o tamanho dos dados de treino

#verificando os resultados
modelo.fit(x_treino, y_treino, epochs=1000, batch_size=500, validation_data=(x_teste, y_teste), verbose=1)
