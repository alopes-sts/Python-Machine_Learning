#dataset que verifica a probabilidade de um estudante ser admitido (problema de regressão)
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import regularizers

arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 3/Admission_Predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)
#Separando variáveis preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)
print(x.head())

print(x_treino.shape) #verificando a quantidade de variáveis (colunas e linhas), informação necessária para criar a rede neural
#Criando conjuntos de dados de treino e teste, usando 30% dos dados para teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, shuffle=True)

#Criando rede neural
#Sempre testar com quantidades diferentes de camadas e quantidade de neurônios, quanto maior, mais tmepo de processamento e nem sempre
#é necessário, no exemplo abaixo estamos usando somente 1 camada com 4 neurônios e o resultado já foi satisfatório.
modelo = Sequential()
modelo.add(Dense(4, input_dim=7, kernel_initializer='normal', activation='relu')) #(input_dim=7, temos 7 colunas no dataset
modelo.add(Dense(1, kernel_initializer='normal', activation='linear'))

#Definindo otimizador
otimizador = keras.optimizers.Adam()
modelo.compile(loss='mean_squared_error', optimizer=otimizador, metrics=['mean_squared_error'])
modelo.fit(x_treino, y_treino, epochs=1000, batch_size=280, validation_data=(x_teste, y_teste), verbose=1) #batch_size=280, temos 280 linhas no dataset

########## Refazendo a mesma rede neural com mais camadas ocultas e mais neurônios e usando 'dropout' para comparar os resultados
modelo = Sequential()
modelo.add(Dense(50, input_dim=7, kernel_initializer='normal', activation='relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(50, input_dim=7, kernel_initializer='normal', activation='relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(50, input_dim=7, kernel_initializer='normal', activation='relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(1, kernel_initializer='normal', activation='linear'))
otimizador = keras.optimizers.Adam()
modelo.compile(loss='mean_squared_error', optimizer=otimizador, metrics=['mean_squared_error'])
modelo.fit(x_treino, y_treino, epochs=1000, batch_size=280, validation_data=(x_teste, y_teste), verbose=1)


