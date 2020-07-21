#Exercicio utilizando o dataset que verifica preço de imóveis na cidade de Boston (Regressão)
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
boston = load_boston()

#Definindo variáveis preditoras e variável target
x = pd.DataFrame(boston.data, columns = boston.feature_names) #Criando o dataset e definindo os nomes das colunas (que já estão no dataset importado do sklearn)
y = boston.target #Já existe uma variável target definida do dataset
print(x.shape) #Precisamos verificar a quantidade amostras e features, a quantidade de features será usada com quantide de entrada de neurônios

print(x.head())
print(y)  #note que a variável y é um array e não será necessário separar em colunas como é feito em problemas de classificação

#Separando os dados entre treino e teste, deixando 30% para teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3)

modelo = Sequential()
modelo.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu')) #Entrada e primeira camada oculta, dim=13 pq temos 13 colunas no dataset
modelo.add(Dense(20, kernel_initializer='normal', activation='relu')) #segunda camada oculta
modelo.add(Dense(1, kernel_initializer='normal', activation='linear')) #saída

otimizador = Adam()
modelo.compile(loss='mean_squared_error', optimizer=otimizador, metrics=['mae']) #usando como função de custo o mean squared error por ser um problema de regressão
print(x_treino.shape)

#verificando os resultados, podemos notar que o erro médio é de 3 pontos para mais ou para menos
historico = modelo.fit(x_treino, y_treino, epochs=1000, batch_size=354, validation_data=(x_teste, y_teste), verbose=1)

#Visualizando graficamente usando ADAM  (alterar o otimizador para RMSprop para comparar)
mae_treino = historico.history['mean_absolute_error']
mae_teste = historico.history['val_mean_absolute_error']
epochs = range(1, len(mae_treino)+1)
plt.plot(epochs, mae_treino, '-g', label='MAE Dados de treino')
plt.plot(epochs, mae_teste, '-b', label='MAE Dados de teste')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.show()

