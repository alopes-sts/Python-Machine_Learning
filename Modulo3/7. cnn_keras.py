#Exemplo de rede convolucional usando Keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D

(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

print(x_treino.shape)

#Usando one hot encoding
y_treino = to_categorical(y_treino)
y_teste = to_categorical(y_teste)

#O modelo 'Conv2D' precisa que tenha na entrada 3 dimensões, altura, largura e padrão de cores
x_treino = x_treino.reshape(60000, 28, 28, 1) #60000 amostras, matriz de 28x28, e padrão de cor = 1 (cinza)
x_teste = x_teste.reshape(10000, 28, 28, 1)

#Criando a rede CNN
modelo = Sequential()
modelo.add(Conv2D(filters=32, kernel_size=5, activation='relu', input_shape=(28, 28, 1))) #primeira camada
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))                 #segunda camada
modelo.add(Conv2D(filters=64, kernel_size=5, activation='relu'))                          #terceira camada
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))                 #quarta camada
modelo.add(Flatten()) #Função que transforma um espaço multidimensional em uma única dimensão, ex., uma matriz de duas
#dimensões 28x28 seria transformada em uma única dimensão de tamanho84. É como se cada linha da matriz fosse colocada
#uma do lado da outra, ficnado um array longo de uma única dimensão
modelo.add(Dense(80, kernel_initializer='normal', activation='relu'))                      #quinta camada
modelo.add(Dropout(0.2))
modelo.add(Dense(10, kernel_initializer='normal', activation='softmax'))                   #última camada

#Definindo o otimizador e a função de custo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrix=['accuracy'])

#Treinando o modelo
modelo.fit(x_treino, y_treino, batch_size=200, epochs=10, validation_data=(x_teste, y_teste), verbose=1)

