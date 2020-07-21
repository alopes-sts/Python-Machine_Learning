# Exemplo de CNN usando imagens coloridas
########## Usando Data augmentation (manipulando as imagens para que fiquem ligeiramente diferentes) ##########
# notar que com o uso do data augmentation, a parte que cria o modelo é um pouco diferente
# Dataset que contêm 60000 imagens coloridas em 10 categorias (carros, aves, automóveis, etc), já treinados
# São 50000 amostras de treino e 10000 de teste

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# baixando dataset já dividido em treino e teste
(x_treino, y_treino), (x_teste, y_teste) = cifar10.load_data()

# Exibindo algumas imagens
for i in range(9):
    plt.subplot(3,3,i+1)    # (linhas, colunas, índices), o índice é gerado automaticamente a partir das imagens que fornecemos
    plt.imshow(x_treino[i]) # Informa qual imagem deve mostrar considerando o índice no dataset
plt.show()

print(x_treino.shape)  # somente para visualizar o tamanho do dataset
print(y_treino.shape)  # sempre é necessário analisar a variável Y, neste caso, devido ao formato, será necessário executar o one hot encode

# Aplicando o one hot encode
y_treino = to_categorical(y_treino)
y_teste = to_categorical(y_teste)
print(y_treino.shape)  # Confirmando o formato

# Verificando os dados da variável X, neste caso, pode-se verificar que os dados são valores inteiros que variam
# de 0 a 255 (escala de cores RGB) e precisamos normalizar, convertendo em valores entre 0 e 1
print(x_treino[0])

# Antes de normalizar, precisa converter para float
# Para normalizar, o maior número que temos é 255, então temos que dividir todos os números por 255
x_treino_float = x_treino.astype('float32')
x_teste_float = x_teste.astype('float32')
# Normalizando
x_treino_normalizado = x_treino_float / 255.0
x_teste_normalizado = x_teste_float / 255.0

# No Keras, é possível forçar a inicialização 'Xavier' que irá facilitar os cálculos de escolha para os melhores pesos
# para isso, é só passar o parâmetro "kernel_initializer='glorot_uniform'", é uma boa prática usar desta forma
# o 'glorot_uniform' já é a opção default para os layer CNN, por isso não foi informado para o Conv2D
# Criando a Rede CNN usando data augmentation
########## Atentar que o modelo foi alterado em relação ao exemplo anterior (sem uso de data augmentation) ##########
modelo = Sequential()
modelo.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32, 32, 3)))  # O kernel size foi reduzido, como a rede neural irá ficar mais profunda, um valor de kernel size maior deixaria a camada da sa´da com valroes negativos
modelo.add(Conv2D(filters=32, kernel_size=3, activation='relu'))                           # Linha adicionada devido ao uso Data augmentation
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
modelo.add(Dropout(0.2))                                                                   # Linha adicionada devido ao uso Data augmentation
modelo.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
modelo.add(Conv2D(filters=64, kernel_size=3, activation='relu'))                           # Linha adicionada devido ao uso Data augmentation
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
modelo.add(Dropout(0.2))                                                                   # Linha adicionada devido ao uso Data augmentation
modelo.add(Conv2D(filters=128, kernel_size=3, activation='relu'))                           # Linha adicionada devido ao uso Data augmentation
modelo.add(Conv2D(filters=128, kernel_size=3, activation='relu'))                           # Linha adicionada devido ao uso Data augmentation
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
modelo.add(Flatten())                                                                      # Transforma um espaço multidimensional em uma única dimensão
modelo.add(Dense(130, kernel_initializer='glorot_uniform', activation='relu'))             # O número de neurônios foi aumentando em relação ao exemplo de CNN sem uso de data augmentation
modelo.add(Dropout(0.3))
modelo.add(Dense(10, kernel_initializer='glorot_uniform', activation='softmax'))           # Alterado para softmax

# Definindo o otimizador e a função de custo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Treinando o modelo
historico = modelo.fit(x_treino_normalizado, y_treino, batch_size=200, epochs=10, validation_data=(x_teste_normalizado, y_teste), verbose=1)

acuracia_treino = historico.history['acc']
acuracia_teste = historico.history['val_acc']
epochs = range(1, len(acuracia_treino)+1)
plt.plot(epochs, acuracia_treino, '-g', label='Acurácia dos dados de treino')
plt.plot(epochs, acuracia_teste, '-b', label='Acurácia dos dados de teste')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acurácia')
plt.show()

########## DATA AUGMENTATION ##########
# Iremos alterar cada imagem de entrada, conforme parâmetros que serão passados aleatoriamente.
# Em cada epoch o conjunto de dados de treino será diferente, as imagens são as mesmas, mas serão deslocadas para o lado ou para cima

aug_data = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True) # Configurando o gerador de dados
treino_aumentado = aug_data.flow(x_treino_normalizado, y_treino, batch_size=200)                   # Passando os dados de entrada para o gerador
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])            # Definindo o otimizador e a função de custo
n_passadas = int(x_treino_normalizado.shape[0] / 200)                                              # Treinando o modelo
historico = modelo.fit_generator(treino_aumentado, steps_per_epoch=n_passadas, epochs=100, validation_data=(x_teste_normalizado, y_teste), verbose=1)

imagem = x_teste[10]
plt.imshow(imagem)
plt.show()

imagem = imagem.astype('float32')
imagem = imagem / 255.0
imagem = np.expand_dims(imagem, axis=0)   # Criando uma dimensão extra para informar que há apenas uma imagem por batch_size

resultado = modelo.predict_classes(imagem)
print(resultado[0])
print(resultado)
