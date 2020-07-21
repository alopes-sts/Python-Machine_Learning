#executado no jupyter notebook
import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import warnings
warnings.filterwarnings('ignore')
import keras
from keras.datasets import mnist #este dataset possui 60mil imagens de digitos escritos a mão

(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

from keras.utils import np_utils
y_treino_convertido = np_utils.to_categorical(y_treino) #converte a coluna de valroes em uma matriz de classes (one hot encode)
y_teste_convertido = np_utils.to_categorical(y_teste)

#y_treino_convertido

#y_treino[0] #cada classe representa a imagem de um número (que foi feito a mão), então o primeiro valro deste array é uma imagem que representa o número cinco
import matplotlib.pyplot as plt
#aqui podemos visualizar a imagem que é formada, será o número 5
plt.imshow(x_treino[0], cmap='gray') #visualizando uma das imagens do dataset, notar a escala de 28x28, totalizando 784 pixels

#x_treino[1000] #se pegarmos qualquer posição do array podemos visualizar a matriz de 28x28 pixels, as posições com números até 255, onde zero é um pixel totalmente preto e conforme os valores aumentam o pixel fica mais claro, isso pode ser visto na imagem gerada pelo matplotlib acima

from keras.models import Sequential
from keras.layers import Dense, Activation
#este dataset não está no formato que podemos usar na rede neural
#x_treino.shape #temos 60mil amostras em uma dimensão de 28x28

x_treino_remodelado = x_treino.reshape((60000, 784)) #precisamos transformar estas 2 colunas em uma única coluna (28x28=784 valores)
x_teste_remodelado = x_teste.reshape((10000, 784)) #o mesmo será feito para os dados de teste
#precisamos normalizar os dados, temos informações que variam de 0 a 255 (onde 255 é nosso 100%) e precisamos de informaç~çoes que variam de 0 a 1
x_treino_normalizado = x_treino_remodelado.astype('float32') / 255 #dividindo por 255 teremos valores de 0 a 1
x_teste_normalizado = x_teste_remodelado.astype('float32') / 255 #o mesmo para os dados de teste
#x_treino_normalizado[0] #visualizado os dados que formam o número 5 (lembrando que a posição zero no dataset é uma imagem do número 5)

modelo = Sequential()
#estamos criando uma camada oculta de 30 neurônios para os 784 dados de entrada (teste tambem com 'sigmoid' no lugar de 'relu')
modelo.add(Dense(30, input_dim=784, kernel_initializer='normal', activation='relu')) #entrada e primeira camada oculta
modelo.add(Dense(30, kernel_initializer='normal', activation='relu')) #segunda camada oculta
modelo.add(Dense(10, kernel_initializer='normal', activation='softmax')) #saída

from keras.optimizers import SGD #iremos usar o gradiente descendente estocástico como otimizador
otimizador = SGD()

#compilando o modelo
modelo.compile(loss='categorical_crossentropy', optimizer=otimizador, metrics=['acc']) #estamos usando 'acc' como métrica de acurácia
# iremos salvar em uma variável para poder visualizar em gráafico mais adiante
#o batch_size estamos definindo 100 amostras para cada iteração (o padrão é 30), todas as 60mil amostras (60000x100 iterações) seria muito demorado
#após executar este comando abaixo, verifique as informações e repare o campo 'acc:' (acurácia), ele vai aumentando (melhorando o resultado)
historico = modelo.fit(x_treino_normalizado, y_treino_convertido, epochs=40, batch_size=100, validation_data=(x_teste_normalizado, y_teste_convertido), verbose=1)

#historico.history['acc'] #tambem podemos visualizar a acurácia da forma abaixo, pode ser feito para outros campos tambem

#iremos gerar um gráfico que compara a acurácia do treino com a acurácia do teste
acuracia_treino = historico.history['acc'] #será a variável Y do gráfico referente ao treino
acuracia_teste = historico.history['val_acc'] #será a variável Y do gráfico referente ao teste
epochs = range(1, len(acuracia_treino)+1) #precisa somar com mais 1 pq o range começa com zero, esta será nossa variável X
plt.plot(epochs, acuracia_treino, '-g', label='Acurácia dados de treino') #'-g' verificar documentação do matplotlib, é referente a cor e o formato da linha
plt.plot(epochs, acuracia_teste, '-b', label='Acurácia dados de teste')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acurácia')
plt.show()
