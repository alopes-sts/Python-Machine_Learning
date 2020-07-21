import warnings
warnings.filterwarnings('ignore')
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import regularizers
from keras.optimizers import Nadam
import matplotlib.pyplot as plt

(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()
y_treino_convertido = np_utils.to_categorical(y_treino) #converte a coluna de valroes em uma matriz de classes (one hot encode)
y_teste_convertido = np_utils.to_categorical(y_teste)
x_treino_remodelado = x_treino.reshape((60000, 784))
x_teste_remodelado = x_teste.reshape((10000, 784))
x_treino_normalizado = x_treino_remodelado.astype('float32') / 255
x_teste_normalizado = x_teste_remodelado.astype('float32') / 255
modelo = Sequential()
#30 = primeira camada oculta, 784 neurônios na camada de entrada (são os dados deste dataset, a matriz de 28x28 pixels),
modelo.add(Dense(30, input_dim=784, kernel_initializer='normal', activation='relu')) #primeira camada oculta
modelo.add(Dense(30, kernel_initializer='normal', activation='relu')) #segunda camada oculta
modelo.add(Dense(10, kernel_initializer='normal', activation='softmax')) #saída
#iremos usar o Nadam como otimizador
#fazer os testes alterando as funções e verificar o resultado mais abaixo gerado na variável 'historico' e no gráfico
otimizador = Nadam()   #é aqui a única diferença em relação ao executado no modelo anterior
modelo.compile(loss='categorical_crossentropy', optimizer=otimizador, metrics=['acc']) #estamos usando 'acc' como métrica de acurácia

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
#este gráfico mostra os dados de treino terminando abaixo dos dados de teste, isso significa que o modelo executado acima ainda está na fase de 'underfiting' e podem ser melhorados

