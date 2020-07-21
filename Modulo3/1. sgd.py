import warnings
warnings.filterwarnings('ignore')  #ignora as mensagens de alerta que aparecem na tela
import pandas as pd
import keras
from sklearn.datasets import load_iris
from keras.utils import np_utils #função one hot encode do keras
from keras.utils import to_categorical


iris = load_iris()
x= pd.DataFrame(iris.data, columns=[iris.feature_names])
y= pd.Series(iris.target)

x.head()
y.head()
#precisamos criar um neurônio para cada classe, então cada classe da nossa variável target precisa ser convertida
#em uma matriz, precisamos criar uma nova coluna para cada classe com o one hot encode. Lembrando que estas novas colunas
#terão somente os valores de '0' ou '1', que é o que precisamos para trabalhar em uma rede neural.
y.value_counts()

#usando a função de one hot encode do keras (matriz de classes)
y_convertido = np_utils.to_categorical(y)
print(y_convertido)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y_convertido, test_size = 0.3)
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

modelo=Sequential()
#O comando '.add' irá contruir uma rede neural adicionando camadas.
#O primeiro argumento passado é o número de neurônios de cada camada ('10'), teremos 10 neurônios na camada oculta
#O segundo argumento é a quantidade de neurônios de entrada (temos 4 classes na nossa variável target) ('input_dim')
#O terceiro argumento é o modelo de inicialização das variáveis
#E o quarto é a função de ativação de cada camada.
#sigmoid classifica a saída em zero ou um, mas não representa uma probabilidade (a soma de todas as saídas pode ser maior que 1 (100%)
#podemos usar a função 'softmax' para que as saídas representem as probabilidade corretas, a soma de toas as saídas será = 1
#outra opção é usar 'relu', a maioria dos modelos atuais usam relu (será visto em outro exemplo, mas ele pode ser testado aqui para ver os resultados)
modelo.add(Dense(10, input_dim=4, kernel_initializer='normal', activation='sigmoid')) #'dense'=rede neural densa (totalmente conectada)
modelo.add(Dense(3, kernel_initializer='normal', activation='sigmoid')) #saída com 3 neurônios (3 classes)

otimizador = SGD()  #Gradiente descendente estocástico
#estamos usando uma função de custo = 'categorical_crossentropy'
modelo.compile(loss='categorical_crossentropy', optimizer=otimizador, metrics=['acc']) #acc= métrica de acurácia
#epochs = número de iterações, batch_size = tamanho do lote de treino (estamos usando todo o dataset neste exemplo), verbose=1 é para mostrar na tela informação de cada iteração
modelo.fit(x_treino, y_treino, epochs=1000, batch_size=105, validation_data=(x_teste, y_teste), verbose=1)

#agora que o modelo foi treinado, podemos fazer previsões em novos conjuntos de dados
predicoes = modelo.predict(x_test)
print(predicoes)
#podemos melhorar a visualização da saída, visualizando somente 2 casas decimais
import numpy as np
np.set_printoptions(formatter={'float': lambda x: '{0:0.2f}'.format(x)})
print(predicoes) #irá mostrar a probabilidade de cada amostra pertencer aquela classe

