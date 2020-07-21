from sklearn.datasets import load_iris
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sklearn.utils
from keras.utils import np_utils

iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)
x_shuffle, y_shuffle = sklearn.utils.shuffle(x, y) #Como boa prática, fazer um embaralhamento dos dados antes de iniciar a rede neural

print('Verificando formato da variável antes do one hot encode: ', y_shuffle.shape) #verificando o formato da variável
y_one_hot = np_utils.to_categorical(y_shuffle)
print('Verificando formato da variável depois do one hot encode: ', y_one_hot.shape) #após aplicar o one hot encode, temos mais três variáveis

#Separando os dados de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x_shuffle, y_one_hot, test_size=0.3, random_state=5)

#Visualizando a quantidade de amostras de treino
print(x_treino.shape)
print(y_treino.shape)

#Parâmetros Gerais
learning_rate = 0.001
epochs = 500  #Visualizando a quantidade de amostras (x_treino.shape) acima, podemos notar que o dataset é muito pequeno, então podemos colocar uma quantidade maior de épocas para que tenhamos muitas iterações
batch_size = 50 #O batch size não pode ser maior que o dataset

#Parâmetros da rede neural
n_entrada = 4    #número de features na entrada da rede neural, visualizando o dataset com o comando 'x_treino.shape', podemos ver que o dataset tem 4 features
n_camada_1 = 10  #total de neurônios da primeira camada oculta
n_classes = 3    #total de classes de saída, visualizando o dataset como comando 'y_treino.shape' podemos visualizar que o dataset tem classes

#Variáveis preditoras e target (formato de placeholders)
x = tf.placeholder(tf.float32, [None, n_entrada])
y = tf.placeholder(tf.float32, [None, n_classes])

#Primeira camada
w1 = tf.Variable(tf.random_normal([n_entrada, n_camada_1], stddev=0.05)) #Pesos da primeira camada
b1 = tf.Variable(tf.zeros([n_camada_1])) #Bias da primeira camada
layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1)) #irá multiplicar os pesos pela entrada e somar com o bias, aplicando a funçao 'relu'

w_out = tf.Variable(tf.random_normal([n_camada_1, n_classes], stddev=0.05))  #Pesos da camada de saída (output)
bias_out = tf.Variable(tf.zeros([n_classes])) #Bias da camada de saída (output)
saida = tf.matmul(layer_1, w_out) + bias_out #Camada de saída (output)

#Criando a função de custo e o otimizador
custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=saida, labels=y))
otimizador = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(custo)

#Testando o modelo e calculo de acurácia
predicoes = tf.equal(tf.argmax(saida, 1), tf.argmax(y, 1))
acuracia = tf.reduce_mean(tf.cast(predicoes, tf.float32))

#Variáveis que serão utilizadas no ciclo de treinamento
tamanho_treino = len(x_treino.index)
total_batches = tamanho_treino/batch_size

#inicializando as variáveis
init = tf.global_variables_initializer()

#Sessão
with tf.Session() as sess:
    sess.run(init)
    #Ciclo de treinamento
    for epoch in range(epochs):
        custo_medio = 0.0
        #Loop por todas as iterações (batches)
        for i in range(0, tamanho_treino, batch_size): #avança dando passos do tamanho de um batch
            batch_x = x_treino[i:i+batch_size]
            batch_y = y_treino[i:i+batch_size]
            sess.run(otimizador, feed_dict={x: batch_x, y: batch_y}) #Rodando o otimizador com os batches de treino
            custo_medio += sess.run(custo, feed_dict={x: batch_x, y: batch_y})/total_batches #Cálculando o custo médio de um epoch completo (soma todos os custos e divide pelo total de batches)
        acuracia_teste = sess.run(acuracia, feed_dict={x: x_teste, y: y_teste}) #Verificando a acurácia para cada epoch
        #Mostrando os resultados após cada epoch
        print('Epoch: ', '{},'.format((epoch + 1 )), 'Custo médio treino= ', '{:.3f}'.format(custo_medio))
        print('Acurácia de teste= ', '{:.3f}'.format(acuracia_teste))
    print('Treinamento concluído')
    print('Acurácia do modelo: ', acuracia.eval({x: x_teste, y: y_teste}))

