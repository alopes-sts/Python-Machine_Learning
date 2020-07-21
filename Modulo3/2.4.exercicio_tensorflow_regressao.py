#Exemplo de rede neural com tensorflow para problemas de regressão logistica
#usando um dataset com dados de admissão de alunos
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 3/Admission_Predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)

#Separando variáveis preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

print(y.shape) #Verificando o formato da variável y, notar que um vetor sem nenhum coluna e precisa ser feito um reshape, precisamos de pelo menos uma coluna

y_remodelado = y.values.reshape(400,1) #Fazendo o reshape
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y_remodelado, test_size=0.3, shuffle=True)

print('Variável x de treino: ', x_treino.shape)
print('Variável y de treino: ', y_treino.shape)

#Criando a rede neural
#Parâmetros gerais e da rede neural
learning_rate = 0.001
epochs = 500
batch_size = 50  #Estamos usando 50, a quantidade de amostras que foi visualizada em x_treino foi de 280 amostras (batch_size tem que ser menor qua a quantidade de amostras)
n_entrada = 7    #Dados de entrada (n. de features), estamos usando a quantidade de colunas que foi visualizada em x_treino
n_camada_1 = 10  #Quantidade de neurônios da primeira camada oculta
n_classes = 1    #Total de classes de saída (problema de regressão só tem um neurônio de saída)

#Transformando as variáveis preditoras e target em placeholders
x = tf.placeholder(tf.float32, [None, n_entrada])
y = tf.placeholder(tf.float32, [None, n_classes])
print(x)  #Apenas para visualização do placeholder
print(y)

#Criando a rede neural
w1 = tf.Variable(tf.random_normal([n_entrada, n_camada_1], stddev=0.05))    #Definindo pesos para a primeira camada
b1 = tf.Variable(tf.zeros([n_camada_1]))                                    #Bias da primeira camada
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1))                            #multiplica os pesos pela entrada e soma o bias, aplicando a função relu

w_out = tf.Variable(tf.random_normal([n_camada_1, n_classes], stddev=0.05)) #Definindo pesos para a camada de saída
bias_out =tf.Variable(tf.zeros([n_classes]))                                #Bias da camada de saída (output)
saida = tf.matmul(layer_1, w_out) + bias_out                                #Camada de saída (output)

custo = tf.reduce_mean(tf.losses.mean_squared_error(predictions=saida, labels=y))  #Funcao de custo (foi usado o 'Mean Squared Error')
otimizador = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(custo)   #Otimizador

#Variáveis para o ciclo de treinamento
tamanho_treino = len(x_treino.index)
total_batches = tamanho_treino/batch_size
init = tf.global_variables_initializer()  #Inicializando as variáveis

#Sessão
with tf.Session() as sess:
    sess.run(init)
    #Ciclo de treinamento
    for epoch in range(epochs):
        custo_medio = 0.0
        #Loop por todas as iterações (batches)
        for i in range(0, tamanho_treino, batch_size): #Avança dando passos de tamanho de um batch
            batch_x = x_treino[i:i + batch_size]
            batch_y = y_treino[i:i + batch_size]
            sess.run(otimizador, feed_dict= {x: batch_x, y: batch_y}) #Rodando o otimizador com os batches de treino
            custo_medio += sess.run(custo, feed_dict={x: batch_x, y: batch_y}) / total_batches

        mse = sess.run(custo, feed_dict={x: x_teste, y: y_teste})  #Rodando o MSE em cada epoch (Erro médio quadrático, pois é um problema de regressão)

        #Visualizando os resultados após cada epoch
        print('Epoch: ', '{},'.format((epoch + 1)), 'Custo médio treino= ', '{:.3f}'.format(custo_medio))
        print('MSE teste= ', '{:.3f}'.format(mse))
    print('Treinamento Finalizado')
    print('MSE final: ', custo.eval({x: x_teste, y: y_teste}))






