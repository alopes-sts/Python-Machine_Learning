#Usando TensorFlow para um modelo de classificação
#Iremos usar o mesmo dataset já usado com o keras para comparar as diferenças, este dataset possui imagens digitalizadas de números escritos a mão, é usado para identificar números
#Exemplo de uso do tensorboard

#Após executar o programa, abrir a linha de comando e navegar até a pasta que foi definida na variável 'writer',
#Executar o tensorboard:  tensorboard --logdir=tensorboardFiles  (notar que o comando deve ser executado uma pasta antes da pasta onde foram salvos os arquivos
#Quando executar este comando, verificar qual a URL e porta que será informada para acessar os dados do tensorboard

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

########## Criando função para os dados do tensorboard, será criado um sumário das informações #########
def sumario_informacoes(matriz):
    with tf.name_scope('sumario_informacoes'):
        tf.summary.scalar('media', tf.reduce_mean(matriz))
        tf.summary.scalar('maximo', tf.reduce_max(matriz))
        tf.summary.scalar('minimo', tf.reduce_min(matriz))
        tf.summary.histogram('histograma', matriz)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True) #este método é especifico do dataset que já existe dentro do tensorflow, já usando o one hot encode para criar as novas colunas com os valores da variável target

#Parâmetros gerais, iremos utilizar como otimizador a função 'ADAMS', (testar com valores diferentes)
learning_rate = 0.001
epochs = 10
batch_size = 200

#Parâmetros da rede neural
n_entrada = 784 # Dados do dataset, cada imagens possui uma dimensão de 28x28 pixels, total de 784
n_camada_1 = 30 # neurônios da primeira camda oculta
n_camada_2 = 30 # neurônios da segunda camada oculta
n_classes = 10  # total de classes MNIST (digitos de 0 a 9)

#Variáveis preditoras e target (formato de placeholders)
#o modelo será treinado com lotes de 200 amostras cada, os valores de X e Y irão variar, por isso será usado placeholder
x = tf.placeholder(tf.float32, [None, n_entrada])
y = tf.placeholder(tf.float32, [None, n_classes])

########## Criando um sumário para as imagens de entrada ##########
redimensionada = tf.reshape(x, [-1, 28, 28, 1])        #Dados necessários para o summary.image (n_images, altrua, base, cor)
tf.summary.image('img_entrada', redimensionada, 1000)  #(nome, dados, steps)

#Definindo pesos para a primeira camada
w1 = tf.Variable(tf.random_normal([n_entrada, n_camada_1], stddev=0.05)) #irá receber valores aleatórios
sumario_informacoes(w1)  #enviando os valores de w1 para a função
#Bias da primeira camada
b1 = tf.Variable(tf.zeros([n_camada_1])) #é boa prática inicializar os bias com valores zerados
sumario_informacoes(b1)  #enviando os valores de b1 para a função
#Primeiro camada
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1)) #irá multiplicar os pesos pela entrada e somar com o bias, depois aplica a função relu

#Definindo pesos para a segunda camada
w2 = tf.Variable(tf.random_normal([n_camada_1, n_camada_2], stddev=0.05))
sumario_informacoes(w2)  #enviando os valores de w2 para a função
#Bias da segunda camada
b2 = tf.Variable(tf.zeros([n_camada_2]))
sumario_informacoes(b2)  #enviando os valores de b2 para a função
#Segunda camada
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,w2),b2))

#Definindo pesos para a última camada (output)
w_out = tf.Variable(tf.random_normal([n_camada_2, n_classes], stddev=0.05))
sumario_informacoes(w_out)  #enviando os valores de w_out para a função
#Bias da segunda de saída (output)
bias_out = tf.Variable(tf.zeros([n_classes]))
sumario_informacoes(bias_out)  #enviando os valores de w1 para a função
#Segunda camada
saida = tf.matmul(layer_2,w_out)  + bias_out

#Criando função de custo (softmax_cross_entropy_with_logits)
custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= saida, labels= y))
#Otimizador
otimizador = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(custo) #Usando ADAM

#Testando o modelo
predicoes = tf.equal(tf.argmax(saida, 1), tf.argmax(y, 1)) #vai comparar o valor da variável 'saida' com o valor da variavel 'y', notar que os valores da variável 'saida' que estiverem próximo de 1 será convertidos para 1
#Calculando a acurácia
acuracia = tf.reduce_mean(tf.cast(predicoes, tf.float32))  #primeiro converte para float e depois tira uma média
tf.summary.scalar('acuracia', acuracia)                    #Criando um sumário escalar para a acurácia
merged = tf.summary.merge_all()                            #Unindo todos os sumários para rodar tudo de uma vez só

# Inicializando as variáveis
init = tf.global_variables_initializer()
# Abrindo a sessão
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('C:/Onedrive/Curso_Machine_Learning/Modulo 3/tensorBoardFiles', sess.graph)   #local onde o arquivo do sumário será salvo
    # Ciclo de treinamento
    for epoch in range(epochs):
        custo_medio = 0.0
        total_batches = int(mnist.train.num_examples / batch_size)  # 'mnist.train.num_examples' retorna o total de amostras, este dataset já foi disponibilizado com os dados de treino e teste dentro da biblioteca do TensorFlow
        # Loop por todas as iterações (batches)
        for i in range(total_batches):
            batch_x, batch_y = mnist.train.next_batch(
                batch_size)  # 'mnist.train.next_batch' separa os dados em lotes (que aqui foi definido em 200) e cada passada do loop pega as próximas 200 amostras
            # Fit training usando batch data
            sess.run(otimizador, feed_dict={x: batch_x, y: batch_y})
            # Calculando o custo (loss) médio de um epoch completo (soma todos os custos de cada batch e divide pelo total de batches)
            custo_medio += sess.run(custo, feed_dict={x: batch_x, y: batch_y}) / total_batches
        # Rodando a acurácia em cada epoch
        acuracia_teste = sess.run(acuracia, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        sumarios = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        writer.add_summary(sumarios, epoch)

        # Visualizando os resultados após cada epoch
        print('Epoch: ', '{},'.format((epoch + 1)), 'Custo médio de treino = ', '{:.3f}'.format(custo_medio))
        print('Acurácia teste = ', '{:.3f}'.format(acuracia_teste))
    print('Treinamento concluido!')
    print('Acurácia do modelo:', acuracia.eval({x: mnist.test.images, y: mnist.test.labels}))


