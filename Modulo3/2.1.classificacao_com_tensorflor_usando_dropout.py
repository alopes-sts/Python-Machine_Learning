
#Usando TensorFlow para um modelo de classificação
#Iremos usar o mesmo dataset já usado com o keras para comparar as diferenças, este dataset possui imagens digitalizadas de números escritos a mão, é usado para identificar números
#Apesar do dataset ser o mesmo, este importado do TensorFlow possui muito mais amostras
#O TensorFlow é mais complexo, porem é mais performático e permite mais ajustes
### Neste modelo estamos adicionado a função 'dropout' (método de regularização para evitar overfiting)###
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
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

#Definindo pesos para a primeira camada
w1 = tf.Variable(tf.random_normal([n_entrada, n_camada_1], stddev=0.05)) #irá receber valores aleatórios
#Bias da primeira camada
b1 = tf.Variable(tf.zeros([n_camada_1])) #é boa prática inicializar os bias com valores zerados
#Primeiro camada
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1)) #irá multiplicar os pesos pela entrada e somar com o bias, depois aplica a função relu
# Criando camada com o Dropout
drop1 = tf.nn.dropout(layer_1, rate=0.2) #estamos retirando 20% dos dados

#Definindo pesos para a segunda camada
w2 = tf.Variable(tf.random_normal([n_camada_1, n_camada_2], stddev=0.05))
#Bias da segunda camada
b2 = tf.Variable(tf.zeros([n_camada_2]))
#Segunda camada
layer_2 = tf.nn.relu(tf.add(tf.matmul(drop1,w2),b2)) #atentar que aqui deve ser colocado o resultado do 'dropout'
# Criando camada com o Dropout
drop2 = tf.nn.dropout(layer_2, rate=0.2) #estamos retirando 20% dos dados

#Definindo pesos para a última camada (output)
w_out = tf.Variable(tf.random_normal([n_camada_2, n_classes], stddev=0.05))
#Bias da segunda de saída (output)
bias_out = tf.Variable(tf.zeros([n_classes]))
#Segunda camada
saida = tf.matmul(drop2,w_out)  + bias_out #atentar que aqui deve ser colocado o resultado do 'dropout'

#Criando função de custo (softmax_cross_entropy_with_logits)
custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= saida, labels= y))
#Otimizador
otimizador = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(custo) #Usando ADAM

#Testando o modelo
predicoes = tf.equal(tf.argmax(saida, 1), tf.argmax(y, 1)) #vai comparar o valor da variável 'saida' com o valor da variavel 'y', notar que os valores da variável 'saida' que estiverem próximo de 1 será convertidos para 1
#Calculando a acurácia
acuracia = tf.reduce_mean(tf.cast(predicoes, tf.float32)) #primeiro converte para float e depois tira uma média

#Inicializando as variáveis
init = tf.global_variables_initializer()
#Abrindo a sessão
with tf.Session() as sess:
    sess.run(init)
    #Ciclo de treinamento
    for epoch in range(epochs):
        custo_medio = 0.0
        total_batches = int(mnist.train.num_examples / batch_size) #'mnist.train.num_examples' retorna o total de amostras, este dataset já foi disponibilizado com os dados de treino e teste dentro da biblioteca do TensorFlow
        #Loop por todas as iterações (batches)
        for i in range(total_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size) #'mnist.train.next_batch' separa os dados em lotes (que aqui foi definido em 200) e cada passada do loop pega as próximas 200 amostras
            #Fit training usando batch data
            sess.run(otimizador, feed_dict={x: batch_x, y: batch_y})
            #Calculando o custo (loss) médio de um epoch completo (soma todos os custos de cada batch e divide pelo total de batches)
            custo_medio += sess.run(custo, feed_dict={x: batch_x, y: batch_y}) / total_batches
        #Rodando a acurácia em cada epoch
        acuracia_teste = sess.run(acuracia, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        #Visualizando os resultados após cada epoch
        print ('Epoch: ', '{},'.format((epoch + 1)), 'Custo médio de treino = ', '{:.3f}'.format(custo_medio))
        print ('Acurácia teste = ', '{:.3f}'.format(acuracia_teste))
    print('Treinamento concluido!')
    print('Acuracia do modelo:', acuracia.eval({x: mnist.test.images, y: mnist.test.labels}))
            

