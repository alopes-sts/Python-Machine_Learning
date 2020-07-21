# Exemplo de rede convolucional usando TensorFlow
# Usando um dataset nativo do tensorflow com imagens de números escritos a mão
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Parâmetros Gerais
learnig_rate = 0.001
epochs = 10
batch_size = 200

# Parâmetros da rede neural
n_entrada = 784 # Dados de entrada MNIST (imagens de dimensão 28x28)
n_classes = 10  # Total de classes MNIST (digítos de 0 a 9)

# Variáveis preditoras e target (formato de placeholders)
x = tf.placeholder(tf.float32, [None, n_entrada])
y = tf.placeholder(tf.float32, [None, n_classes])

def conv2d(entrada, w, b):
    # Estamos adicionando um feature map multiplicando a entrad pelos pesos (w) e depois somamos bias e aplicamos relu
    # strides = [1, deslocamento_horizontal, deslocamento_vertical, 1]
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(entrada, w, strides=[1, 1, 1, 1], padding='VALID'), b))

def max_pool(entrada, k):
    # ksize = tamanho da janel que irá realizar o pooling
    return tf.nn.max_pool(entrada, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


# O modelo 'Conv2D' precisa que na entrada tenha 4 dimensões: amostras, altrua, largura e padrão de cores
x_redimensionado = tf.reshape(x, shape=[-1, 28, 28, 1])

# Iniciando os pesos
w1 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.05))         # kernel 5x5, 1 imagem para varrer (entrada), 32 features maps (saída)
w2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.05))        # 5x5, 32 imagens para varrer (features maps do layer 1), 64 features maps
w_denso_1 = tf.Variable(tf.random_normal([4*4*64, 80], stddev=0.0063)) # 80 neurônios se conectam a 1024 neurônios (saída anterior)
w_out = tf.Variable(tf.random_normal([80, n_classes], stddev=0.04))    # 10 neurônios (saída) se conectam a 80 neurônios (entrada anterior)

# Iniciando os bias
b1 = tf.Variable(tf.zeros([32]))           #1 bias para cada feature map da camada 1
b2 = tf.Variable(tf.zeros([64]))           #1 bias para cada feature map da camada 2
b_denso_1 = tf.Variable(tf.zeros([80]))    # 1 bias para cada neurônio da camada densa
b_out = tf.Variable(tf.zeros([n_classes])) # 1 bias para cada neurônio correspondente a uma classe

# Camada CNN 1
conv1 = conv2d(x_redimensionado, w1, b1)
pool1 = max_pool(conv1, k=2)

# Camada CNN 2
conv2 = conv2d(pool1, w2, b2)
pool2 = max_pool(conv2, k=2)

# Camada densa oculta
drop2_redimensionada = tf.reshape(pool2, shape=[-1, w_denso_1.get_shape().as_list()[0]]) # aplica Flatten antes de ligar na densa
densa = tf.nn.relu(tf.add(tf.matmul(drop2_redimensionada, w_denso_1), b_denso_1))
drop_densa = tf.nn.dropout(densa, keep_prob=1)

# Camada de saída
out = tf.add(tf.matmul(drop_densa, w_out), b_out)

# Inicializando as variáveis
init = tf.global_variables_initializer()

# Custo e Otimizador
custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
otimizador = tf.train.AdamOptimizer(learning_rate=learnig_rate).minimize(custo)

# Avaliando o modelo
acertos = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))  # Equal checa quantos elementos são iguais
acuracia = tf.reduce_mean(tf.cast(acertos, tf.float32)) # tf.cast transforma um tensro em outro tipo, neste caso para float32

# Inicializando as variáveis
init = tf.global_variables_initializer()

# Abrindo a Sessão
with tf.Session() as sess:
    sess.run(init)
    # Ciclo de treinamento
    for epoch in range(epochs):
        custo_medio = 0.0
        total_batches = int(mnist.train.num_examples / batch_size)
        # loop  por todas as iterações (batches)
        for i in range(total_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Fit training usando batch data
            sess.run(otimizador, feed_dict={x: batch_x, y: batch_y})
            # Computando o custo (loss) médio de um epoch completo (soma todos os custos de cada batch e divide pelo total de batches)
            custo_medio += sess.run(custo, feed_dict={x: batch_x, y: batch_y}) / total_batches
        # Rodando a acurácia em cada epoch
        acuracia_teste = sess.run(acuracia, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        # mostrando os resultados após cada epoch
        print('Epoch: ', '{},'.format((epoch + 1)), 'Custo médio treino= ', '{:.3f}'.format(custo_medio))
        print('Acurácia teste= ', '{:.3f}'.format(acuracia_teste))
    print('treinamento concluido')
    print('Acurácia do modelo: ', acuracia.eval({x: mnist.test.images, y: mnist.test.labels}))
