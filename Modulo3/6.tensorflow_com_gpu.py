#Exemplo para verificar o tempo de processamento de GPU e CPU
#Existem tarefas que a CPU consegue ser mais rápida, pois o clock da CPU geralmente é muito maior que o da GPU

import numpy as np
import tensorflow as tf
import time

matriz_a = np.random.rand(10000, 10000).astype('float32')
matriz_b = np.random.rand(10000, 10000).astype('float32')

resultados = []

def mult_matrizes(matriz):
    return tf.matmul(matriz, matriz)

#Definindo o uso da GPU ou não
with tf.device('/gpu:0'):  #para usar cpu = /cpu:0 (o número '0' significa que estamos usando a gpu 0, se tivessemos outra, seria GPU:1 ou CPU:1
    a = tf.placeholder(tf.float32, [10000, 10000])
    b = tf.placeholder(tf.float32, [10000, 10000])
    resultados.append(mult_matrizes(a))
    resultados.append(mult_matrizes(b))

#Definindo o uso da CPU ou não
with tf.device('/cpu:0'):
    soma = tf.add_n(resultados)

inicio = time.time()

with tf.Session() as sess:
    sess.run(soma, {a:matriz_a, b:matriz_b})

fim = time.time()
print('Tempo em segundos: ', fim - inicio)

