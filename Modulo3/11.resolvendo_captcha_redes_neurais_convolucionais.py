# Exemplo de uso de rede neural convolucional (CNN) para identificar figuras de captcha
# Execute o programa pelo jupyter notebook em blocos para facilitar a visualização

import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

nomes_amostras = os.listdir('C:/Onedrive/Curso_Machine_Learning/Modulo 3/captcha')  # cria uma lista de todos os arquivos
print(nomes_amostras)

imagens = []
for i in nomes_amostras:
    endereco = 'C:/Onedrive/Curso_Machine_Learning/Modulo 3/captcha/' + i  # Adiciona o nome ao endereço para formar o diretório completo
    img = cv2.imread(endereco)                                            # lê cada imagem
    imagens.append(img)                                                   # adiciona na lista

#print(imagens[0].shape)  # verificando o shape de uma imagem, notar que a terceira coluna (3) é a quantidade de camdas de cores (as imagens estão em formato RGB)

# mostrando uma imagem
cv2.imshow('Figura', imagens[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Checando o formato das imagens em BGR
(b, g, r) = cv2.split(imagens[0])
zeros = np.zeros(imagens[0].shape[:2], dtype='uint8')
cv2.imshow('Vermelho', cv2.merge([zeros, zeros, r]))
cv2.imshow('Verde', cv2.merge([zeros, g, zeros]))
cv2.imshow('Azul', cv2.merge([b, zeros, zeros]))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convertendo para Grayscale
imagens_cinzas = []
for i in imagens:
    img_cinza = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    imagens_cinzas.append(img_cinza)

print(len(imagens_cinzas))  # para verificar quantidade de imagens, deve ter 1070 imagens

#imagens_cinzas[0].shape  # verificando o formato das imagens, notar que a terceira coluna não aparece mais, pois foi retirado o BGR

# verificando uma imagem, que deve estar em escala de cinza
cv2.imshow('Figura', imagens_cinzas[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Notar que só temos uma amostra de cada imagem, dificultando o aprendizado do modelo
# Cada imagem deve ser quebrada em 5 partes, uma para cada caracter
# Visualizando cada caracter separadamente
cv2.imshow('Figura', imagens_cinzas[0][12:50, 30:50])  # estes valores foram testados até conseguir criar uma imagem separada para o caracter
cv2.waitKey(0)
cv2.destroyAllWindows()

# Temos um dataset de (1070, 50, 200), 1070 imagens de 50 x 200 pixels
# Iremos transformar este dataset em (5350, 38, 20), ou seja, 5350 imagens de 38 x 20 pixels

# Separando cada caracter em uma imagem distinta
x_novo = np.zeros((len(nomes_amostras) * 5, 38, 20))  # (n_amostras, (dimensão))
inicio_x, inicio_y, w, h, = 30, 12, 20, 50  # ponto inicial (x, y), largura, ponto final y
for i in range(len(imagens_cinzas)):
    px = inicio_x
    for j in range(5):  # para cada caracter da imagem
        x_novo[i*5+j] = imagens_cinzas[i][inicio_y:h, px:px+w]  # recortando os caracteres
        px += w
#print(x_novo.shape)  # Visualizando o formato do dataset com as imagens separadas

# Exibindo uma imagem em escala de cinza e os caracteres recortados
cv2.imshow('Figura6', imagens_cinzas[1])
cv2.imshow('Figura1', x_novo[5])
cv2.imshow('Figura2', x_novo[6])
cv2.imshow('Figura3', x_novo[7])
cv2.imshow('Figura4', x_novo[8])
cv2.imshow('Figura5', x_novo[9])
cv2.waitKey(0)
cv2.destroyAllWindows()

print(x_novo[0])  # verificando o formato da variável, temos valores de 0 a 255, será necessário normalizar estes dados, é necessários que os valores estejam entre 0 e 1

# Preparando as imagens para a rede neural (normalizando e mudando o shape)
x = np.zeros((x_novo.shape[0], x_novo.shape[1], x_novo.shape[2], 1))  # n_amostras, (dimensao), gray_scale
for i in range(x_novo.shape[0]):
    norm = x_novo[i]/250  # normaliza
    img = np.reshape(x_novo[i], (x_novo.shape[1], x_novo.shape[2], 1))  # cria uma dimensão extra para indicar gray_scale
    x[i] = img  # adiciona cada uma das imagens no array x

#x.shape  # verificando o formato da variável, deve ter 4 dimensões (quantidade de amostras, largura, altura e no formato gray scale(1)

# Criando a variável target
y_atual = nomes_amostras
print(y_atual[0][0:5])  # verificando o nome dos arquivos, precisamos remover a extensão do arquivo para que fique igual ao que foi impresso aqui
# Removendo o .png no final de cada nome de imagem
for i in range(len(y_atual)):
    y_atual[i] = y_atual[i][0:5]
print(y_atual[0])  # confirmando que ficou no formato desejado
print(len(y_atual))  # apenas confirmando o número de amostras, os números ainda estão todos no mesmo arquivo

# Precisamos separar cada um dos valores de y_atual, para ficar com uma única lista de 5350 valores
y = [None] * x.shape[0]  # define a dimensão final da lista
for i in range(len(y_atual)):
    for j in range(5):
        y[i*5+j] = y_atual[i][j]  # recorta os caracteres, após isso, teremos todos os caracteres recortados

print(len(y))  # confirmando o tamanho que ficou a variável

# Necessário fazer one hot encode na variável y, antes, precisamos definir quais são as possíveis classes, agrupando todos os simbolos possíveis em um única string
# Vamos criar uma variável que contem todos os caracteres do alfabeto + números
# Esta variável se tornará colunas com valores 0, e 1 na coluna referente ao caracter desejado
# Por exemplo, a letra 'a', é o primeiro caracter do alfabeto, será a primeira posição da variável, então teremos o
# valor 1 na primeira posição e o restante será preenchido com valores '0'
#     abcdefghijklmnopqrstuvxz0123456789
# a = 1000000000000000000000000000000000
# b = 0100000000000000000000000000000000
# Dessa forma, cada imagem vai receber todas essas colunas e a coluna referente ao caracter terá o valor '1'
import string

simbolos = string.ascii_lowercase + '0123456789'
print(simbolos)     # visualizando o conteúdo que foi criado na variável
simbolos.find('f')  # retorna o menor indice do caracter procurado

y_final = np.zeros((len(y), 36))  # Define a dimensão final (5350 amostras e 36 classes)
for i in range(len(y)):
    caracter = y[i]
    loc_caracter = simbolos.find(caracter)  # encontra a localização correspondente desse caracter
    y_final[i, loc_caracter] = 1  # atribui qual a classe de cada elemento contido em y[i]. Teremos valro 1 na coluna certa e zero

print(y_final[0])  # Verificando o array criado com o one hot encode
#y_final.shape  # confirmando que temos 5350 amostras e 36 colunas

########## Criando o modelo ##########
# Separando dados entre treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y_final, test_size=0.2, random_state=8)
#x_treino.shape

# Parâmetros gerais
learning_rate = 0.001
epochs = 200
batch_size = 428

# Variáveis preditoras e target (formato de placeholders)
x = tf.placeholder(tf.float32, [None, 38, 20,1])
y = tf.placeholder(tf.float32, [None, 36])

def conv2d(entrada, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(entrada, w, strides=[1, 1, 1, 1], padding='VALID'), b))

def max_pool(entrada, k):
    return tf.nn.max_pool(entrada, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

# Iniciando os pesos
w1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.05))   # kernel 3x3, 1 imagem para varrer (entrada), 32 features maps (saída)
w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.05))  # 3x3, 32 imagens para varrer (feature maps do layer 1), 64 features
w_denso_1 = tf.Variable(tf.random_normal([16*7*64, 80], stddev=0.035))  # 80 neurônios recebem cada um 16*7*64 pesos (saída anterior)
w_out = tf.Variable(tf.random_normal([80, 36], stddev=0.05))  # 36 neurônios (saída) recebem cda um 80 pesos (entrada)

# Iniciando os bias
b1 = tf.Variable(tf.zeros([32]))         # 1 bias para cada feature map da camada 1
b2 = tf.Variable(tf.zeros([64]))         # 1 bias para cada feature map da camada 2
b_denso_1 = tf.Variable(tf.zeros([80]))  # 1 bias para cada neurônio da camada densa
b_out = tf.Variable(tf.zeros([36]))      # 1 bias para cada neurônio correspondente a uma classe

# Camada CNN 1
conv1 = conv2d(x, w1, b1)
pool1 = max_pool(conv1, k=2)
#drop1 = tf.nn.dropout(pool1, rate=0.2) # poderiamos usar dropout, mas o resultado não foi bom

# Camada CNN 2
conv2 = conv2d(pool1, w2, b2)
#pool2 = max_pool(conv2, k=2)
#drop2 = tf.nn.dropout(conv2, rate=0.2)

# Camada densa oculta
drop2_redimensionada = tf.reshape(conv2, shape=[-1, w_denso_1.get_shape().as_list()[0]])  # Aplica flatten antes de ligar na densa
densa = tf.nn.relu(tf.add(tf.matmul(drop2_redimensionada, w_denso_1), b_denso_1))
#drop_densa = tf.nn.dropout(densa, rate=0.2)

# Camada de saída
out = tf.add(tf.matmul(densa, w_out), b_out)

# Custo e otimizador
custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
otimizador = tf.train.AdamOptimizer(learning_rate=learning_rate). minimize(custo)

# Avaliando o modelo
acertos = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
acuracia = tf.reduce_mean(tf.cast(acertos, tf.float32))

# Parâmetros do plot
historic_acc = []
historic_epochs = []

# Variáveis que serão utilizadas no ciclo de treinamento
tamanho_treino = x_treino.shape[0]
total_batches = tamanho_treino/batch_size

# Inicializando as variáveis
init = tf.global_variables_initializer()

# Sessão
with tf.Session() as sess:
    sess.run(init)
    # Ciclo de treinamento
    for epoch in range(epochs):
        custo_medio = 0.0
        # loop por todas as iterações (batches)
        for i in range(0, tamanho_treino, batch_size):  # avança dando passos do tamanho de um batch
            batch_x = x_treino[i:i + batch_size]
            batch_y = y_treino[i:i + batch_size]
            # rodando o otimizador com os batches de treino
            sess.run(otimizador, feed_dict= {x: batch_x, y: batch_y})
            # Computando o custo médio de um epoch completo (soma todos os custos e divide pelo total de batches)
            custo_medio += sess.run(custo, feed_dict={x: batch_x, y: batch_y})/total_batches
        # Rodando a acurácia em cada epoch
        acuracia_teste = sess.run(acuracia, feed_dict={x: x_teste, y: y_teste})
        # Mostrando os resultados após cada epoch
        print('Epoch: ', '{},'.format((epoch + 1)), 'Custo médio treino= ', '{:.3f}'.format(custo_medio))
        print('Acurácia teste= ', '{:.3f}'.format(acuracia_teste))
        historic_acc.append(acuracia_teste)
        historic_epochs.append(epoch+1)

    print('Treinamento Concluido')
    print('Acurácia do model: ', acuracia.eval({x: x_teste, y: y_teste}))

plt,plot(historic_epochs,historic_acc, 'o', label='MLP - Fase de Treinamento')
plt.ylabel('Custo')
plt.xlabel('Epoch')
plt.legend()
plt.show()

















