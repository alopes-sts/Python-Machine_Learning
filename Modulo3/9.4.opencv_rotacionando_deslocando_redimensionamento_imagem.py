# Primeiro a imagem será rotacionada com o openCV
# Depois a imagem será deslocada para um dos lados
# Depois será redimensionada
# para facilitar a compreensão, execute no jupyter notebook

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregando a imagem
imagem = cv2.imread('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate.jpg')

# Obtendo as dimensões da imagem e seu centro
(altura, largura) = imagem.shape[:2]
centro = (largura//2, altura//2)

# Rotacionando a imagem em 45 graus
parametros_rot = cv2.getRotationMatrix2D(centro, 45, 1.0) # (centro, ângulo de rotação, escala) - escala=zoom
rotacionado = cv2.warpAffine(imagem, parametros_rot, (largura, altura)) # (imagem, parâmetros, dimensões_img_saída)
plt.imshow(rotacionado)
plt.show()

# Deslocando a imagem
(altura, largura) = imagem.shape[:2] #obtem as dimensões da imagem
parametros_shift = np.float32([[1, 0, -46], [0, 1, 120]]) # movendo 46 pixels para a esquerda e 120 pixels para baixo
deslocado = cv2.warpAffine(imagem, parametros_shift, (largura, altura))
plt.imshow(deslocado)
plt.show()

# Redimensionamento
(altura, largura) = imagem.shape[:2] # Obtendo as dimensões da imagem
nova_largura = 180
proporcao = altura / largura
nova_dim = (nova_largura, int(nova_largura * proporcao)) #Sempre converter para números inteiros

# Executando o redimensionamento
# Existem 4 funções para interpolação: INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4.
# INTER_LINEAR é a mais rápida mas tem menos qualidade, as outras tem resultados parecidos
redimensionada = cv2.resize(imagem, nova_dim, interpolation=cv2.INTER_AREA)
plt.imshow(redimensionada)
plt.show()


