# Exemplos de funções da biblioteca openCV
# para facilitar a compreensão, execute cada bloco no jupyter notebook

import cv2   # Biblioteca do openCV
import numpy as np
from matplotlib import pyplot as plt

# Carregando a imagem
imagem = cv2.imread('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate.jpg')

# Verificando as dimensões da imagem, notar o formato (largura, altura e cor), temos uma imagem em 3 dimensões
# Cada dimensão é a imagem em uma cor e a sobre posição das 3 cores (dimensões) é que irá formar a imagem com as cores corretas
print(imagem.shape)

# Exibindo as informações de outra forma
print('Altura: {} pixels'.format(imagem.shape[0]))
print('Largura: {} pixels'.format(imagem.shape[1]))
print('Canais de cores: {}'.format(imagem.shape[2]))

cv2.imshow('Figura', imagem) # Mostrando o imagem com o openCV
cv2.waitKey(0)               # Aguarda uma tecla ser pressionada para fechar a imagem que foi aberta
cv2.destroyAllWindows()      # Fecha a janela aberta
# O openCV salva a imagem no formato BGR (e não RGB)
cv2.imwrite('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate-opencv.jpg', imagem)  # Salva uma cópia da imagem

# Visualizando a imagem no formato BGR do opencv
plt.imshow(imagem)
plt.show()

# Convertendo a imagem para o formato RGB no matplotlib
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
plt.imshow(imagem_rgb)
plt.show()

# Splitting & Merging
# Carregando a imagem e obtendo cada canal BGR separadamente, ou seja, estamos salvando a imagem 3x, uma em cada cor
# b= imagem em azul (blue), g= imagem em verde (green) e r= imagem em vermelho (red)
(b, g, r) = cv2.split(imagem)

zeros = np.zeros(imagem.shape[:2], dtype='uint8')  # Cria uma matriz de 2 dimensões (altura e largura) preenchida com zeros
cv2.imshow('Vermelho', cv2.merge([zeros, zeros, r]))
cv2.imshow('Verde', cv2.merge([zeros, g, zeros]))
cv2.imshow('Azul', cv2.merge([b, zeros, zeros]))
cv2.waitKey(0)

# Juntando novamente a imagem
merged = cv2.merge([b, g, r])
cv2.imshow('Merged', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convertendo para escala de cinza, é útil quando as cores não são importantes, reduz consideravelmente o tamanho do dataset
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imshow('Figura cinza', imagem_cinza)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Coletando as informações de um pixel localizado em determinada posição do imagem
(b, g, r) = imagem[238, 71]
print('O pixel localizado na posição [238, 71] possui as cores RGB: {}, {}, {}'.format(r, g, b))

# Mudando o valor de um pixel
imagem[238, 71] = (52, 229, 14) # está no formato BGR
(b, g, r) = imagem[238, 71]
print('O pixel localizado na posição [238, 71] possui as cores RGB: {}, {}, {}'.format(r, g, b))  # agora está no formato RGB
imagem_pintada = imagem.copy()   # Se não usar o comando .copy, a imagem original também será alterada
imagem_pintada[82:146, 607:723] = (52, 229, 14)
imagem_rgb = cv2.cvtColor(imagem_pintada, cv2.COLOR_BGR2RGB)
plt.imshow(imagem_rgb)
plt.show()

# Recortando a imagem
retangulo = imagem[265:330, 402:488]
imagem_rgb = cv2.cvtColor(retangulo, cv2.COLOR_BGR2RGB)
plt.imshow(imagem_rgb)
plt.show()

