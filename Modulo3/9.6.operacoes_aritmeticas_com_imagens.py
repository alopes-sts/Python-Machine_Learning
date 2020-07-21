# operações aritméticas com imagens, controlar brilho, operações lógicas, máscaras
# para facilitar a compreensão, execute no jupyter notebook

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregando a imagem
imagem = cv2.imread('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate.jpg')

matriz = np.ones(imagem.shape, dtype='uint8') * 100  # Cria uma matriz da mesma dimensão da imagem, com cada pixel = 100
soma = cv2.add(imagem, matriz)
plt.imshow(soma)
plt.show()

# Operações lógicas bitwise (AND, OR, XOR, NOT)
fundo = np.zeros((300, 300), dtype='uint8')
cv2.rectangle(fundo, (25, 25), (275, 275), 255, -1)

# Desenhando um círculo
fundo2 = np.zeros((300, 300), dtype='uint8')
cv2.circle(fundo2, (150, 150), 150, 255, -1)

bitwise_AND = cv2.bitwise_and(fundo, fundo2) # aplica a operação AND
bitwise_OR = cv2.bitwise_or(fundo, fundo2) # aplica a operação OR
bitwise_XOR = cv2.bitwise_xor(fundo, fundo2) # aplica a operação XOR
bitwise_NOT = cv2.bitwise_not(fundo2) # aplica a operação NOT

plt.imshow(fundo)
plt.show()
plt.imshow(fundo2)
plt.show()

# Juntando as 2 figuras acima
plt.imshow(bitwise_AND)  # Testar com OR, XOR, etc
plt.show()

# Masking
mask = np.zeros(imagem.shape[:2], dtype='uint8')
(cX, cY) = (imagem.shape[1]//2, imagem.shape[0]//2)
cv2.rectangle(mask, (cX - 150, cY - 80), (cX + 150, cY + 80), 255, -1)

# Máscara retangulaar
primeira_mascara = cv2.bitwise_and(imagem, imagem, mask=mask)

# Máscara circular
mask2 = np.zeros(imagem.shape[:2], dtype='uint8')
cv2.circle(mask2, (cX, cY), 120, 255, -1)
segunda_mascara = cv2.bitwise_and(imagem, imagem, mask=mask2)

plt.imshow(primeira_mascara)
plt.show()
plt.imshow(segunda_mascara)
plt.show()
