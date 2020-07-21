# Usando uma figura como background no openCV e adicionado uma figura por cima
# para facilitar a compreensão, execute no jupyter notebook

import cv2
from matplotlib import pyplot as plt

# Carregando a imagem
imagem = cv2.imread('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate.jpg')

# Desenhando textos por cima das imagens
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
fonte = cv2.FONT_HERSHEY_SIMPLEX
linha = cv2.LINE_AA
cv2.putText(imagem_rgb, 'Golden Gate', (450, 80), fonte, 2, (255, 255, 255), 3, linha)  # (imagem, texto, localização inicial, tipo de fonte, escala da fonte, cor BGR, espessuraa, tipo de linha)

plt.imshow(imagem_rgb)
plt.show()
