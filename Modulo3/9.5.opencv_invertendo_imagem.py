# usando openCV para inverter uma imagem (flipping)
# para facilitar a compreensão, execute no jupyter notebook

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregando a imagem
imagem = cv2.imread('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate.jpg')

# Flip horizontal
flipped = cv2.flip(imagem, 1) #1 é horizontal, 0 é vertical, -1 são os 2 (horizontal e vertical)
plt.imshow(flipped)
plt.show()

