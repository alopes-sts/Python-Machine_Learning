# Usando uma figura como background no openCV e adicionado uma figura por cima
# para facilitar a compreensão, execute no jupyter notebook

import cv2
from matplotlib import pyplot as plt

# Carregando a imagem
imagem = cv2.imread('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate.jpg')

# Desenhando em cima da imagem
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
(centro_x, centro_y) = (imagem_rgb.shape[1]//2, imagem_rgb.shape[0]//2)  # localização do x, y do centro do circulo
branco = (255, 255, 255)

for raio in range(0, 175, 25):
    cv2.circle(imagem_rgb, (centro_x, centro_y), raio, branco)

plt.imshow(imagem_rgb)
plt.show()

# Desenhando vários retângulos muito pequenos
for y in range(0, imagem_rgb.shape[0], 10):
    for x in range(0, imagem_rgb.shape[1], 10):
        imagem_rgb[y:y+3, x:x+3] = (50, 200, 50)
plt.imshow(imagem_rgb)
plt.show()


