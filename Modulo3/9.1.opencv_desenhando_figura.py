# Criando figuras com o OpenCV
# Pressionar qualquer tecla para a próxima figura
# para facilitar a compreensão, execute no jupyter notebook

import numpy as np
import cv2

# Inicializando nossa tela como um espaço  de pixels 500x500 com 3 canais para o RGB
espaco = np.zeros((500, 500, 3), dtype='uint8')

# Desenhando uma linha verde do canto superior esquerda da nossa tela para o canto inferior direito
verde = (0, 255, 0)
cv2.line(espaco, (0,0), (500, 500), verde)
cv2.imshow('Figura', espaco)
cv2.waitKey(0)

# Desenhando uma linha vermelha de 3 pixels de espessura do canto inferior esquerdo para a parte superior direita
vermelho = (0, 0, 255)
cv2.line(espaco, (500, 0), (0, 500), vermelho, 3)
cv2.imshow('Figura', espaco)
cv2.waitKey(0)

# Desenhando um quadrado verde de 40x40 pixels, começando no ponto 20x20 e que terminando no ponto 60x60
cv2.rectangle(espaco, (20, 20), (60, 60), verde)
cv2.imshow('Figura', espaco)
cv2.waitKey(0)

# Desenhando um retângulo vermelho com 5 pixels de espessura
cv2.rectangle(espaco, (10, 150), (100, 300), vermelho, 5)
cv2.imshow('Figura', espaco)
cv2.waitKey(0)

# Desenhando um retângulo azul preenchido
azul = (255, 0, 0)
cv2.rectangle(espaco, (400, 80), (480, 160), azul, -1)
cv2.imshow('Figura', espaco)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Desenhando um circulo branco no centro com vários circulos ao redor
espaco = np.zeros((500, 500, 3), dtype='uint8')
(centro_x, centro_y) = (espaco.shape[1]//2, espaco.shape[0]//2) # localizacao x, y do centro do círculo
branco = (255, 255, 255)
for raio in range(0, 250, 25):
    cv2.circle(espaco, (centro_x, centro_y), raio, branco)

cv2.imshow('Figura', espaco)
cv2.waitKey(0)
cv2.destroyAllWindows()
