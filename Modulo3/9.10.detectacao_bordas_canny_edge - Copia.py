# Exemplos de detecção de borda com OpenCV (função Canny Edge Detector)
# Execute o programa pelo jupyter notebook em blocos para facilitar a visualização

import cv2

# Carregando as imagens
imagem = cv2.imread('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate.jpg')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicando o Canny Edge
imagem_com_filtro = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)  # Aplicando filtro para o ruido, necessário para melhor detecção de bordas
cv2.imshow('Blurred', imagem_com_filtro)  # Exibindo a imagem desfocada
cv2.waitKey(0)

# Calculando os filtros canny com diferentes limiares
largo = cv2.Canny(imagem_com_filtro, 50, 220)
medio = cv2.Canny(imagem_com_filtro, 70, 140)
apertado = cv2.Canny(imagem_com_filtro, 210, 220)

# Mostrando os mapas de bordas (map edges) com diferentes limiares
cv2.imshow('largo', largo)
cv2.waitKey(0)
cv2.imshow('medio', medio)
cv2.waitKey(0)
cv2.imshow('apertado', apertado)
cv2.waitKey(0)
cv2.destroyAllWindows()