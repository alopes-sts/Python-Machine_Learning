# Exemplo de como remover 'ruído' de imagens, um exemplo de ruído pode ser visualizado ao carregar a imagem abaixo
# pode-se notar vários pontos brancos e pretos, estes pontos podem interferir na eficácia do modelo ao ser executado
# Execute o programa pelo jupyter notebook em blocos para facilitar a visualização

import cv2

# Carregando a imagem
imagem_ruido = cv2.imread('C:/Onedrive/Curso_Machine_Learning/Modulo 3/cerebro-ruido.jpeg')

# Aplicando filtro com a função 'medianBlur', que é muito utilizada para remover ruidos como o desta imagem
# Este método calcula o valor da media dos pixels ao redor do centro do kernel e atribui este valor para o pixel do centro
# O kernel deve sempre ter um número impar para garantir que teremos um pixel central
borrado3 = cv2.medianBlur(imagem_ruido, 3)  # Usando um kernel de 3x3
borrado5 = cv2.medianBlur(imagem_ruido, 5)  # Usando um kernel de 5x5
borrado7 = cv2.medianBlur(imagem_ruido, 7)  # Usando um kernel de 7x7
cv2.imshow('Figura sem filtro', imagem_ruido)
cv2.imshow('Média 3x3', borrado3)
cv2.imshow('Média 5x5', borrado5)
cv2.imshow('Média 7x7', borrado7)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Testando filtro com mediana (função blur)
averaging3 = cv2.blur(imagem_ruido, (3, 3))  # Usando um kernel de 3x3
averaging7 = cv2.blur(imagem_ruido, (7, 7))  # Usando um kernel de 7x7
cv2.imshow('Mediana 3x3', averaging3)
cv2.imshow('Mediana 7x7', averaging7)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Testando com média ponderada (função Gaussian Blur)
# Esta função substitui o pixel central por uma média ponderada dos pixels vizinhos, dando mais peso para os pixels mais próximos
# O valor do desvio padrão é passado por parâmetro, quanto maior, mais borrada a imagem irá ficar a imagem
# O tamanho do kernel tambem influência, quanto maior, mais borrado
gaussian3_0 = cv2.GaussianBlur(imagem_ruido, (3, 3), 0)  # Kernel de 3x3, desvio padrão= 0
gaussian7_0 = cv2.GaussianBlur(imagem_ruido, (7, 7), 0)  # Kernel de 3x3, desvio padrão= 0
gaussian7_1 = cv2.GaussianBlur(imagem_ruido, (7, 7), 1)  # Kernel de 3x3, desvio padrão= 1
gaussian7_2 = cv2.GaussianBlur(imagem_ruido, (7, 7), 2)  # Kernel de 3x3, desvio padrão= 2
gaussian7_3 = cv2.GaussianBlur(imagem_ruido, (7, 7), 3)  # Kernel de 3x3, desvio padrão= 3

cv2.imshow('kernel 3x3, desvio padrão=0', gaussian3_0)
cv2.imshow('kernel 7x7, desvio padrão=0', gaussian7_0)
cv2.imshow('kernel 7x7, desvio padrão=1', gaussian7_1)
cv2.imshow('kernel 7x7, desvio padrão=2', gaussian7_2)
cv2.imshow('kernel 7x7, desvio padrão=3', gaussian7_3)
cv2.waitKey(0)
cv2.destroyAllWindows()




