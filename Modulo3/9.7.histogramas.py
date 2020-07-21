# 2 exemplos de como criar um histograma de uma imagem, 1 com o openCV e a outra com o maatplotlib
# O histograma pode ser útil para verificar áreas da imagem onde os pixels estão se destacando mais que outros
# isso pode ser uma área com muito brilho ou muito escura por exemplo, convém equalizar a imagem para executar o modelo de deep learning
# Execute o programa pelo jupyter notebook em blocos para facilitar a visualização

import cv2
from matplotlib import pyplot as plt

# Carregando a imagem
imagem = cv2.imread('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate.jpg')

# Convertendo de BGR para Grayscale
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Criando e exibindo o histograma em openCV
histograma = cv2.calcHist(images=[imagem_cinza], channels=[0], mask=None, histSize=[255], ranges=[0,255])
plt.figure()
plt.title('Histograma em tons de cinza')
plt.xlabel('Intensidade dos pixels')
plt.ylabel('Total de pixels')
plt.plot(histograma)
plt.xlim([0, 256])
plt.show()

# Outro método de criar o histograma utilizando a função 'ravel()' do matplotlib
plt.hist(imagem_cinza.ravel(), 255, [0, 255])
plt.show()

# Equalização do histograma
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # converte de BGR para grayscale
equalizado = cv2.equalizeHist(imagem_cinza)              # Cria um histograma equalizado

# Exibindo a imagem equalizada
cv2.imshow('Original', imagem_cinza)
cv2.imshow('Equalizada', equalizado)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Visualizando o histograma da imagem equalizada
plt.hist(equalizado.ravel(), 255, [0, 255])
plt.show()
