# Exemplos de operações morfológicas (erosão e dilatação), opening e closing com OpenCV
# Execute o programa pelo jupyter notebook em blocos para facilitar a visualização

import cv2

# Carregando as imagens
imagem = cv2.imread('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate.jpg')
imagem_ruido = cv2.imread('C:/Onedrive/Curso_Machine_Learning/Modulo 3/cerebro-ruido.jpeg')

# Aplicando a função de erosão
# Erosão é a técnica de remover pixels perto dos limites de um objeto, com o objetivo de ressaltar esse objeto
# Na função de erosão, o kernel de convolução substitui o valor central pelo menor valor encontrado no kernel. O objetivo é reduzir áreas de menor intensidade
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
eroded = cv2.erode(imagem_cinza.copy(), (5, 5), iterations=3)  # (imagem, tamanho do kernel, número de iterações de aplcação da erosão)
cv2.imshow('Imagem com erosao', eroded)
cv2.imshow('Imagem cinza', imagem_cinza)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Aplicando a função de dilatação
# Dilatação é o contrário de erosão, pixels são adicionados próximos aos limites de um objeto, com o objetico de preencher pontos faltantes
# A dilataçãao pode ser usada para ampliar pequenos detalhes
# Na função de dilatação, o kernel de convolução substitui o valor central pelo mairo valor encontrado no kernel com o objetivo de aumentar regiões com pouca intensidade
dilated = cv2.dilate(imagem_cinza.copy(), None, iterations=3)  # Usando o 'None' o tamanho do kernel é default de 3x3
cv2.imshow('Imagem com dilatacao', dilated)
cv2.imshow('Imagem cinza', imagem_cinza)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Usando a função 'opening', opening faz a função de erosão e depois a dilatação
opened = cv2.morphologyEx(imagem_ruido, cv2.MORPH_OPEN, None)
cv2.imshow('Imagem com funcao opening', opened)
cv2.imshow('Imagem ruido', imagem_ruido)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Usando a função 'closing', faz a função de dilatação e depois a erosão
closing = cv2.morphologyEx(imagem_ruido, cv2.MORPH_CLOSE, None)
cv2.imshow('Imagem com funcao closing', closing)
cv2.imshow('Imagem ruido', imagem_ruido)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Usando Gradiente Morfológico, é a diferença entre dilatação e erosão
gradient = cv2.morphologyEx(imagem_cinza, cv2.MORPH_GRADIENT, None)
cv2.imshow('Imagem com funcao gradiente', gradient)
cv2.imshow('Imagem cinza', imagem_cinza)
cv2.waitKey(0)
cv2.destroyAllWindows()

