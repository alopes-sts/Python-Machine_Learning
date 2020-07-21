# Exemplo de uso da biblioteca Pillow
# Execute o programa pelo jupyter notebook em blocos para facilitar a visualização

from PIL import Image

imagem = Image.open('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate.jpg')
imagem.show()  # Abrindo a imagem com visualizador (fora do jupyter notebook)

# para visualizar a imagem dentro do jupyter notebook
from IPython.display import display
display(imagem)

# Para verificar as dimensões (altura e largura) da imagem
largura, altura = imagem.size
print('Largura: ', largura)
print('Altura', altura)

# Para salvar uma imagem
imagem.save('C:/Onedrive/Curso_Machine_Learning/Modulo 3/golden-gate-pillow.jpg')

# Para cortar a imagem
posicoes = (200, 200, 500, 300)  # Definição de um retângulo (esquerda, cima, direita, baixo)
image_cortada = imagem.crop(posicoes)
display(image_cortada)  # Comando no jupyter notebook
image_cortada.show()    # Para visualizar em uma janela externa

# Para colocar a imagem em escala de cinza
imagem_cinza = imagem.convert('L')
display(imagem_cinza)   # Comando no jupyter notebook
imagem_cinza.show()     # Para visualizar em uma janela externa

# Para definir quantidade de cores
imagem_RGB = imagem.convert('P', palette=Image.ADAPTIVE, colors=5)  # colors pode ter valor até 255, que representa todas as cores
display(imagem_RGB)     # Comando no jupyter notebook
imagem_RGB.show()       # Para visualizar em uma janela externa

# Para rotacionar a imagem
rotacionada = imagem.rotate(180)
display(rotacionada)    # Comando no jupyter notebook
rotacionada.show()      # Para visualizar em uma janela externa

# Para redimensionar a imagem
redimensionada = imagem.resize((400, 200))
display(redimensionada) # Comando no jupyter notebook
redimensionada.show()   # Para visualizar em uma janela externa

# Para ajustar o brilho da imagem
from PIL import ImageEnhance
realcador = ImageEnhance.Brightness(imagem)
nova_imagem = realcador.enhance(0.8)  # Define o nível de brilho
display(nova_imagem)    # Comando no jupyter notebook
nova_imagem.show()      # Para visualizar em uma janela externa

# Para ajustar o contraste da imagem
from PIL import ImageEnhance
realcador = ImageEnhance.Contrast(imagem)
nova_imagem = realcador.enhance(2)  # Define o nível de contraste
display(nova_imagem)    # Comando no jupyter notebook
nova_imagem.show()      # Para visualizar em uma janela externa

# Para coletar um pixel específico da imagem
pixel = imagem.getpixel((150,87))
print(pixel)            # Terá como resultado o RGB do pixel específico
