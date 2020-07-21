# Exemplo de correlação entre as variaveis do dataset
# Download do dataset: https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# definindo numero de colunas e tamanho da tela
pd.set_option('display.max_columns', 32)
pd.set_option('display.width', 900)

# lendo arquivo
dados = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/2015-building-energy-benchmarking.csv')

# imprimindo o arquivo
print (dados.corr(method='pearson'))

# gerando mapa de calor

plt.figure(figsize=(30,30))
sns.heatmap(dados.corr())
plt.show()