# Exemplo de correlação entre as variaveis do dataset
# O dataset usado é sobre informações de diabetes
# Download do dataset: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# define o tamanho da tela e numero de colunas que devem aparecer
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 9)

colunas = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dados = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/pima-indians-diabetes.csv', names=colunas)
print(dados.corr(method='pearson'))

print()
print()

plt.figure(figsize=(5, 5))
sns.heatmap(dados.corr())
plt.show()

