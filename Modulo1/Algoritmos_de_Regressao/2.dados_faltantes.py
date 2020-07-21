# Exemplo de como tratar dados faltantes em colunas
# Download do dataset: https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking

import pandas as pd

planilha = pd.read_csv ('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/2015-building-energy-benchmarking.csv')

# mostra o total de linhas vazias na coluna ENERGYSTARScore
print('Total de campos vazios na coluna ENERGYSTARScore antes: ', planilha['ENERGYSTARScore'].isnull().sum())
print('Porcentagem total de colunas vazias (ENERGYSTARScore): ', planilha['ENERGYSTARScore'].isnull().sum() / len(planilha['ENERGYSTARScore']) * 100)

# substitui os campos vazios na coluna ENERGYSTARScore pela mediana e mostra na tela o resultado
planilha['ENERGYSTARScore'].fillna(planilha['ENERGYSTARScore'].median(), inplace = True)
print()
print('Total de campos vazios na coluna ENERGYSTARScore depois: ', planilha['ENERGYSTARScore'].isnull().sum())
print('Porcentagem total de colunas vazias (ENERGYSTARScore): ', planilha['ENERGYSTARScore'].isnull().sum() / len(planilha['ENERGYSTARScore']) * 100)


