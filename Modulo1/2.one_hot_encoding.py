import pandas as pd

pd.set_option('display.max_columns', 50)
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/traffic-collision-data-from-2010-to-present.csv')

# get_dummies irá transformar cada valor da coluna 'Area Name' em uma nova coluna
area_encode = pd.get_dummies(arquivo['Area Name'])
# print(area_encode.head())

# concatenando os 2 arquivos
concatenado = pd.concat([arquivo, area_encode], axis=1)
concatenado.drop('Area Name', axis=1, inplace=True)
# print(concatenado.head())

