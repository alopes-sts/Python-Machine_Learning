import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

pd.set_option('display.max_columns', 91)
pd.set_option('display.max_rows', 91)
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 2/BRAZIL_CITIES.csv', sep=";", decimal=",")
arquivo.drop(['CITY', 'IDHM Ranking 2010', 'IDHM_Renda', 'IDHM_Longevidade', 'IDHM_Educacao', 'LONG', 'LAT', 'GVA_MAIN', 'REGIAO_TUR', 'MUN_EXPENDIT', 'HOTELS', 'BEDS', 'Pr_Agencies', 'Pu_Agencies', 'Pr_Bank', 'Pu_Bank', 'Pr_Assets', 'Pu_Assets','UBER','MAC','WAL-MART'], axis=1, inplace=True)
arquivo['AREA']=arquivo['AREA'].str.replace(',','')   #substituindo valores com erro na variável 'AREA'

#transformando colunas com strings em novas colunas com números inteiros
arquivo_encode = pd.get_dummies(arquivo['RURAL_URBAN'])
arquivo.drop('RURAL_URBAN', axis=1, inplace=True)
arquivo_concatenado = pd.concat([arquivo, arquivo_encode], axis=1)
arquivo_encode = pd.get_dummies(arquivo_concatenado['STATE'])
arquivo_concatenado.drop('STATE', axis=1, inplace=True)
arquivo_concatenado2 = pd.concat([arquivo_concatenado, arquivo_encode], axis=1)

#convertendo variável com dados null para 'zero'
arquivo_concatenado2['CATEGORIA_TUR'] = arquivo_concatenado2['CATEGORIA_TUR'].replace('A', 1)
arquivo_concatenado2['CATEGORIA_TUR'] = arquivo_concatenado2['CATEGORIA_TUR'].replace('B', 2)
arquivo_concatenado2['CATEGORIA_TUR'] = arquivo_concatenado2['CATEGORIA_TUR'].replace('C', 3)
arquivo_concatenado2['CATEGORIA_TUR'] = arquivo_concatenado2['CATEGORIA_TUR'].replace('D', 4)
arquivo_concatenado2['CATEGORIA_TUR'] = arquivo_concatenado2['CATEGORIA_TUR'].replace('E', 5)
arquivo_concatenado2['CATEGORIA_TUR'] = arquivo_concatenado2['CATEGORIA_TUR'].replace(np.nan, 0)

#convertendo variáveis para float
arquivo_concatenado2['AREA'] = arquivo_concatenado2['AREA'].astype(float)
arquivo_concatenado2['IDHM'] = arquivo_concatenado2['IDHM'].astype(float)
arquivo_concatenado2['ALT'] = arquivo_concatenado2['ALT'].astype(float)
arquivo_concatenado2['GVA_AGROPEC'] = arquivo_concatenado2['GVA_AGROPEC'].astype(float)
arquivo_concatenado2['GVA_INDUSTRY'] = arquivo_concatenado2['GVA_INDUSTRY'].astype(float)
arquivo_concatenado2['GVA_SERVICES'] = arquivo_concatenado2['GVA_SERVICES'].astype(float)
arquivo_concatenado2['GVA_PUBLIC'] = arquivo_concatenado2['GVA_PUBLIC'].astype(float)
arquivo_concatenado2[' GVA_TOTAL '] = arquivo_concatenado2[' GVA_TOTAL '].astype(float)
arquivo_concatenado2['TAXES'] = arquivo_concatenado2['TAXES'].astype(float)
arquivo_concatenado2['GDP'] = arquivo_concatenado2['GDP'].astype(float)
arquivo_concatenado2['GDP_CAPITA'] = arquivo_concatenado2['GDP_CAPITA'].astype(float)

#substituindo valores zerados pela mediana
arquivo_concatenado2['IBGE_RES_POP'].fillna(arquivo_concatenado2['IBGE_RES_POP'].median(), inplace=True)
arquivo_concatenado2['IBGE_RES_POP_BRAS'].fillna(arquivo_concatenado2['IBGE_RES_POP_BRAS'].median(), inplace=True)
arquivo_concatenado2['IBGE_RES_POP_ESTR'].fillna(arquivo_concatenado2['IBGE_RES_POP_ESTR'].median(), inplace=True)
arquivo_concatenado2['IBGE_DU'].fillna(arquivo_concatenado2['IBGE_DU'].median(), inplace=True)
arquivo_concatenado2['IBGE_DU_URBAN'].fillna(arquivo_concatenado2['IBGE_DU_URBAN'].median(), inplace=True)
arquivo_concatenado2['IBGE_DU_RURAL'].fillna(arquivo_concatenado2['IBGE_DU_RURAL'].median(), inplace=True)
arquivo_concatenado2['IBGE_POP'].fillna(arquivo_concatenado2['IBGE_POP'].median(), inplace=True)
arquivo_concatenado2['IBGE_1'].fillna(arquivo_concatenado2['IBGE_1'].median(), inplace=True)
arquivo_concatenado2['IBGE_1-4'].fillna(arquivo_concatenado2['IBGE_1-4'].median(), inplace=True)
arquivo_concatenado2['IBGE_5-9'].fillna(arquivo_concatenado2['IBGE_5-9'].median(), inplace=True)
arquivo_concatenado2['IBGE_10-14'].fillna(arquivo_concatenado2['IBGE_10-14'].median(), inplace=True)
arquivo_concatenado2['IBGE_15-59'].fillna(arquivo_concatenado2['IBGE_15-59'].median(), inplace=True)
arquivo_concatenado2['IBGE_60+'].fillna(arquivo_concatenado2['IBGE_60+'].median(), inplace=True)
arquivo_concatenado2['IBGE_PLANTED_AREA'].fillna(arquivo_concatenado2['IBGE_PLANTED_AREA'].median(), inplace=True)
arquivo_concatenado2['IBGE_CROP_PRODUCTION_$'].fillna(arquivo_concatenado2['IBGE_CROP_PRODUCTION_$'].median(), inplace=True)
arquivo_concatenado2['IDHM'].fillna(arquivo_concatenado2['IDHM'].median(), inplace=True)
arquivo_concatenado2['ALT'].fillna(arquivo_concatenado2['ALT'].median(), inplace=True)
arquivo_concatenado2['PAY_TV'].fillna(arquivo_concatenado2['PAY_TV'].median(), inplace=True)
arquivo_concatenado2['FIXED_PHONES'].fillna(arquivo_concatenado2['FIXED_PHONES'].median(), inplace=True)
arquivo_concatenado2['AREA'].fillna(arquivo_concatenado2['AREA'].median(), inplace=True)
arquivo_concatenado2['ESTIMATED_POP'].fillna(arquivo_concatenado2['ESTIMATED_POP'].median(), inplace=True)
arquivo_concatenado2['GVA_AGROPEC'].fillna(arquivo_concatenado2['GVA_AGROPEC'].median(), inplace=True)
arquivo_concatenado2['GVA_INDUSTRY'].fillna(arquivo_concatenado2['GVA_INDUSTRY'].median(), inplace=True)
arquivo_concatenado2['GVA_SERVICES'].fillna(arquivo_concatenado2['GVA_SERVICES'].median(), inplace=True)
arquivo_concatenado2['GVA_PUBLIC'].fillna(arquivo_concatenado2['GVA_PUBLIC'].median(), inplace=True)
arquivo_concatenado2[' GVA_TOTAL '].fillna(arquivo_concatenado2[' GVA_TOTAL '].median(), inplace=True)
arquivo_concatenado2['TAXES'].fillna(arquivo_concatenado2['TAXES'].median(), inplace=True)
arquivo_concatenado2['GDP'].fillna(arquivo_concatenado2['GDP'].median(), inplace=True)
arquivo_concatenado2['POP_GDP'].fillna(arquivo_concatenado2['POP_GDP'].median(), inplace=True)
arquivo_concatenado2['GDP_CAPITA'].fillna(arquivo_concatenado2['GDP_CAPITA'].median(), inplace=True)
arquivo_concatenado2['COMP_TOT'].fillna(arquivo_concatenado2['COMP_TOT'].median(), inplace=True)
arquivo_concatenado2['COMP_A'].fillna(arquivo_concatenado2['COMP_A'].median(), inplace=True)
arquivo_concatenado2['COMP_B'].fillna(arquivo_concatenado2['COMP_B'].median(), inplace=True)
arquivo_concatenado2['COMP_C'].fillna(arquivo_concatenado2['COMP_C'].median(), inplace=True)
arquivo_concatenado2['COMP_D'].fillna(arquivo_concatenado2['COMP_D'].median(), inplace=True)
arquivo_concatenado2['COMP_E'].fillna(arquivo_concatenado2['COMP_E'].median(), inplace=True)
arquivo_concatenado2['COMP_F'].fillna(arquivo_concatenado2['COMP_F'].median(), inplace=True)
arquivo_concatenado2['COMP_G'].fillna(arquivo_concatenado2['COMP_G'].median(), inplace=True)
arquivo_concatenado2['COMP_H'].fillna(arquivo_concatenado2['COMP_H'].median(), inplace=True)
arquivo_concatenado2['COMP_I'].fillna(arquivo_concatenado2['COMP_I'].median(), inplace=True)
arquivo_concatenado2['COMP_J'].fillna(arquivo_concatenado2['COMP_J'].median(), inplace=True)
arquivo_concatenado2['COMP_K'].fillna(arquivo_concatenado2['COMP_K'].median(), inplace=True)
arquivo_concatenado2['COMP_L'].fillna(arquivo_concatenado2['COMP_L'].median(), inplace=True)
arquivo_concatenado2['COMP_M'].fillna(arquivo_concatenado2['COMP_M'].median(), inplace=True)
arquivo_concatenado2['COMP_N'].fillna(arquivo_concatenado2['COMP_N'].median(), inplace=True)
arquivo_concatenado2['COMP_O'].fillna(arquivo_concatenado2['COMP_O'].median(), inplace=True)
arquivo_concatenado2['COMP_P'].fillna(arquivo_concatenado2['COMP_P'].median(), inplace=True)
arquivo_concatenado2['COMP_Q'].fillna(arquivo_concatenado2['COMP_Q'].median(), inplace=True)
arquivo_concatenado2['COMP_R'].fillna(arquivo_concatenado2['COMP_R'].median(), inplace=True)
arquivo_concatenado2['COMP_S'].fillna(arquivo_concatenado2['COMP_S'].median(), inplace=True)
arquivo_concatenado2['COMP_T'].fillna(arquivo_concatenado2['COMP_T'].median(), inplace=True)
arquivo_concatenado2['COMP_U'].fillna(arquivo_concatenado2['COMP_U'].median(), inplace=True)
arquivo_concatenado2['Cars'].fillna(arquivo_concatenado2['Cars'].median(), inplace=True)
arquivo_concatenado2['Motorcycles'].fillna(arquivo_concatenado2['Motorcycles'].median(), inplace=True)
arquivo_concatenado2['Wheeled_tractor'].fillna(arquivo_concatenado2['Wheeled_tractor'].median(), inplace=True)
arquivo_concatenado2['POST_OFFICES'].fillna(arquivo_concatenado2['POST_OFFICES'].median(), inplace=True)


#verificando dados faltantes
faltantes_percentual = (arquivo_concatenado2.isnull().sum() / len(arquivo_concatenado2['CAPITAL'])) * 100
print(faltantes_percentual)

#definindo variáveis
x = arquivo_concatenado2.drop('IDHM', axis= 1)
y = arquivo_concatenado2['IDHM']

#normalizando as variáveis preditoras
normalizador = MinMaxScaler(feature_range=(0, 1))
x_norm = normalizador.fit_transform(x)

#transformando os dados em componentes PCA
pca = PCA(n_components=15)
x_pca = pca.fit_transform(x_norm)
print('Variância explica dos componentes PCA:\n', pca.explained_variance_ratio_)
print('\nSoma: ',sum(pca.explained_variance_ratio_))   #demonstra a quantidade de dados total do dataset que estão sendo preservados, usando-se somente 15 variáveis

#criação do modelo
modelo = RandomForestRegressor(n_estimators=100, n_jobs=-1)
kfold = KFold(n_splits=3)
resultado = cross_val_score(modelo, x_pca, y, cv=kfold, n_jobs=-1)
print('\nResultado do modelo de regressão: ',resultado.mean())

#definindo valores que serão testados em RandomForest
#notar que não é necessário  criar um array com os valores para os estimadores, pq é uma árvore de descisão, basta testar
#aumentando o valor do parâmetro 'n_estimators' na criação do modelo, para que ele gera mais árvores de decisão
minimos_split = np.array([2, 3, 4])
maximo_nivel = np.array([3, 5, 7, 9, 11, 14])
minimo_leaf = np.array([3, 4, 5, 6])
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'min_samples_leaf': minimo_leaf}

#criação do modelo
modelo = RandomForestRegressor(n_estimators=100, n_jobs=-1)

#criando os grids
gridRandomForest = RandomizedSearchCV(estimator=modelo, param_distributions=valores_grid, cv=3, n_iter=50, n_jobs=-1)
gridRandomForest.fit(x_pca, y)

#imprimindo melhores parâmetros
print('Minimo split: ', gridRandomForest.best_estimator_.min_samples_split)
print('Maxima profundidade: ', gridRandomForest.best_estimator_.max_depth)
print('Minimo leaf: ', gridRandomForest.best_estimator_.min_samples_leaf)
print('R2: ', gridRandomForest.best_score_)


