import pandas as pd
pd.set_option('display.max_columns', 23)
selecao = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Exercicio_final/recipeData.csv', encoding='ISO-8859-1')

#removendo dados menores que 1000 linhas
selecao = selecao.loc[selecao['StyleID'].isin([7, 10, 134, 9, 4, 30, 86, 12, 92, 6, 175, 39])]

#removendo colunas com muitos dados faltantes ou que não serão usadas
selecao.drop('BeerID', axis=1, inplace=True)
selecao.drop('Name', axis=1, inplace=True)
selecao.drop('URL', axis=1, inplace=True)
selecao.drop('UserId', axis=1, inplace=True)
selecao.drop('Style', axis=1, inplace=True)
selecao.drop('PrimingAmount', axis=1, inplace=True)
selecao.drop('PrimingMethod', axis=1, inplace=True)

#alterando strings para inteiros
selecao['SugarScale'] = selecao['SugarScale'].replace('Specific Gravity', 0)
selecao['SugarScale'] = selecao['SugarScale'].replace('Plato', 1)

#transformando variáveis texto da coluna 'Brewmethod' em categorias com one hot encoding
brewmethod_encode=pd.get_dummies(selecao['BrewMethod'])
selecao.drop('BrewMethod', axis=1, inplace=True)    #excluindo a coluna de texto 'BrewMethod'
concatenado = pd.concat([selecao, brewmethod_encode], axis=1)    #inserindo  as variáveis on hot encode novamente no dataset

#verificando dados faltantes
faltantes = concatenado.isnull().sum()
faltantes_percentual = (concatenado.isnull().sum() / len(concatenado['StyleID'])) * 100
print(faltantes_percentual)

#calculando media para a coluna 'BoilGravity', que possui poucos dados faltantes
concatenado['BoilGravity'].fillna(concatenado['BoilGravity'].median(), inplace=True)

#separando as variáveis de treino (x) para valores não nulos
x_treino = concatenado[concatenado['PitchRate'].notnull()] # pegando os valores que não são nulos
x_treino.drop('PitchRate', axis=1, inplace=True)  # excluindo a variável 'PitchRate', pois ela é a variável target
#separando as variáveis de treino (Y) para valores não nulos
y_treino = concatenado[concatenado['PitchRate'].notnull()]['PitchRate']  # filtrando somente a coluna 'PitchRate', este formato de comando substitui o drop da linha de cima
# fazendo as previsões (usando os valores nulos)
x_preench = concatenado[concatenado['PitchRate'].isnull()]  # pegando as linhas com valores nulos na coluna 'PitchRate'
y_preench = concatenado[concatenado['PitchRate'].isnull()]['PitchRate'] # separando a coluna 'PitchRate' com valores nulos
x_preench.drop('PitchRate', axis=1, inplace=True) # excluindo a variável target 'PitchRate'

#removendo variáveis com muitos valores nulos para não influenciar no resultado
x_treino.drop('MashThickness', axis=1, inplace=True)
x_treino.drop('PrimaryTemp', axis=1, inplace=True)
x_preench.drop('MashThickness', axis=1, inplace=True)
x_preench.drop('PrimaryTemp', axis=1, inplace=True)

from sklearn.tree import DecisionTreeRegressor
modelo = DecisionTreeRegressor()   #aplicando modelo
modelo.fit(x_treino, y_treino)

#predição dos novos valores
y_preench = modelo.predict(x_preench)
concatenado.PitchRate[concatenado.PitchRate.isnull()] = y_preench
faltantes = concatenado.isnull().sum()
faltantes_percentual = (concatenado.isnull().sum() / len(concatenado['StyleID'])) * 100
print(faltantes_percentual)
#concatenado.head(15)










