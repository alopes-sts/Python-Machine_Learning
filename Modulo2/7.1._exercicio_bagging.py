import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

arquivo = pd.read_excel ('C:/Onedrive/Curso_Machine_Learning/Modulo 2/Concrete_Data.xls')

#verificando dataset, analisando dados que não são números ou dados faltantes
#print(arquivo.info(),'\n')
#faltantes = arquivo.isnull().sum()
#faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['Age (day)']))
#print(faltantes_percentual)

#definindo variáveis preditorias e variável target
y = arquivo['Concrete compressive strength(MPa, megapascals) ']
x = arquivo.drop('Concrete compressive strength(MPa, megapascals) ', axis=1)

#Criação do modelo
modelo = BaggingRegressor(n_estimators=100, max_samples=0.5, n_jobs=-1) #como não foi informado como parâmetro, será usado 'decision tree' como default para o estimador
kfold = KFold(n_splits=3, shuffle=True)
resultado = cross_val_score(modelo, x, y, cv=kfold, n_jobs=-1)
print('Resultado 1: ', resultado.mean())

#testando com Gradient Boosting
modelo = BaggingRegressor(base_estimator=GradientBoostingRegressor(n_estimators=100), n_estimators=100, max_samples=0.5, n_jobs=-1) #atentar que foi definido estimadores para o Gradient Boosting e o Bagging Regressor
kfold = KFold(n_splits=3, shuffle=True)
resultado = cross_val_score(modelo, x, y, cv= kfold, n_jobs=-1)
print('Resultado com Gradient Boosting: ', resultado.mean())

#usando o parâmetro 'scoring' para demonstrar a variação do erro nas previsões do algoritmo
modelo = BaggingRegressor(base_estimator=GradientBoostingRegressor(n_estimators=100), n_estimators=100, max_samples=0.5, n_jobs=-1)
kfold = KFold(n_splits=3, shuffle=True)
resultado = cross_val_score(modelo, x, y, cv= kfold, n_jobs=-1, scoring='neg_mean_absolute_error') #notar o parâmetro 'scoring'
#o resultado irá dizer qual a precisão do modelo aplicado
print('A variação do erro é de ', resultado.mean(), ' para mais ou para menos')