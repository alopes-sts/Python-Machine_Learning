import pandas as pd
import numpy as np
from sklearn.svm import SVR #(algoritmo para regressão do SVM, para classificação é o SVC)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler # como é um algoritmo de cálculo de distância, precisa normalizar
from sklearn.model_selection import GridSearchCV

arquivo = pd.read_excel ('C:/Onedrive/Curso_Machine_Learning/Modulo 2/Concrete_Data.xls')

#verificando dataset, analisando dados que não são números ou dados faltantes
#print(arquivo.info(),'\n')
#faltantes = arquivo.isnull().sum()
#faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['Age (day)']))
#print(faltantes_percentual)

#definindo variáveis preditorias e variável target
y = arquivo['Concrete compressive strength(MPa, megapascals) ']
x = arquivo.drop('Concrete compressive strength(MPa, megapascals) ', axis=1)

#normalização
normalizador = MinMaxScaler(feature_range=(0, 1))
X_norm = normalizador.fit_transform(x)

#Criação do modelo
modelo = SVR()
kfold = KFold(n_splits=3)
resultado = cross_val_score(modelo, X_norm, y, cv=kfold, n_jobs=-1)
print('Resultado com parâmetros default', resultado.mean())

#Ajustes finos
c = np.array([1.0, 0.95, 1.05, 1.1, 1.2, 2, 0.9, 0.8]) #testar valores bem menores e bem maiores que o default, verificar o resultado e testar novamente com valores maiores e menores que este resultado
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
polinomio = np.array([2, 3, 4])
epsilon = np.array([0.1, 0.2, 0.05])
valores_grid = {'C': c, 'kernel': kernel, 'degree': polinomio, 'epsilon': epsilon}

#criação do modelo
modelo = SVR()

#Criando os grids
kfold = KFold(n_splits=3, shuffle=True)
gridSVM = GridSearchCV(estimator= modelo, param_grid= valores_grid, cv=kfold, n_jobs=-1)
gridSVM.fit(X_norm, y)

#imprimindo resultados
print('\nMelhor valor constante: ', gridSVM.best_estimator_.C)
print('Melhor kernel: ', gridSVM.best_estimator_.kernel)
print('Melhor grau polinômio: ', gridSVM.best_estimator_.degree)
print('Melhor epsilon: ', gridSVM.best_estimator_.epsilon)
print('R2: ', gridSVM.best_score_)

#Testando com outros valores
c = np.array([2, 4, 8, 16, 32])
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
polinomio = np.array([2, 3, 4])
epsilon = np.array([0.2, 0.3, 0.4, 0.5])   #parâmetro usado somente para problemas de regressão, para classificação é o 'gamma'
valores_grid = {'C': c, 'kernel': kernel, 'degree': polinomio, 'epsilon': epsilon}

#criação do modelo
modelo = SVR()

#Criando os grids
kfold = KFold(n_splits=3, shuffle=True)
gridSVM = GridSearchCV(estimator= modelo, param_grid= valores_grid, cv=kfold, n_jobs=-1)
gridSVM.fit(X_norm, y)

#imprimindo resultados
print('\nTestando com outros valores')
print('Melhor valor constante: ', gridSVM.best_estimator_.C)
print('Melhor kernel: ', gridSVM.best_estimator_.kernel)
print('Melhor grau polinômio: ', gridSVM.best_estimator_.degree)
print('Melhor epsilon: ', gridSVM.best_estimator_.epsilon)
print('R2: ', gridSVM.best_score_)



