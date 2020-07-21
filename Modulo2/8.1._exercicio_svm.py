import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

#abrindo o arquivo diretamente do site, estamos tambem informando o nome das colunas, pois não existe esta informação no arquivo principal
#sem isso o pandas irá definir a primeira linha como nome das colunas
arquivo = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
                      names=['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'])

#verificando dataset, analisando dados que não são números ou dados faltantes
#print(arquivo.info(),'\n')
#faltantes = arquivo.isnull().sum()
#faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['Class']))
#print(faltantes_percentual)

#definindo variáveis preditorias e variável target
x = arquivo.drop('Class', axis=1)
y = arquivo['Class']

#normalização
normalizador = MinMaxScaler(feature_range=(0, 1))
X_norm = normalizador.fit_transform(x)

#Criação do modelo
modelo = SVC() #SVC= SVM para classificação
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo, X_norm, y, cv= skfold, n_jobs=-1)
print('Resultado sem ajustes finos: ', resultado.mean())

#Ajustes finos
c = np.array([1.0, 0.95, 1.05, 1.1, 1.2, 2, 0.9, 0.8]) #testar valores bem menores e bem maiores que o default, verificar o resultado e testar novamente com valores maiores e menores que este resultado
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
polinomio = np.array([2, 3, 4, 5])
gamma = ['auto', 'scale'] #parâmetro usado somente para problemas de classificação, para regressão é o 'epsilon'
valores_grid = {'C': c, 'kernel': kernel, 'degree': polinomio, 'gamma': gamma}

#criação do modelo
modelo = SVC()

#Criando os grids
gridSVM = GridSearchCV(estimator= modelo, param_grid= valores_grid, cv=3, n_jobs=-1)
gridSVM.fit(X_norm, y)

#imprimindo resultados
print('\nResultados com ajustes finos 1')
print('Melhor valor constante: ', gridSVM.best_estimator_.C)
print('Melhor kernel: ', gridSVM.best_estimator_.kernel)
print('Melhor grau polinômio: ', gridSVM.best_estimator_.degree)
print('Melhor gamma: ', gridSVM.best_estimator_.gamma)
print('Acurácia: ', gridSVM.best_score_)

#Refazendo os testes com outros valores de constante, a melhor constante do modelo anterior foi o maior número que passamos, então vamos refazer o modelo usando valores maiores
c = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0])
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
polinomio = np.array([2, 3, 4, 5])
gamma = ['auto', 'scale'] #parâmetro usado somente para problemas de classificação, para regressão é o 'epsilon'
valores_grid = {'C': c, 'kernel': kernel, 'degree': polinomio, 'gamma': gamma}

#criação do modelo
modelo = SVC()

#Criando os grids
gridSVM = GridSearchCV(estimator= modelo, param_grid= valores_grid, cv=3, n_jobs=-1)
gridSVM.fit(X_norm, y)

#imprimindo resultados
print('\nResultados com ajustes finos 2')
print('Melhor valor constante: ', gridSVM.best_estimator_.C)
print('Melhor kernel: ', gridSVM.best_estimator_.kernel)
print('Melhor grau polinômio: ', gridSVM.best_estimator_.degree)
print('Melhor gamma: ', gridSVM.best_estimator_.gamma)
print('Acurácia: ', gridSVM.best_score_)


