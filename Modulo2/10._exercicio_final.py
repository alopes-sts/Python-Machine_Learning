import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', 65)
#o parâmetro 'header' é para quando o dataset não tem nome das colunas, evita que a primeira linha do dataset seja usada como nome da coluna
#os nomes das colunas serão uma sequência numérica
arquivo1 = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 2/datasets_88836_206004_0.csv', header=None)
arquivo2 = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 2/datasets_88836_206004_1.csv', header=None)
arquivo3 = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 2/datasets_88836_206004_2.csv', header=None)
arquivo4 = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 2/datasets_88836_206004_3.csv', header=None)

#concatenado os datasets
#atentar que agora iremos usar o 'axis' com valor zero, pq estamos concatenando linhas e não colunas como em todos os
#exemplos anteriores onde só era necessário concatenar colunas (axis=1)
concatenado = pd.concat([arquivo1, arquivo2, arquivo3, arquivo4], axis=0)
#atentar que a primeira coluna, é o indice, mas ele está duplicado, pois o indice era o mesmo para os 4 datasets
print(concatenado)
#será necessário modificar os indices com o comando abaixo
concatenado.reset_index(drop=True, inplace=True) #se for usado 'drop=false', será criado uma nova coluna apenas para preservar os indices antigos
print(concatenado)

#Definindo variáveis preditoras e variável target
x = concatenado.iloc[:,0:64] #quando usamos : estamos pegando todas as linhas, para pegar as linhas 0 até 9, usariamos [0:9,0:64]. 0:64 são todas as colunas [linha,coluna]
y = concatenado.iloc[:,64]   #a coluna da variável target é a última (64), (estou usando todas as linhas da coluan 64.
print(x.head())

########## Criando primeiro modelo - GaussianNB ##########
modelo = GaussianNB()
kfold = KFold(n_splits=3)
resultado = cross_val_score(modelo, x, y, cv=kfold, n_jobs=-1)
print('Resultado do primeiro modelo: ', resultado.mean())

########## Criando segundo modelo - KNN ##########
#Normalizando variáveis preditoras
normalizador = MinMaxScaler(feature_range=(0, 1))
X_norm = normalizador.fit_transform(x)
#Criando modelo
modelo = KNeighborsClassifier()
kfold = KFold(n_splits=3)
resultado = cross_val_score(modelo, X_norm, y, cv=kfold, n_jobs=-1)
print('\nResultado do segundo modelo:', resultado.mean())

#Variando os valores para o segundo modelo
#Definindo os valores que serão testasdos no KNN
valores_K = np.array([3, 5, 7, 9])
calculo_distancia = ['minkowski', 'chebyshev']
valores_p = np.array([1, 2, 3])
valores_grid = {'n_neighbors': valores_K, 'metric': calculo_distancia, 'p': valores_p}
#Criando modelo
modelo = KNeighborsClassifier(n_jobs=-1)
#Criando os grids
gridKNN = GridSearchCV(estimator= modelo, param_grid= valores_grid, cv=3, n_jobs=-1)
gridKNN.fit(X_norm, y)
print('\nResultado com ajustes finos para o segundo modelo')
print('Melhor acurácia: ', gridKNN.best_score_)
print('Melhor K: ', gridKNN.best_estimator_.n_neighbors)
print('Método distância: ', gridKNN.best_estimator_.metric)
print('Melhor valor p: ', gridKNN.best_estimator_.p)

########## Criando terceiro modelo - SVM ##########
#Definindo valores que serão testados em SVM
c = np.array([1.0, 2.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
polinomio = np.array([2, 3, 4])
gamma = ['auto', 'scale']
valores_grid = {'C': c, 'kernel': kernel, 'degree': polinomio, 'gamma': gamma}
#Criação do modelo
modelo = SVC()
#Criando os grids
gridSVM = RandomizedSearchCV(estimator= modelo, param_distributions= valores_grid, n_iter=100, n_jobs=-1)
gridSVM.fit(X_norm, y)
print('\nResultado para o terceiro modelo')
print('Melhor valor constante: ', gridSVM.best_estimator_.C)
print('Melhor kernel: ', gridSVM.best_estimator_.kernel)
print('Melhor grau polinômio: ', gridSVM.best_estimator_.degree)
print('Melhor gamma: ', gridSVM.best_estimator_.gamma)
print('Acurácia: ', gridSVM.best_score_)

########## Criando quarto modelo - ExtraTreesClassifier ##########
#definindo valores que serão testados em Extratrees
minimos_split = np.array([2, 3, 4])
maximo_nivel = np.array([3, 4, 5, 6])
minimo_leaf = np.array([2, 3, 4])
algoritmo = ['gini', 'entropy']
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'min_samples_leaf': minimo_leaf, 'criterion': algoritmo}
#criação do modelo
modelo = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
#Criando os grids
gridExtraTrees = RandomizedSearchCV(estimator=modelo, param_distributions=valores_grid, n_iter=50, cv=3, n_jobs=-1)
gridExtraTrees.fit(x, y)
print('\nResultado para o quarto modelo')
print('Mínimo split: ', gridExtraTrees.best_estimator_.min_samples_split)
print('Máxima profundidade: ', gridExtraTrees.best_estimator_.max_depth)
print('Mínimo leaf: ', gridExtraTrees.best_estimator_.min_samples_leaf)
print('Algoritmo: ', gridExtraTrees.best_estimator_.criterion)
print('Acurácia: ', gridExtraTrees.best_score_)

########## Criando quinto modelo - RandomForest ##########
#Definindo valores que serão testados
minimos_split = np.array([2, 3, 4])
maximo_nivel = np.array([3, 4, 5, 6])
minimo_leaf = np.array([2, 3, 4])
algoritmo = ['gini', 'entropy']
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'min_samples_leaf': minimo_leaf, 'criterion': algoritmo}
#Criando o modelo
modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1)
#Criando os grids
gridExtraTrees = RandomizedSearchCV(estimator=modelo, param_distributions=valores_grid, n_iter=50, cv=3, n_jobs=-1)
gridExtraTrees.fit(x, y)
print('\nResultado para o quinto modelo')
print('Mínimo split: ', gridExtraTrees.best_estimator_.min_samples_split)
print('Máxima profundidade: ', gridExtraTrees.best_estimator_.max_depth)
print('Mínimo leaf: ', gridExtraTrees.best_estimator_.min_samples_leaf)
print('Algoritmo: ', gridExtraTrees.best_estimator_.criterion)
print('Acurácia: ', gridExtraTrees.best_score_)

########## Criando sexto modelo - GradientBoosting ##########
#Definindo os valores que serão testados
minimos_split = np.array([2, 3, 4])
maximo_nivel = np.array([3, 4, 5, 6])
minimo_leaf = np.array([2, 3, 4])
valores_grid = {'min_samples_split': minimos_split, 'max_depth': maximo_nivel, 'min_samples_leaf': minimo_leaf, 'criterion': algoritmo}
#Criação do modelo
modelo = GradientBoostingClassifier()
#Criando os grids
gridExtraTrees = RandomizedSearchCV(estimator=modelo, param_distributions=valores_grid, n_iter=50, cv=3, n_jobs=-1)
gridExtraTrees.fit(x, y)
print('\nResultado para o sexto modelo')
print('Mínimo split: ', gridExtraTrees.best_estimator_.min_samples_split)
print('Máxima profundidade: ', gridExtraTrees.best_estimator_.max_depth)
print('Mínimo leaf: ', gridExtraTrees.best_estimator_.min_samples_leaf)
print('Algoritmo: ', gridExtraTrees.best_estimator_.criterion)
print('Acurácia: ', gridExtraTrees.best_score_)

########## Usando BaggingClassifier no GradientBoosting ##########
#Criação do modelo
modelo = BaggingClassifier(base_estimator=GradientBoostingClassifier(), n_estimators=100, max_samples=0.7, n_jobs=-1)
kfold = KFold(n_splits=3)
resultado = cross_val_score(modelo, x, y, cv=kfold, n_jobs=-1)
print('Resultado do Gradient Boosting com Bagging Classifier')
print(resultado.mean())

########## Testando mais parâmetros no GradientBoosting ##########
#Definindo os valores que serão testados
criterion = ['friedman_mse', 'mse']
max_features = np.array([64, 6, 8, 12, 16, 32])
#Criação do modelo
modelo = GradientBoostingClassifier(n_estimators=500, learning_rate=0.085, min_samples_split=3, min_samples_leaf=4, max_depth=5)
#Criando os grids
gridExtraTrees = RandomizedSearchCV(estimator=modelo, param_distributions=valores_grid, n_iter=50, cv=3, n_jobs=-1)
gridExtraTrees.fit(x, y)
print('\nResultados com outros parâmetros para o GradientBoosting')
print('max_features: ', gridExtraTrees.best_estimator_.max_features)
print('criterion: ', gridExtraTrees.best_estimator_.criterion)
print('Acurácia: ', gridExtraTrees.best_score_)

########## Testando mais parâmetros no GradientBoosting ##########
learning_rate = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09])
#Criação do modelo
modelo = GradientBoostingClassifier(n_estimators=500, min_samples_split=3, min_samples_leaf=4, max_depth=5, max_features=6, criterion=mse)
#Criando os grids
gridExtraTrees = RandomizedSearchCV(estimator=modelo, param_distributions=valores_grid, n_iter=50, cv=3, n_jobs=-1)
gridExtraTrees.fit(x, y)
print('Imprimindo os melhores parâmetros, ainda no sexto modelo')
print('Learning rate: ', gridGradient.best_estimator_.learning_rate)
print('Acurácia: ', gridGradient.best_score_)

########## Criando sétimo modelo - AdaBoost ##########
valores_grid = {'learning_rate': np.array([1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35])}
#Criação do modelo
modelo = AdaBoostClassifier(n_estimators=500)
#Criando os grids
gridAdaBoost = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=3, n_jobs=-1)
gridAdaBoost.fit(x, y)
print('\nResultado para o sétimo modelo')
print('Melhor taxa de apredizagem: ', gridAdaBoost.best_estimator_.learning_rate)
print('Acurácia: ', gridAdaBoost.best_score_)

########## Usando PCA ##########
#Alguns dos algoritmos acima podem demorar bastante, o PCA é útil nestes casos
#O PCA irá reduzir a dimensionalidade do problema
#Transformando os dados para componentes PCA
pca = PCA(n_components=30)
x_pca = pca.fit_transform(X_norm)
print('Testando PCA')
print('Variância explicada dos componentes: ', pca.explained_variance_ratio_)
print('Variância total: ', sum(pca.explained_variance_ratio_))
