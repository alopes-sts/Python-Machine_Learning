from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

#o dataset escolhido é um problema de classificação e foi escolhido o algoritmo do KNN mas poderia ser outro algoritmo
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#normalizando as variáveis preditoras
normalizador = MinMaxScaler(feature_range=(0, 1))
x_norm = normalizador.fit_transform(x)

#transformando os dados em componentes PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_norm)

#criando os conjuntos de dados de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x_pca, y, test_size= 0.3, random_state=14)

#criaçao do modelo
modelo = KNeighborsClassifier()
modelo.fit(x_treino, y_treino)

#imprimindo resultados
score = modelo.score(x_teste, y_teste)  #cálculo de acurácia
print('Acurácia: ', score)

# verificando a variância de cada componente conforme explicado na teoria (retas PC1, PC2, etc)
print('Variância explicada dos componentes: ', pca.explained_variance_ratio_)