from sklearn.datasets import load_breast_cancer
import pandas as pd
pd.set_option('display.max_columns', 30)
dados = load_breast_cancer()
x = pd.DataFrame(dados.data, columns=[dados.feature_names])
y = pd.Series(dados.target)   # só serão usadas os dados das variáveis preditorias (x), o k-means nã usa a variável target

#o k-means utiliza dados normalizados
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0, 1))
x_norm = normalizador.fit_transform(x)

from sklearn.cluster import KMeans
#criação do modelo
modelo = KMeans(n_clusters=2, random_state=16) #notar que selecionamos 2 clusters
modelo.fit(x_norm)
print (modelo.cluster_centers_)
print()
print(modelo.predict(x_norm))


