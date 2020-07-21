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
modelo = KMeans(n_clusters=2, random_state=16) #notar que selecionamos 2 clusters OBS.: verificar vcom valores diferentes de random_state
modelo.fit(x_norm)
print (modelo.cluster_centers_)
print()
print(modelo.predict(x_norm))

#comparando o resultado do k-means com a variável target
clusters = modelo.predict(x_norm)

#função para comparar dados entre dois arrays (este é apenas um exemplo, existe a função accuracy_score para isso, ver mais abaixo)
def compara(resultado1, resultado2):
    acertos = 0
    for i in range(len(resultado1)):
        if resultado1[i] == resultado2[i]:
            acertos += 1
        else:
            pass
    return acertos/len(resultado1)

resultado = compara(clusters, y)
print('\nA relação entre os dados foi de: ', resultado)

#já existe uma função do sklearn para fazer a relação entre as variáveis
from sklearn.metrics import accuracy_score
print('resultado usando a função do sklearn: ', accuracy_score(y, clusters))