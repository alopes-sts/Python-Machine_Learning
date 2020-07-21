from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# usando um dos datasets padrão do sklearn
iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

# separando os dados em folds
skfold = StratifiedKFold(n_splits=5, random_state=8)

# criação do modelo e usando validação cruzada
modelo = DecisionTreeClassifier()
resultado = cross_val_score(modelo, x, y, cv=skfold)

# imprimindo a acurácia
print (resultado.mean())

# usando graphviz para visualizar a árvore de decisões
import graphviz
from sklearn.tree import export_graphviz

# criando arquivo que irá armazenar a árvore
arquivo = 'C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Classificação/datasets/exemplo_graphviz_decision_tree.dot'
modelo.fit(x, y)

# gerando o gráfico da árvore de decisão
# modelo = é o modelo das dados após o treino (modelo.fit)
# arquivo = nome do arquivo de saída que foi criado ".dot"
# feature_names = é o nome das colunas do dataset, conferir na linha 9 onde declara a variável 'x'
export_graphviz(melhor_modelo, out_file=arquivo2, feature_names=iris.feature_names)
with open(arquivo2) as aberto:
    grafico_dot = aberto.read()
h = graphviz.Source(grafico_dot)
h.view()



