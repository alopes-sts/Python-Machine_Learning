# Exemplo de regressão logistica para problemas de classificação usando as funções de 'roc' e 'auc roc'
# O dataset faz parte da biblioteca do sklearn, referente a dados de cancer de mama

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

pd.set_option('display.max_columns', 30)
dados = load_breast_cancer()

# o dataset está em um formato específico do sklearn, por isso as variáveis estão sendo definidas desta forma
# o dataset do sklearn será convertido para um dataframe pandas
# as informações que estão no parâmetro 'columns' é padrão do sklearn, esta é a identificação das colunas (features_names)
x = pd.DataFrame(dados.data, columns=[dados.feature_names])
# no dataset do sklearn já existe uma coluna com a informação de 'target', basta associar com a variável y
y = pd.Series(dados.target)

# este comando mostra o total de cada dado na coluna target da planilha, neste caso vai ser zero(negativo para cancer) e 1(positivo para cancer)
#  print(y.value_counts())

# separando dados para treino e teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(x, y, test_size=0.3, random_state=9)

# criação do modelo
modelo = LogisticRegression(C=95, penalty='l2', max_iter=6000)
modelo.fit(X_treino, Y_treino)

# score
resultado = modelo.score(X_teste, Y_teste)
print('Acurácia:', resultado)

# algoritmo usado para o modelo ROC/AUC
predicoes = modelo.predict_proba(X_teste)
# comando apenas para efeitos didáticos
print(predicoes)

# excluido a segunda coluna (não nos interessa por enquanto)
probs = predicoes[:, 1]

# fpr (false positive rates) tpr (true positive rates)
# passando para a função roc_curve os valores reais (Y_teste) e os valores previstos (probs)
fpr, tpr, thresholds = roc_curve(Y_teste, probs)

print('TPR:', tpr)
print('FPR:', fpr)
print('Threshoulds:', thresholds)

# visualizando graficamente os dados
plt.scatter(fpr,tpr)
plt.show()

# verifica o resultado da curva, quanto maior o resultado melhor (máximo=1)
print('\n\nResultado da curva',roc_auc_score(Y_teste,probs))


