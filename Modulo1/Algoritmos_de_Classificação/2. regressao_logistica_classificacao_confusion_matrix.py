# Exemplo de regressão logistica para problemas de classificação usando 'confusion matrix'
# O dataset faz parte da biblioteca do sklearn, referente a dados de cancer de mama

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

# cria um array com os dados de teste para ser comparado com os dados gerados pela função confusion_matrix
predicao = modelo.predict(X_teste)
# impressao dos dados apenas para fins de verificação
#  print(predicao)

# cria a matriz da função confusion_matriz  comparando os dados do array de teste com o array do modelo
# esse array vai mostrar a soma dos resultados de falso positivo, falso negativo, etc, array de 2x2
matriz = confusion_matrix(Y_teste, predicao)
print(matriz)


