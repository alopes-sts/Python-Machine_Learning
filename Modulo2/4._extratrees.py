import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', 30)
dados = load_breast_cancer()
x = pd.DataFrame(dados.data, columns=[dados.feature_names])
y = pd.Series(dados.target)

#criação do modelo
modelo = ExtraTreesClassifier(n_estimators=50)
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo, x, y, cv=skfold)
print(resultado.mean())
