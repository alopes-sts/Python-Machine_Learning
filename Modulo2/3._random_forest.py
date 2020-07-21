from sklearn.datasets import load_breast_cancer
import pandas as pd
pd.set_option('Display.max_columns', 30)
dados = load_breast_cancer()
x = pd.DataFrame(dados.data, columns=[dados.feature_names])
y = pd.Series(dados.target)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold   #é um problema de classificação (poderia ser train_test_split)
from sklearn.model_selection import cross_val_score

#criação do modelo
modelo = RandomForestClassifier(n_estimators=50)
skfold = StratifiedKFold(n_splits=5)
resultado = cross_val_score(modelo, x, y, cv=skfold)
print(resultado.mean())