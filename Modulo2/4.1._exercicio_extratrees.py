import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', 23)
arquivo = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 2/mushroom_dataset.csv')
arquivo['mushroom'] = arquivo['mushroom'].replace('EDIBLE','0')
arquivo['mushroom'] = arquivo['mushroom'].replace('POISONOUS','1')

#verificando se existem dados faltantes
#faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['cap-shape'])) * 100
#print(faltantes_percentual)

#verificando a quantidade de informações em cada coluna, se houver apenas 2 em cada coluna,
#podemos usar o método 'one hot encode'
#print([arquivo[c].value_counts() for c in list(arquivo.columns)])

#convertendo variáveis em colunas, o drop_first remove colunas duplicadas
arquivo_encode = pd.get_dummies(arquivo, drop_first=False)
arquivo_encode.rename(columns={'mushroom_1':'comestivel'}, inplace=True)  #renomeando a variável target apenas para melhor a visualização

#criando variáveis preditoras e variável target
x = arquivo_encode.drop('comestivel', axis=1)
y = arquivo_encode['comestivel']

#criação do modelo
modelo = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
skfold = StratifiedKFold(n_splits=3, shuffle=True)   #shuffle = embaralha as amostras É UMA BOA PRÁTICA USAR ESSE PARÂMETRO
resultado = cross_val_score(modelo, x, y, cv=skfold)
print('\nO resultado foi: ',resultado.mean())





