# Exemplo de regressão linear usando um dataset para analisar de preço de casas
# Download do dataset: https://www.kaggle.com/harlfoxem/housesalesprediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', 21)  # força a exibição de todas as colunas

# importando arquivo com os dados e removendo as colunas indesejadas
dataframe_casas = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 1/Algoritmos_de_Regressao/datasets/kc_house_data.csv')
dataframe_casas = dataframe_casas.drop(['id', 'date', 'zipcode', 'lat', 'long'], axis=1)

# checando se existem campos em branco, apenas para efeitoss didaticos
if dataframe_casas.isnull().values.any() == False:
    print('Não existem campos em branco na planilha, o programa irá continuar')
    print()
else:
    print('Existem campos em branco na planilha que podem gerar inconsistências no resultado')
    print('Corrija a planilha antes de continuar')
    quit()


# definindo variaveis preditoras e variável target
y_preco = dataframe_casas['price']
x_dados_casas = dataframe_casas.drop('price', axis=1)

# criando dados de teste e treino
# test_size= 0.3 (pega 30% dos dados para teste)
# random_state=10 (define que os valores de testes serão sempre os mesmo todas as vezes que o programa for executado
x_dados_casas_treino, x_dados_casas_teste, y_preco_treino, y_preco_teste = train_test_split(x_dados_casas, y_preco, test_size=0.3, random_state=10)

# criando o modelo de regressao linear
modelo = LinearRegression()
modelo.fit(x_dados_casas_treino, y_preco_treino)

# calculo do coeficiente R2
resultado = modelo.score(x_dados_casas_teste, y_preco_teste)
print('O resultado do cálculo do coeficiente R2 foi:', resultado)



