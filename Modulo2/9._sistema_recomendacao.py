import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import surprise
from surprise import accuracy
from surprise import KNNBasic
from collections import defaultdict

#Criando uma função que retorna as top-N recomendações para cada usuário
#Essa função irá retornar um dicionário onde as chaves são os usuários e os valores são as listas de tuplas (filme, rating_previstos)
def obtem_top_n(previsoes, n=5):
    top_n = defaultdict(list) # cria um dicionário onde os vaalores são listas vazias
    for usuario, filme, _, previsao, _ in previsoes: #O 'underline' significa que o valor referente a este campo não nos interessa e não será salvo em nenhuma variável
        top_n[usuario].append((filme, previsao)) #adiciona os pares de chave:valor ao dicionário
    for usuario, previsoes_usuario in top_n.items():
        previsoes_usuario.sort(key=lambda x: x[1], reverse=True) #ordena as previsões de rating do maior para o menor
        top_n[usuario] = previsoes_usuario[:n] #salva somente os n primeiros valores (n foi passado como valor na chamada da função)
    return top_n

dataset = pd.read_csv('C:/Onedrive/Curso_Machine_Learning/Modulo 2/dataset_filmes/ratings.txt', sep=" ", names = ['id_usuario', 'id_filme', 'rating']) #o parâmetro 'sep' é o separador dos campos que está no dataset, neste caso é um espaço em branco
print(dataset.head())

filmes = len(dataset['id_filme'].unique()) #vai mostrar o total de filmes que existe na planilha, o parâmetro 'unique' é para não somar filmes repetidos
usuarios = len(dataset['id_usuario'].unique())
amostras = dataset.shape[0]
print('Total de filmes: ', filmes)
print('Ttoal de usuários: ', usuarios)
print('Total de amostras: ', amostras)

dataset['rating'].value_counts().plot(kind='bar')
plt.show()

#neste algoritmo é importante saber qual a escala iremos trabalhar, por isso é necessário verificar o menor e o maior valor
menor_rating = dataset['rating'].min() #exibe o menor valor da coluna 'rating'
maior_rating = dataset['rating'].max()
print('Variação de rating: {0} a {1}'.format(menor_rating, maior_rating)) #os parâmetros {0} e {1} significam que iremos pegar o menor e o maior valor, que são passados pelo format()

#Começando a criar o algoritmo do SVD++
#Redefinindo o range de ratings de acordo com os valores verificados acima
reader = surprise.Reader(rating_scale=(0.5, 4.0)) #pode carregar um dataset bult-in também
dataset_surprise = surprise.Dataset.load_from_df(dataset, reader)

#Escolhendo o algoritmo e treinando o modelo
dataset_preenchido = dataset_surprise.build_full_trainset() #Cria o dataset de treino
algoritmo = surprise.SVDpp(n_factors=20) #algoritmo SVD++ (chamado de SVDpp)
algoritmo.fit(dataset_preenchido)

dataset_missing = dataset_preenchido.build_testset()
previsoes = algoritmo.test(dataset_missing)

print(previsoes[0])
print(len(previsoes))

top_5 = obtem_top_n(previsoes, n=5)  #Chamando a função 'obtem_top_n'
print(top_5)

for usuario, previsoes_usuario in top_5.items():
    print(usuario, [filme for (filme, _) in previsoes_usuario]) #irá mostrar as 5 previsões para cada usuário, são os filmes que devem ser recomendados quando o usuário

#Fazendo uma previsão somente para um usuário e filme especificos (somente um exemplo de uso)
previsao_usuario = algoritmo.predict(uid='30', iid='87')
rating = previsao_usuario.est
print(rating)

#Validando modelo
from surprise.model_selection import train_test_split
dataset_treino, dataset_teste = train_test_split(dataset_surprise, test_size=0.3)
algoritmo = surprise.SVDpp(n_factors=20)
algoritmo.fit(dataset_preenchido)
previsoes_gerais = algoritmo.test(dataset_teste)

print(previsoes_gerais)

print (accuracy.rmse(previsoes_gerais)) # vai mostrar o raiting de quanto o modelo está errando, para cima ou para baixo

#ajustando os parâmetros para tentar melhor o resultado visto acima
param_grid = {'lr_all': [.007, .01, 0.05, 0.001], 'reg_all': [0.02, 0.1, 1.0, 0.005]}
surprise_grid = surprise.model_selection.GridSearchCV(surprise.SVDpp, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
surprise_grid.fit(dataset_surprise)
print(surprise_grid.best_params['rmse'])  #irá mostrar o melhor learning rate e o melhor parâmetro de regularização

#Mostrando os dados vizinhos
dataset_preenchido = dataset_surprise.build_full_trainset()
algoritmo = KNNBasic(sim_options={'name': 'cosine', 'user_based': False}) #o algoritmo irá encontrar os vizinhos mais próximos usando o algoritmo 'cosine'
#'name' é o algoritmo de similaridade, 'user_based'==True é apra calcular a similaridade entre usuários
algoritmo.fit(dataset_preenchido)

#Mostrando os k vizinhos mais próximos
vizinhos = algoritmo.get_neighbors(343, k=10) #foi pego um id de usuário aleatório para exemplo (343) e irá mostrar os 10 vizinhos mais próximos

print('Os 10 filmes vizinhos para o id escolhido são:')
for filme in vizinhos:
    print(filme)





