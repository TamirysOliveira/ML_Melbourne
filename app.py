# -*- coding: utf-8 -*-

# intenção: ao ver diversos exemplos de características de entrada (X) que resultam em diferentes saídas (y), o modelo irá começar
# a buscar e entender padrões nos dados. Uma vez que padrões tenham sido encontrados, o modelo formula um conjunto de regras para
# predizer qual saída y para um conjunto de características (dados) de entrada x ainda não visto pelo modelo
# Analisado aqui: preço de imóveis em Melbourne - Austrália a partir de características dos mesmos
# ML usada -> Aprendizado supervisionado de Regressão

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# buscando arquivo (dataset proveniente do Kaggle)
arquive_path = 'melb_data.csv'
# lendo o arquivo
dataset_melb = pd.read_csv(arquive_path)
# dropando valores ausentes (na -> not available)
dataset_melb = dataset_melb.dropna(axis=0)
# print(dataset_melborne.describe())

# apresentando as colunas
print(dataset_melb.columns)

# Escolhendo a coluna preço - minha saída real (y)
y = dataset_melb.Price
print(y)

# Conjunto de dados / características (X)
dataset_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt']
X = dataset_melb[dataset_features]
print(X.describe())

# Iniciando o modelo de ML de Aprendizado Supervisionado de Regressão e controlando a aleatoriedade durante a construção do modelo (random_state).
# Ao fornecer um valor fixo para random_state, o resultado será previsível, ou seja, se o código for executado várias vezes
# com o mesmo conjunto de dados, obterá exatamente os mesmos resultados
dataset_model = DecisionTreeRegressor(random_state=1)

# Ajustando o modelo aos dados
dataset_model.fit(X, y)

print("\nMaking predictions for the following 5 houses: ")
print(X.head())
print("\nThe predictions are:")
print(dataset_model.predict(X.head()))

# Erro Absoluto Médio | Mean Absolute Error -> erro = valor real - valor previsto
predicted_home_prices = dataset_model.predict(X)
# a função nativa do sklearn d(mean_absolute_error) irá calcular a média dos valores absolutos das diferenças entre os valores reais (y)
# e as previsões de preço (predicted_home_prices). Quanto menor esse valor, melhor a performance
print("Mean Absolute Error on full dataset:")
print(mean_absolute_error(y, predicted_home_prices))

# dividindo o conjunto de dados em subconjuntos de treino e teste com o train_test_split
# a linha de código abaixo divide o conjunto de dados em 4 partes:
# 2 conjuntos de treino e validação das características (train_X e val_X;)
# e 2 conjuntos de treino e validação dos rótulos | saídas (train_y e val_y)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# 2 parâmetros do train_test_split estão ocultos; test_size e train_size.
# Quando isso ocorre, por padrão, test_size é definido como 25% e, por consequência, train_test 75%
dataset_model = DecisionTreeRegressor()
# no fit abaixo - ajustando o modelo aos dados de treino, ou seja, ele está aprendendo a relação entre
# característica (train_X) e as saídas | rótulos (train_y)
dataset_model.fit(train_X, train_y)
# usando o modelo treinado, previsões são feitas para os dados de validação (val_X) usando o método predict()
# isso vai permitir que nosso modelo faça previsões com base nas características dos dados de validação
val_predictions = dataset_model.predict(val_X)
# printando algumas previsões de validações
print("\ntop few validation predictions")
print(val_predictions[:5])
# printando alguns preços reais
print("\ntop few actual prices from validation data")
print(val_y[:5])
print("\nMean Absolute Error:")
# erro absoluto médio entre as previsões (val_predictions) e as saídas/rótulos reais (val_y) dos dados de validação
# isso fornece quão bem o modelo está performando. LEMBRETE: quanto menor o erro médio absoluto, melhor o desempenho
print(mean_absolute_error(val_y, val_predictions))
# devido ao tamanho (pequeno) do conjunto de dados, apesar do random_state setado em zero, o MAE mostra algumas diferenças a cada iteração
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: {} \t\t Mean Absolute Error: {}".format(max_leaf_nodes, my_mae))

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_forest_preds = forest_model.predict(val_X)
print("\nMean Absolute Error / Random Forest Regressor")
print (mean_absolute_error(val_y, melb_forest_preds))
