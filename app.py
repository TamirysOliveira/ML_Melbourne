import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Função para calcular o erro absoluto médio (MAE)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Leitura do arquivo CSV
@st.cache_data
def load_data():
    return pd.read_csv('melb_data.csv')

# Carregamento dos dados
data = load_data()

# Escolha das características (features)
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt']

# Separando as features e o alvo
X = data[features]
y = data['Price']

# Divisão dos dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de Regressão de Árvore de Decisão
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Título do aplicativo
st.title('Melbourne Housing Price Prediction')

# Descrição do aplicativo
st.write("""
Enter the property details and we'll predict the price for you!
""")

# Inputs do usuário
num_rooms = st.number_input('Number of Rooms', min_value=1, max_value=10, value=1, step=1)
num_bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=1, step=1)
land_size = st.number_input('Land Size (sqm)', value=100)
building_area = st.number_input('Building Area (sqm)', value=100)
year_built = st.number_input('Year Built', min_value=1800, max_value=2022, value=2000)

# Previsão do preço com base nos inputs do usuário
predicted_price = model.predict([[num_rooms, num_bathrooms, land_size, building_area, year_built]])

# Exibição da previsão
st.write(f'Predicted Price: ${predicted_price[0]:,.2f} (AUD)')
