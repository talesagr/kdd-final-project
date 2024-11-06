from sklearn.preprocessing import StandardScaler  # Importa a classe StandardScaler para escalonamento de dados.
from sklearn.neural_network import MLPRegressor  # Importa o MLPRegressor para criar um modelo de rede neural.


def train_regressor(X_train, y_train):
    regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

    regressor.fit(X_train, y_train)

    return regressor


def scale_data(train_data, test_data):
    scaler = StandardScaler()

    return scaler.fit_transform(train_data), scaler.transform(test_data)
