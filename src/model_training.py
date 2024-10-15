from sklearn.preprocessing import StandardScaler  # Importa a classe StandardScaler para escalonamento de dados.
from sklearn.neural_network import MLPRegressor  # Importa o MLPRegressor para criar um modelo de rede neural.


def train_regressor(X_train, y_train):
    """Treina um modelo de regressão usando um MLP (Multilayer Perceptron)."""
    # Cria um regressor de rede neural com uma camada oculta de 100 neurônios,
    # max_iter define o número máximo de iterações para o ajuste do modelo,
    # random_state garante reprodutibilidade nos resultados.
    regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)

    # Ajusta o modelo aos dados de treinamento.
    regressor.fit(X_train, y_train)

    return regressor  # Retorna o modelo treinado.


def scale_data(train_data, test_data):
    """Escalar dados de treino e teste usando StandardScaler."""
    # Cria uma instância do escalador.
    scaler = StandardScaler()

    # Ajusta o escalador aos dados de treinamento e transforma ambos os conjuntos de dados.
    # O método fit_transform é usado para os dados de treinamento,
    # enquanto o transform é usado para os dados de teste para evitar vazamento de dados.
    return scaler.fit_transform(train_data), scaler.transform(test_data)
