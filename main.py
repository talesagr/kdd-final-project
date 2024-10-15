# Importação das funções de preparação, treinamento e visualização de módulos internos.
from src.data_preparation import load_data, clean_data, fill_missing_values
from src.model_training import scale_data, train_regressor
from src.visualization import plot_predictions, plot_distributions
from src.utils.filesystem import create_directory

# Caminho do arquivo de dados e diretório de resultados.
DATA_FILE = 'data/household_power_consumption.txt'
RESULT_DIR = 'results'

def main():
    # Cria o diretório de resultados se não existir.
    create_directory(RESULT_DIR)

    # Carrega os dados a partir do arquivo especificado.
    data = load_data(DATA_FILE)
    # Realiza a limpeza básica dos dados.
    data = clean_data(data)
    # Preenche valores ausentes com a média das colunas.
    data = fill_missing_values(data)

    # Separa os dados em treino (antes de 2009) e teste (a partir de 2009).
    train_data = data[data['Date'] < '2009-01-01']
    test_data = data[data['Date'] >= '2009-01-01']

    # Escala as colunas (exceto as duas primeiras) para padronizar os dados.
    X_train, X_test = scale_data(train_data.iloc[:, 2:], test_data.iloc[:, 2:])
    # Extrai a variável alvo ('Global_active_power') para treino e teste.
    y_train = train_data['Global_active_power'].values
    y_test = test_data['Global_active_power'].values

    # Treina um modelo de regressão com os dados de treino.
    regressor = train_regressor(X_train, y_train)

    # Gera previsões com o modelo treinado.
    predictions = regressor.predict(X_test)
    # Cria um gráfico comparando valores reais e previstos.
    plot_predictions(y_test, predictions, RESULT_DIR)
    # Gera múltiplos gráficos de distribuições dos dados.
    plot_distributions(data, RESULT_DIR)

# Verifica se o script está sendo executado diretamente.
if __name__ == '__main__':
    main()
