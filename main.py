# Importação das funções de preparação, treinamento e visualização de módulos internos.
from src.data_preparation import load_data, clean_data, fill_missing_values
from src.model_training import scale_data, train_regressor
from src.visualization import plot_predictions, plot_distributions
from src.utils.filesystem import create_directory

DATA_FILE = 'data/data.csv'
RESULT_DIR = 'results'


def main():
    create_directory(RESULT_DIR)

    data = load_data(DATA_FILE)
    data = clean_data(data)
    data = fill_missing_values(data)

    data['title'] = data['title'].fillna("Unknown")
    data['genres'] = data['genres'].fillna("Unknown")
    data['type'] = data['type'].fillna("Unknown")
    data['availableCountries'] = data['availableCountries'].fillna("Unknown")
    data['imdbId'] = data['imdbId'].fillna("Unknown")
    data['imdbAverageRating'] = data['imdbAverageRating'].fillna(7)

    if data.isna().sum().any():
        print("Valores NaN restantes por coluna após imputação:", data.isna().sum())
        data = data.dropna()

    if 'releaseYear' in data.columns:
        train_data = data[data['releaseYear'] < 2009]
        test_data = data[data['releaseYear'] >= 2009]
    else:
        print("A coluna 'releaseYear' não está presente no dataset.")
        return

    X_train, X_test = scale_data(train_data.iloc[:, 2:], test_data.iloc[:, 2:])
    y_train = train_data['imdbAverageRating'].values
    y_test = test_data['imdbAverageRating'].values

    regressor = train_regressor(X_train, y_train)
    predictions = regressor.predict(X_test)

    plot_predictions(y_test, predictions, RESULT_DIR)
    plot_distributions(data, RESULT_DIR)


if __name__ == '__main__':
    main()
