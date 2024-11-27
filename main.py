from src.data_preparation import load_data, clean_data, fill_missing_values
from src.model_training import scale_data, train_regressor
from src.visualization import plot_predictions, plot_distributions
from src.utils.filesystem import create_directory
import numpy as np

DATA_FILE = 'data/data_2.csv'
RESULT_DIR = 'results'


def main():
    create_directory(RESULT_DIR)

    data = load_data(DATA_FILE)
    data = clean_data(data)
    data = fill_missing_values(data)

    data['title'] = data['title'].fillna("Unknown")
    data['genres'] = data['genres'].fillna("Unknown")
    data['type'] = data['type'].fillna("movie")
    data['availableCountries'] = data['availableCountries'].fillna("BR")
    data['imdbId'] = data['imdbId'].fillna("tt0000000")
    data['imdbAverageRating'] = data['imdbAverageRating'].fillna(7.0)
    data['imdbNumVotes'] = data['imdbNumVotes'].fillna(1)
    data['NumVotesPerYear'] = data['imdbNumVotes'] / (2024 - data['releaseYear'])
    data['GenresCount'] = data['genres'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    data['PopularityScore'] = data['imdbAverageRating'] * 0.7 + \
                              (data['imdbNumVotes'] / data['imdbNumVotes'].max()) * 0.3


    if data.isna().sum().any():
        data = data.dropna()

    if 'releaseYear' in data.columns:
        train_data = data[data['releaseYear'] < 2009]
        test_data = data[data['releaseYear'] >= 2009]
    else:
        return

    numeric_cols = train_data.select_dtypes(include=['float64', 'int64']).columns

    train_data.loc[:, numeric_cols] = train_data.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)
    test_data.loc[:, numeric_cols] = test_data.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)

    train_data.loc[:, numeric_cols] = train_data.loc[:, numeric_cols].fillna(train_data[numeric_cols].mean())
    test_data.loc[:, numeric_cols] = test_data.loc[:, numeric_cols].fillna(test_data[numeric_cols].mean())

    X_train, X_test = scale_data(train_data[numeric_cols], test_data[numeric_cols])

    y_train = train_data['imdbAverageRating'].values
    y_test = test_data['imdbAverageRating'].values

    regressor = train_regressor(X_train, y_train)
    predictions = regressor.predict(X_test)

    plot_predictions(y_test, predictions, RESULT_DIR)
    plot_distributions(data, RESULT_DIR)


if __name__ == '__main__':
    main()
