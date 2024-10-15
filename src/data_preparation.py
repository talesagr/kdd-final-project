import pandas as pd  # Importa a biblioteca pandas para manipulação de dados.
import numpy as np  # Importa a biblioteca NumPy para operações numéricas.


def load_data(file_path):
    """Carregar dados de um arquivo CSV e substituir '?' por NaN."""
    # Carrega os dados de um arquivo CSV especificado, usando ';' como delimitador.
    data = pd.read_csv(file_path, sep=';', low_memory=False)

    # Substitui os valores '?' por NaN (Not a Number) para facilitar o tratamento de dados ausentes.
    data.replace('?', np.nan, inplace=True)

    return data  # Retorna o DataFrame carregado.


def clean_data(data):
    """Converter colunas para numéricas e remover linhas com NaN em excesso."""
    # Itera sobre cada coluna do DataFrame.
    for column in data.columns:
        # Converte colunas para o tipo numérico, exceto as colunas 'Date' e 'Time'.
        if column not in ['Date', 'Time']:
            data[column] = pd.to_numeric(data[column], errors='coerce')  # Converte e substitui erros por NaN.

    # Remove linhas onde todas as colunas, exceto 'Date', são NaN (todas as colunas numéricas estão ausentes).
    data = data.dropna(how='all', subset=data.columns[1:])

    return data  # Retorna o DataFrame limpo.


def fill_missing_values(data):
    """Preencher valores ausentes com a média das colunas numéricas."""
    # Itera sobre as colunas a partir da terceira coluna do DataFrame.
    for column in data.columns[2:]:
        # Verifica se a coluna não está completamente ausente (não tem apenas NaN).
        if not data[column].isnull().all():
            # Preenche os valores ausentes com a média da coluna.
            data[column] = data[column].fillna(data[column].mean())

    return data  # Retorna o DataFrame com valores ausentes preenchidos.
