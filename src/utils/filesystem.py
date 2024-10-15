import os  # Importa a biblioteca os para interações com o sistema operacional.

def create_directory(path):
    """Cria um diretório se ele não existir."""
    # Verifica se o diretório especificado já existe.
    if not os.path.exists(path):
        # Cria o diretório, incluindo qualquer diretório pai necessário.
        os.makedirs(path)  # Cria o diretório se não existir.
