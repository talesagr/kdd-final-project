# Importar bibliotecas necessárias
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' para não depender do Tkinter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# Definir caminhos dos arquivos
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
result_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

# Leitura do dataset
data = pd.read_csv(os.path.join(data_dir, 'household_power_consumption.txt'), sep=';', low_memory=False)

# Conversão de '?' para NaN
data.replace('?', np.nan, inplace=True)

# Converter colunas numéricas para float
for column in data.columns:
    if column not in ['Date', 'Time']:
        data[column] = pd.to_numeric(data[column], errors='coerce')

# Verificar valores nulos
missing_data = data.isnull().sum()
print("Valores nulos por coluna:\n", missing_data)

# Remover datas com todos os valores NaN
data_cleaned = data.dropna(how='all', subset=data.columns[1:])

# Verificar se ainda há valores nulos após a limpeza
if data_cleaned.isnull().values.any():
    for column in data_cleaned.columns[2:]:  # Começar do índice 2 para ignorar 'Date' e 'Time'
        # Atribuindo a média diretamente
        data_cleaned[column] = data_cleaned[column].fillna(data_cleaned[column].mean())

# Convertendo a coluna de 'Date' para datetime
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%d/%m/%Y')

# Verificar as datas disponíveis
print("Datas disponíveis no conjunto de dados:", data_cleaned['Date'].unique())

# Separar dados de treino e teste
treino = data_cleaned[(data_cleaned['Date'] >= '2006-12-17') & (data_cleaned['Date'] <= '2008-12-31')].reset_index(drop=True)
teste = data_cleaned[(data_cleaned['Date'] >= '2009-01-01') & (data_cleaned['Date'] <= '2010-11-25')].reset_index(drop=True)

# Verificar se os dados de treino e teste não estão vazios
if treino.empty or teste.empty:
    raise ValueError("Os conjuntos de dados de treino ou teste estão vazios!")

# Inicialização de listas
listSet = []
listTeach = []
listevidence = []
real = []

# Criar dataset de treino e labels
for i in range(len(treino) // 1440 - 1):
    listSet.append([
        treino['Sub_metering_1'][i*1440:(i+1)*1440].mean(),
        treino['Sub_metering_2'][i*1440:(i+1)*1440].mean(),
        treino['Sub_metering_3'][i*1440:(i+1)*1440].mean()
    ])
    # Labels correspondentes
    listTeach.append(i % 3 + 1)  # Adiciona 1, 2 ou 3 para as labels de cada sub-metro

# Criar dataset de teste
for i in range(len(teste) // 1440 - 1):
    listevidence.append([
        teste['Sub_metering_1'][i*1440:(i+1)*1440].mean(),
        teste['Sub_metering_2'][i*1440:(i+1)*1440].mean(),
        teste['Sub_metering_3'][i*1440:(i+1)*1440].mean()
    ])
    # Labels correspondentes
    real.append(i % 3 + 1)  # Adiciona 1, 2 ou 3 para as labels de cada sub-metro

# Verifica se os conjuntos de dados foram preenchidos corretamente
if not listSet or not listevidence:
    raise ValueError("Os datasets de treino ou teste estão vazios!")

# Escalar os dados
scaler = StandardScaler()
listSet_scaled = scaler.fit_transform(listSet)
listevidence_scaled = scaler.transform(listevidence)

# Configurar o classificador MLP com mais iterações
clf = MLPClassifier(solver='lbfgs', alpha=1e-15, hidden_layer_sizes=(17,), random_state=1, max_iter=1000)

# Treinar o classificador
clf.fit(listSet_scaled, listTeach)

# Prever com os dados de teste
prediction = clf.predict(listevidence_scaled)

# Avaliar a acurácia
acuracia = accuracy_score(real, prediction)
print(f'A acurácia do classificador utilizando MLP é: {acuracia:.4f}')

# Plotar os resultados
plt.figure(figsize=(12, 6))
plt.plot(real, 'x', color='green', label='Real')
plt.plot(prediction, 'o', color='red', label='Predição')
plt.xlim(0, len(real))  # Ajustado para o comprimento de 'real'
plt.legend()
plt.title('Resultados da Predição')
plt.xlabel('Amostra')
plt.ylabel('Classe')
plt.savefig(os.path.join(result_dir, 'resultados_predicao.png'))  # Salvar o gráfico no diretório de resultados
plt.close()  # Fechar a figura após salvar

# Medidas de posição
print("Medidas de Posição:")
print(data.describe().round(3))

# Medidas de dispersão
print("Medidas de Dispersão:")
est_disp = data.describe().transpose()
print(est_disp[['mean', 'std', '25%', '50%', '75%']].round(3))

# Salvar o dataset após imputação
with open(os.path.join(data_dir, "household_power_consumption_completo.p"), "wb") as f:
    pickle.dump(data_cleaned, f)
