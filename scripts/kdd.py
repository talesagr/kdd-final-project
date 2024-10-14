# Importar bibliotecas necessárias
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' para não depender do Tkinter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# Definir caminhos dos arquivos
data_file = 'C:\\projetos\\kdd\\data\\household_power_consumption.txt'
result_dir = 'C:\\projetos\\kdd\\results'  # Adicione este caminho para salvar resultados

# Leitura do dataset
data = pd.read_csv(data_file, sep=';', low_memory=False)

# Conversão de '?' para NaN
data.replace('?', np.nan, inplace=True)

# Converter colunas numéricas para float
for column in data.columns:
    if column not in ['Date', 'Time']:
        data[column] = pd.to_numeric(data[column], errors='coerce')

# Remover datas com todos os valores NaN
data_cleaned = data.dropna(how='all', subset=data.columns[1:])

# Preencher valores faltantes com a média
for column in data_cleaned.columns[2:]:
    data_cleaned[column].fillna(data_cleaned[column].mean(), inplace=True)

# Convertendo a coluna de 'Date' para datetime
data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%d/%m/%Y')

# Separar dados de treino e teste
treino = data_cleaned[(data_cleaned['Date'] >= '2006-12-17') & (data_cleaned['Date'] <= '2008-12-31')].reset_index(drop=True)
teste = data_cleaned[(data_cleaned['Date'] >= '2009-01-01') & (data_cleaned['Date'] <= '2010-11-25')].reset_index(drop=True)

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
    listTeach.append(i % 3 + 1)  # Adiciona 1, 2 ou 3 para as labels de cada sub-metro

# Criar dataset de teste
for i in range(len(teste) // 1440 - 1):
    listevidence.append([
        teste['Sub_metering_1'][i*1440:(i+1)*1440].mean(),
        teste['Sub_metering_2'][i*1440:(i+1)*1440].mean(),
        teste['Sub_metering_3'][i*1440:(i+1)*1440].mean()
    ])
    real.append(i % 3 + 1)  # Adiciona 1, 2 ou 3 para as labels de cada sub-metro

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

# Gráfico de Previsão (Real vs Predição)
plt.figure(figsize=(12, 6))
plt.plot(real, 'x', color='green', label='Real')
plt.plot(prediction, 'o', color='red', label='Predição')
plt.xlim(0, len(real))
plt.ylim(0.5, 3.5)  # Limitar o eixo Y para as classes 1, 2, 3
plt.legend()
plt.title('Resultados da Predição')
plt.xlabel('Amostra')
plt.ylabel('Classe')
plt.grid(True)
plt.savefig(os.path.join(result_dir, 'resultados_predicao.png'))

# Fechar a figura
plt.close()

# Gráficos de Distribuição de Frequência com ajustes de estilo
plt.figure(figsize=(8, 12))

# Definir o fundo verde para todos os gráficos
background_color = '#A0D6B4'  # Cor verde clara

# Subplots ajustados para diferentes colunas
ax1 = plt.subplot(711)
ax1.set_facecolor(background_color)  # Cor de fundo
sns.kdeplot(data_cleaned['Global_active_power'], fill=True, color='blue', ax=ax1, lw=1.5)  # Suavizar linhas e preencher
ax1.grid(alpha=0.4, color='white')
ax1.set_title('Global_active_power')
ax1.tick_params(left=False, bottom=False)  # Remover ticks

ax2 = plt.subplot(712)
ax2.set_facecolor(background_color)
sns.kdeplot(data_cleaned['Global_reactive_power'], fill=True, color='blue', ax=ax2, lw=1.5)
ax2.grid(alpha=0.4, color='white')
ax2.set_title('Global_reactive_power')
ax2.tick_params(left=False, bottom=False)

ax3 = plt.subplot(713)
ax3.set_facecolor(background_color)
sns.kdeplot(data_cleaned['Voltage'], fill=True, color='blue', ax=ax3, lw=1.5)
ax3.grid(alpha=0.4, color='white')
ax3.set_title('Voltage')
ax3.tick_params(left=False, bottom=False)

ax4 = plt.subplot(714)
ax4.set_facecolor(background_color)
sns.kdeplot(data_cleaned['Global_intensity'], fill=True, color='blue', ax=ax4, lw=1.5)
ax4.grid(alpha=0.4, color='white')
ax4.set_title('Global_intensity')
ax4.tick_params(left=False, bottom=False)

ax5 = plt.subplot(715)
ax5.set_facecolor(background_color)
sns.kdeplot(data_cleaned['Sub_metering_1'], fill=True, color='blue', ax=ax5, lw=1.5)
ax5.grid(alpha=0.4, color='white')
ax5.set_title('Sub_metering_1')
ax5.tick_params(left=False, bottom=False)

ax6 = plt.subplot(716)
ax6.set_facecolor(background_color)
sns.kdeplot(data_cleaned['Sub_metering_2'], fill=True, color='blue', ax=ax6, lw=1.5)
ax6.grid(alpha=0.4, color='white')
ax6.set_title('Sub_metering_2')
ax6.tick_params(left=False, bottom=False)

ax7 = plt.subplot(717)
ax7.set_facecolor(background_color)
sns.kdeplot(data_cleaned['Sub_metering_3'], fill=True, color='blue', ax=ax7, lw=1.5)
ax7.grid(alpha=0.4, color='white')
ax7.set_title('Sub_metering_3')
ax7.tick_params(left=False, bottom=False)

plt.tight_layout()
plt.savefig(os.path.join(result_dir, 'distribuicoes_verde_suave.png'))

# Fechar o gráfico
plt.close()
