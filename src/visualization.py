import matplotlib
# Define o backend do Matplotlib para evitar problemas com interfaces gráficas (Tkinter).
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # Importa a biblioteca pyplot do Matplotlib para plotagem.
import seaborn as sns  # Importa a biblioteca Seaborn para visualizações estatísticas.
import os  # Importa a biblioteca os para interações com o sistema de arquivos.

def plot_predictions(real, prediction, save_path):
    """Gera e salva um gráfico comparando valores reais e predições."""
    # Cria o diretório para salvar o gráfico, se não existir.
    os.makedirs(save_path, exist_ok=True)

    # Cria uma nova figura para o gráfico.
    plt.figure(figsize=(12, 6))
    # Plota os valores reais com um marcador 'x' verde.
    plt.plot(real, 'x', color='green', label='Real')
    # Plota as previsões com um marcador 'o' vermelho.
    plt.plot(prediction, 'o', color='red', label='Predição')
    plt.legend()  # Adiciona a legenda ao gráfico.
    plt.title('Resultados da Predição')  # Define o título do gráfico.
    plt.xlabel('Índice')  # Rotula o eixo X.
    plt.ylabel('Valor')  # Rotula o eixo Y.
    plt.grid(True)  # Ativa a grade no gráfico.
    # Salva o gráfico no caminho especificado como 'resultados_predicao.png'.
    plt.savefig(os.path.join(save_path, 'resultados_predicao.png'))
    plt.close()  # Fecha a figura para liberar memória.

def plot_distributions(data, save_path):
    """Gera e salva vários gráficos de distribuição."""
    # Cria o diretório para salvar os gráficos, se não existir.
    os.makedirs(save_path, exist_ok=True)
    # Cria uma nova figura com um tamanho específico.
    plt.figure(figsize=(8, 12))

    # Define uma cor de fundo suave para os gráficos.
    background_color = '#A0D6B4'

    # Lista das colunas que serão plotadas.
    columns_to_plot = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]

    # Loop para criar subgráficos para cada coluna especificada.
    for i, column in enumerate(columns_to_plot):
        ax = plt.subplot(len(columns_to_plot), 1, i + 1)  # Cria um subplot para cada coluna.
        ax.set_facecolor(background_color)  # Define a cor de fundo do subplot.
        # Plota a distribuição da coluna atual usando um gráfico de densidade (KDE).
        sns.kdeplot(data[column], fill=True, color='blue', ax=ax, lw=1.5)
        ax.grid(alpha=0.4, color='white')  # Adiciona uma grade suave ao subplot.
        ax.set_title(column)  # Define o título do subplot com o nome da coluna.
        ax.tick_params(left=False, bottom=False)  # Remove os ticks dos eixos.

    plt.tight_layout()  # Ajusta o layout para evitar sobreposições.
    # Salva todos os gráficos de distribuição em um único arquivo.
    plt.savefig(os.path.join(save_path, 'distribuicoes_verde_suave.png'))
    plt.close()  # Fecha a figura para liberar memória.
