import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_predictions(real, prediction, save_path):

    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(real, 'x', color='green', label='Real')
    plt.plot(prediction, 'o', color='red', label='Predição')
    plt.legend()
    plt.title('Resultados da Predição')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.grid(True)

    plt.savefig(os.path.join(save_path, 'resultados_predicao.png'))
    plt.close()

def plot_distributions(data, save_path):
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 12))
    background_color = '#A0D6B4'

    columns_to_plot = ['imdbAverageRating', 'imdbNumVotes', 'releaseYear']

    for i, column in enumerate(columns_to_plot):
        if column in data.columns:
            ax = plt.subplot(len(columns_to_plot), 1, i + 1)
            ax.set_facecolor(background_color)

            sns.kdeplot(data[column], fill=True, color='blue', ax=ax, lw=1.5)
            ax.grid(alpha=0.4, color='white')
            ax.set_title(column)
            ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'distribuicoes_verde_suave.png'))
    plt.close()
