import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

from src.gestures.sklearn_classifiers import load_data


def plot(name: str, database_name: str, x, y):
    n_classes = np.unique(y).shape[0]
    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], s=10, c=y, cmap=plt.cm.get_cmap('jet', n_classes))
    ax.set_title(f'{name} of "{database_name}" gesture database')
    ax.axis('tight')
    plt.xticks([], [])
    plt.yticks([], [])
    fig.tight_layout()
    plt.savefig(f'docs/readme/{name}.png')
    fig.show()


def visualize_TSNE(x, y, database_name: str):
    tsne = TSNE(n_components=2)
    x_transformed = tsne.fit_transform(x)
    plot('t-SNE', database_name, x_transformed, y)


def visualize_LDA(x, y, database_name: str):
    x_transformed = LDA(n_components=2).fit_transform(x, y)
    plot(f'LDA', database_name, x_transformed, y)


if __name__ == "__main__":
    gesture_database_name = 'color'
    x, y = load_data(gesture_database_name)
    visualize_TSNE(x, y, gesture_database_name)
    visualize_LDA(x, y, gesture_database_name)
