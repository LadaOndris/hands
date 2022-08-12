from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

from system.database.reader import UsecaseDatabaseReader


def load_data(gesture_database_name: str) -> Tuple:
    reader = UsecaseDatabaseReader()
    reader.load_from_subdir(gesture_database_name)

    x = reader.hand_poses
    y = reader.labels

    x_flattened = np.reshape(x, [x.shape[0], -1])
    y = y.astype(int)
    return x_flattened, y


def plot(name: str, x, y):
    n_classes = np.unique(y).shape[0]
    fig, ax = plt.subplots()
    ax.scatter(x[:, 0], x[:, 1], s=1, c=y, cmap=plt.cm.get_cmap('jet', n_classes))
    ax.set_title(f'{name} of gesture_recognizers')
    ax.axis('tight')
    fig.show()


def visualize_TSNE(x, y):
    tsne = TSNE(n_components=2)
    x_transformed = tsne.fit_transform(x)
    plot('t-SNE', x_transformed, y)


def visualize_LDA(x, y):
    x_transformed = LDA(n_components=2).fit_transform(x, y)
    plot('LDA', x_transformed, y)


gesture_database_name = 'color'
x, y = load_data(gesture_database_name)
visualize_TSNE(x, y)
visualize_LDA(x, y)
