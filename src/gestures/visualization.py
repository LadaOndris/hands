import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


from system.database.reader import UsecaseDatabaseReader


def visualize(x, y):
    n_classes = np.unique(y).shape[0]
    # tsne = TSNE(n_components=2)
    # x_transformed = tsne.fit_transform(x)
    x_transformed = LDA(n_components=2).fit_transform(x, y)
    fig, ax = plt.subplots()
    ax.scatter(x_transformed[:, 0], x_transformed[:, 1], s=1, c=y, cmap=plt.cm.get_cmap('jet', n_classes))
    ax.set_title('t-SNE of gesture_recognizers')
    ax.axis('tight')
    fig.show()

reader = UsecaseDatabaseReader()
reader.load_from_subdir('demo')

x = reader.hand_poses
y = reader.labels

x_flattened = np.reshape(x, [x.shape[0], -1])
y = y.astype(int)
visualize(x_flattened, y)
