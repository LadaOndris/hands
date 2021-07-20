import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from src.acceptance.base import _upper_tri, scaled_distance_matrix
from src.datasets.msra.dataset import load_dataset


def features_as_distances():
    matrix = scaled_distance_matrix(joints)
    features = _upper_tri(matrix)
    return features

def features_as_angles():


gesture_names, joints, labels = load_dataset(shuffle=True)
# joints = np.reshape(joints, [-1, 21 * 3])

n = 10000
joints = joints[:n]
labels = labels[:n]

features = features_as_distances()

tsne = TSNE(n_components=2)
tsne_res = tsne.fit_transform(features)

sns.scatterplot(x=tsne_res[:, 0],
                y=tsne_res[:, 1],
                hue=labels,
                palette=sns.hls_palette(len(gesture_names)),
                legend='full')
plt.show()

pass
