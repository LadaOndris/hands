import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from src.gestures.relative_distance.base import _upper_tri, hand_pose_angles, scaled_distance_matrix, viewpoint_angles
from src.datasets.msra.dataset import load_dataset


def features_as_distances(joints):
    matrix = scaled_distance_matrix(joints)
    features = _upper_tri(matrix)
    return features


def features_as_angles(joints):
    angles = hand_pose_angles(joints)
    features = np.reshape(angles, (np.shape(angles)[0], -1))
    return features


def viewpoint_features(joints):
    angles = viewpoint_angles(joints)
    features = np.reshape(angles, (np.shape(angles)[0], -1))
    return features


gesture_names, joints, labels = load_dataset(shuffle=True)
# joints = np.reshape(joints, [-1, 21 * 3])

n = 5000
joints = joints[:n]
labels = labels[:n]

features = viewpoint_features(joints)

tsne = TSNE(n_components=2)
tsne_res = tsne.fit_transform(features)

sns.scatterplot(x=tsne_res[:, 0],
                y=tsne_res[:, 1],
                hue=labels,
                palette=sns.hls_palette(len(gesture_names)),
                legend='full')
plt.show()

pass
