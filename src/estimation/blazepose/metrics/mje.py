import tensorflow as tf
from tensorflow.keras.metrics import Metric

from src.utils.camera import Camera


class MeanJointErrorMetric(Metric):

    def __init__(self, camera: Camera, name='mje', **kwargs):
        super(MeanJointErrorMetric, self).__init__(name=name, **kwargs)
        self.camera = camera
        self._total = self.add_weight(name='total', initializer='zeros')
        self._count = self.add_weight(name='count', initializer='zeros')

    def reset_states(self):
        self._total.assign(0.)
        self._count.assign(0.)

    def result(self):
        return tf.math.divide_no_nan(self._total, self._count)

    def update_state(self, uvz_true, uvz_pred, sample_weight=None):
        xyz_true = self.camera.world_to_pixel_3d(uvz_true[..., :3])
        xyz_pred = self.camera.world_to_pixel_3d(uvz_pred[..., :3])

        mje = self.reduced_mean_joint_error(xyz_true, xyz_pred)
        self._total.assign_add(tf.cast(mje, dtype=tf.float32))
        self._count.assign_add(1)

    def reduced_mean_joint_error(self, joints1, joints2):
        distances = tf.norm(joints1 - joints2, axis=2)
        return tf.reduce_mean(distances)

    def mean_joint_error(self, joints1, joints2):
        distances = tf.norm(joints1 - joints2, axis=2)
        return tf.reduce_mean(distances, axis=-1)
