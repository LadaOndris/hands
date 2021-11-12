import tensorflow as tf
from tensorflow.keras.models import Model

from src.estimation.blazepose.models.layers import BlazeBlock


class BlazePoseLight:

    def __init__(self, num_keypoints: int, num_keypoint_features: int):

        self.num_keypoints = num_keypoints
        self.num_keypoint_features = num_keypoint_features

        filters = [16, 32, 64, 128, 192]

        self.conv1_input = tf.keras.layers.Conv2D(
            filters=filters[0], kernel_size=3, strides=(2, 2), padding='same', activation='relu'
        )

        self.conv2_1_input = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=filters[0], kernel_size=1, activation=None)
        ])

        self.conv2_2_input = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=filters[0], kernel_size=1, activation=None)
        ])

        # === Heatmap ===

        self.conv3 = BlazeBlock(block_num=3, channel=filters[1])
        self.conv4 = BlazeBlock(block_num=4, channel=filters[2])
        self.conv5 = BlazeBlock(block_num=5, channel=filters[3])
        self.conv6 = BlazeBlock(block_num=6, channel=filters[4])

        self.conv7a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=filters[1], kernel_size=1, activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.conv7b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=filters[1], kernel_size=1, activation="relu")
        ])

        self.conv8a = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear")
        self.conv8b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=filters[1], kernel_size=1, activation="relu")
        ])

        self.conv9a = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear")
        self.conv9b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=filters[1], kernel_size=1, activation="relu")
        ])

        self.conv11 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=filters[1], kernel_size=1, activation="relu"),
            tf.keras.layers.Conv2D(
                filters=self.num_keypoints, kernel_size=3, padding="same", activation=None)  # -> Heatmap output
        ])

        # === Regression ===

        #  In: 1, 64, 64, 48)
        self.conv12a = BlazeBlock(block_num=4, channel=filters[2], name="regression_conv12a_")  # input res: 64
        self.conv12b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None, name="regression_conv12b_depthwise"),
            tf.keras.layers.Conv2D(
                filters=filters[2], kernel_size=1, activation="relu", name="regression_conv12b_conv1x1")
        ], name="regression_conv12b")

        self.conv13a = BlazeBlock(block_num=5, channel=filters[3], name="regression_conv13a_")  # input res: 32
        self.conv13b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None, name="regression_conv13b_depthwise"),
            tf.keras.layers.Conv2D(
                filters=filters[3], kernel_size=1, activation="relu", name="regression_conv13b_conv1x1")
        ], name="regression_conv13b")

        self.conv14a = BlazeBlock(block_num=6, channel=filters[4], name="regression_conv14a_")  # input res: 16
        self.conv14b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None, name="regression_conv14b_depthwise"),
            tf.keras.layers.Conv2D(
                filters=filters[4], kernel_size=1, activation="relu", name="regression_conv14b_conv1x1")
        ], name="regression_conv14b")

        self.conv15 = tf.keras.models.Sequential([
            BlazeBlock(block_num=7, channel=filters[4], channel_padding=0, name="regression_conv15a_"),
            BlazeBlock(block_num=7, channel=filters[4], channel_padding=0, name="regression_conv15b_")
        ], name="regression_conv15")

        joints_features = 3
        self.conv16_joints = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                filters=joints_features * self.num_keypoints, kernel_size=2, activation=None),
            tf.keras.layers.Reshape((self.num_keypoints, joints_features), name="regression_final_dense")
        ], name="joints")

        presence_features = 1
        self.conv16_presence = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                filters=presence_features * self.num_keypoints, kernel_size=2, activation=None),
            tf.keras.layers.Reshape((self.num_keypoints, presence_features), name="regression_final_dense"),
            tf.keras.layers.Activation("sigmoid")
        ], name="presence")

    def build_model(self, model_type):

        input_x = tf.keras.layers.Input(shape=(256, 256, 1))

        # Block 1
        # In: 1x256x256x1
        x = self.conv1_input(input_x)

        # Block 2
        # In: 1x128x128x16
        x = x + self.conv2_1_input(x)
        x = tf.keras.activations.relu(x)

        # Block 3
        # In: 1x128x128x16
        x = x + self.conv2_2_input(x)
        y0 = tf.keras.activations.relu(x)

        # === Heatmap ===

        # In: 1, 128, 128, 16
        y1 = self.conv3(y0)  # 64, 64, 32
        y2 = self.conv4(y1)  # 32, 32, 64
        y3 = self.conv5(y2)  # 16, 16, 128
        y4 = self.conv6(y3)  # 8, 8, 192

        x = self.conv7a(y4) + self.conv7b(y3)  # 16, 16, 32
        x = self.conv8a(x) + self.conv8b(y2)  # 32, 32, 32
        x = self.conv9a(x) + self.conv9b(y1)  # 64, 64, 32
        y = self.conv11(x)

        # In: 1, 64, 64, n_points
        heatmap = tf.keras.layers.Activation("sigmoid", name="heatmap")(y)

        # === Regression ===

        # Stop gradient for regression on 2-head model
        if model_type == "TWOHEAD" or model_type == "REGRESSION":
            x = tf.keras.backend.stop_gradient(x)
            y2 = tf.keras.backend.stop_gradient(y2)
            y3 = tf.keras.backend.stop_gradient(y3)
            y4 = tf.keras.backend.stop_gradient(y4)

        # In: [1, 64, 64, 64],  [1,  32, 32, 64]
        x = self.conv12a(x) + self.conv12b(y2)  # 32, 32, 64
        # In: [1, 32, 32, 64],
        x = self.conv13a(x) + self.conv13b(y3)  #
        # In: 1, 16, 16, 128
        x = self.conv14a(x) + self.conv14b(y4)
        # In: 1, 8, 8, 196
        x = self.conv15(x)
        # In: 1, 2, 2, 196
        joints = self.conv16_joints(x)
        presence = self.conv16_presence(x)

        if model_type == "TWOHEAD":
            return Model(inputs=input_x, outputs=[joints, heatmap, presence])
        elif model_type == "HEATMAP":
            return Model(inputs=input_x, outputs=heatmap)
        elif model_type == "REGRESSION":
            return Model(inputs=input_x, outputs=joints)
        else:
            raise ValueError("Wrong model type.")
