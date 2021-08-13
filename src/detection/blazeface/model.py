import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, Conv2D, DepthwiseConv2D, MaxPool2D, UpSampling2D
from tensorflow.python.keras import Input

"""
See the BlazeFace paper:
https://arxiv.org/pdf/1907.05047.pdf
"""


def single_blaze_block(inputs, filters, stride=1):
    y = inputs
    x = DepthwiseConv2D((5, 5), strides=stride, padding="same")(inputs)
    x = Conv2D(filters, (1, 1), padding="same")(x)
    if stride == 2:
        y = MaxPool2D((2, 2))(y)
        y = Conv2D(filters, (1, 1), padding="same")(y)
    output = Add()([x, y])
    return Activation("relu")(output)


def double_blaze_block(inputs, filters, stride=1):
    y = inputs
    x = DepthwiseConv2D((5, 5), strides=stride, padding="same")(inputs)
    x = Conv2D(filters[0], (1, 1), padding="same")(x)
    x = Activation("relu")(x)
    x = DepthwiseConv2D((5, 5), padding="same")(x)
    x = Conv2D(filters[1], (1, 1), padding="same")(x)
    if stride == 2:
        y = MaxPool2D((2, 2))(y)
        y = Conv2D(filters[1], (1, 1), padding="same")(y)
    output = Add()([x, y])
    return Activation("relu")(output)


def concatenate_boxes(inputs, last_dimension):
    batch_size = tf.shape(inputs[0])[0]
    outputs = []
    for conv_layer in inputs:
        outputs.append(tf.reshape(conv_layer, (batch_size, -1, last_dimension)))
    #
    return tf.concat(outputs, axis=1)


def build_blaze_face(detections_per_layer, channels=1):
    # Each detection is defined by 4 values (bounding box)
    values_per_layer = tf.convert_to_tensor(detections_per_layer) * 4

    input = Input(shape=(256, 256, channels))
    first_conv = Conv2D(24, (5, 5), strides=2, padding="same", activation="relu")(input)
    single_1 = single_blaze_block(first_conv, 24)
    single_2 = single_blaze_block(single_1, 24)
    single_3 = single_blaze_block(single_2, 96, 2)
    single_4 = single_blaze_block(single_3, 96)
    single_5 = single_blaze_block(single_4, 96)  # 64x64x48
    single_6 = single_blaze_block(single_5, 96, 2)
    single_7 = single_blaze_block(single_6, 96)
    single_8 = single_blaze_block(single_7, 96)  # 32x32x96
    double_1 = double_blaze_block(single_8, [24, 96], 2)
    double_2 = double_blaze_block(double_1, [24, 96])
    double_3 = double_blaze_block(double_2, [24, 96])  # 16x16x96
    double_4 = double_blaze_block(double_3, [24, 96], 2)
    double_5 = double_blaze_block(double_4, [24, 96])
    double_6 = double_blaze_block(double_5, [24, 96])  # 8x8x96

    # Predict scale 8
    scale_8_boxes = Conv2D(values_per_layer[0], (3, 3), padding="same")(double_6)
    scale_8_confs = Conv2D(detections_per_layer[0], (3, 3), padding="same")(double_6)

    # Predict scale 16
    upsampled_16 = UpSampling2D()(double_6)
    scale_16 = Add()([double_3, upsampled_16])
    scale_16_boxes = Conv2D(values_per_layer[1], (3, 3), padding="same")(scale_16)
    scale_16_confs = Conv2D(detections_per_layer[1], (3, 3), padding="same")(scale_16)

    # Predict scale 32
    upsampled_32 = UpSampling2D()(scale_16)
    scale_32 = Add()([single_8, upsampled_32])
    scale_32_boxes = Conv2D(values_per_layer[2], (3, 3), padding="same")(scale_32)
    scale_32_confs = Conv2D(detections_per_layer[2], (3, 3), padding="same")(scale_32)

    # Predict scale 64
    upsampled_64 = UpSampling2D()(scale_32)
    scale_64 = Add()([single_5, upsampled_64])
    scale_64_boxes = Conv2D(values_per_layer[3], (3, 3), padding="same")(scale_64)
    scale_64_confs = Conv2D(detections_per_layer[3], (3, 3), padding="same")(scale_64)

    boxes = concatenate_boxes([scale_8_boxes, scale_16_boxes, scale_32_boxes, scale_64_boxes], last_dimension=4)
    confs = concatenate_boxes([scale_8_confs, scale_16_confs, scale_32_confs, scale_64_confs], last_dimension=1)
    confs = Activation('sigmoid')(confs)
    return Model(inputs=input, outputs=[boxes, confs])


if __name__ == "__main__":
    model = build_blaze_face(detections_per_layer=[6, 2, 2, 2])

    print(model.summary(line_length=150))
