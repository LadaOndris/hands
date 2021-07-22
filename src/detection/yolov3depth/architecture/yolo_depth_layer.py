import tensorflow as tf

"""
The implementation of the YoloLayer is inspired 
by the implementation of YunYang1994 published under the MIT license:
YunYang1994. Tensorflow-yolov3 [online]. GitHub, 2020 [cit. 2020-8-10].
Available at: https://github.com/YunYang1994/tensorflow-yolov3
"""


class YoloDepthLayer(tf.keras.layers.Layer):
    def __init__(self, anchors, n_classes, input_layer_shape, name=None):
        super(YoloDepthLayer, self).__init__(name=name)
        self.anchors = tf.convert_to_tensor(anchors)
        self.n_anchors = len(anchors)
        self.n_classes = n_classes
        self.input_layer_shape = input_layer_shape

    def call(self, inputs_and_images):
        """
        Reshapes inputs to [batch_size, grid_size, grid_size, anchors_per_grid, 6]
        where the axis=-1 contains [x, y, w, h, conf, raw_conf].

        Parameters
        ----------
        inputs : 
            Outputs of previous layer in the model.

        Returns
        -------
        yolo_outputs : Tensor of shape [batch_size, grid_size, grid_size, anchors_per_grid, 5 + n_classes]
            Returns raw predictions [tx, ty, tw, th].
            It is ready for loss calculation, but needs to gor through further postprocessing 
            to convert it to real dimensions.
        """
        # transform to [None, B * grid size * grid size, 5 + C]
        # The B is the number of anchors and C is the number of classes.
        # inputs = tf.reshape(inputs, [-1, self.n_anchors * out_shape[1] * out_shape[2], \
        #                             5 + self.n_classes])
        inputs, images = inputs_and_images
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        grid_size_y = inputs_shape[1]
        grid_size_x = inputs_shape[2]
        reshaped_inputs = tf.reshape(inputs, [batch_size, grid_size_y, grid_size_x,
                                              self.n_anchors, 5 + self.n_classes])

        # for example: 416 x 416 pixel images, 13 x 13 tiles
        # 416 // 13 = 32
        stride = (self.input_layer_shape[1] // grid_size_y,
                  self.input_layer_shape[2] // grid_size_x)
        scaled_anchors = self.scale_anchors(images, stride[0])

        # extract information
        box_centers = reshaped_inputs[..., 0:2]
        box_shapes = reshaped_inputs[..., 2:4]
        confidence = reshaped_inputs[..., 4:5]

        # create coordinates for each anchor for each cell
        y = tf.tile(tf.range(grid_size_y, dtype=tf.int32)[:, tf.newaxis], [1, grid_size_y])
        x = tf.tile(tf.range(grid_size_x, dtype=tf.int32)[tf.newaxis, :], [grid_size_x, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, self.n_anchors, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # get dimensions in pixels instead of grid boxes by multiplying with stride
        pred_xy = (tf.sigmoid(box_centers) + xy_grid) * stride
        pred_wh = (tf.exp(box_shapes) * scaled_anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(confidence)  # confidence is objectness

        return tf.concat([pred_xywh, pred_conf, confidence], axis=-1)

    def scale_anchors(self, images, stride):
        mean_depths = self.cells_mean_depths(images, stride)  # (batch_size, out, out)
        base_anchor = self.anchors  # (anchors_per_scale, 2)
        scaled_depths = tf.math.divide_no_nan(100., mean_depths)
        scaled_depths = scaled_depths[:, :, :, tf.newaxis, tf.newaxis]  # (batch_size, out, out, 1, 1)
        base_anchor = base_anchor[tf.newaxis, tf.newaxis, tf.newaxis, :, :]  # (1, 1, 1, anchors_per_scale, 2)
        base_anchor = tf.cast(base_anchor, dtype=tf.float32)
        scaled_anchors = base_anchor * scaled_depths
        return scaled_anchors

    def cells_mean_depths(self, images, cell_stride):
        images_shape = tf.shape(images)
        batch_size = images_shape[0]
        rows = images_shape[1]
        cell_stride = tf.cast(cell_stride, dtype=tf.int32)

        cells_in_row = tf.cast(tf.math.divide(rows, cell_stride), dtype=tf.int32)
        cells_total = tf.cast(tf.math.divide(tf.math.pow(rows, 2), tf.math.pow(cell_stride, 2)), dtype=tf.int32)

        imgs_transformed = tf.reshape(images, [batch_size, rows, cells_in_row, cell_stride])
        imgs_transformed = tf.transpose(imgs_transformed, [0, 2, 1, 3])
        imgs_transformed = tf.reshape(imgs_transformed, [batch_size, cells_total, cell_stride ** 2])

        sums_of_squares = tf.cast(tf.reduce_sum(imgs_transformed, axis=-1), dtype=tf.float32)
        nonzeros_in_squares = tf.cast(tf.math.count_nonzero(imgs_transformed, axis=-1), dtype=tf.float32)

        mean_depth = tf.math.divide_no_nan(sums_of_squares, nonzeros_in_squares)
        mean_depth = tf.reshape(mean_depth, [batch_size, cells_in_row, cells_in_row])
        mean_depth = tf.transpose(mean_depth, [0, 2, 1])
        return mean_depth
