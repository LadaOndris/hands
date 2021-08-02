import tensorflow as tf
from src.utils.debugging import timing

RESIZE_MODE_CROP = 'crop'
RESIZE_MODE_PAD = 'pad'


def resize_images(cropped_imgs, target_size):
    def _resize(img):
        if type(img) is tf.RaggedTensor:
            img = img.to_tensor()
        return resize_bilinear_nearest(img, target_size)

    return tf.map_fn(_resize, cropped_imgs,
                     fn_output_signature=tf.TensorSpec(shape=(target_size[0], target_size[1], 1),
                                                       dtype=tf.float32))


def tf_resize_image(depth_image, shape, resize_mode: str):
    """

    Parameters
    ----------
    depth_image
    shape
    resize_mode : str
       "crop" - Crop mode first crops the image to a square and then resizes
            with a combination of bilinear and nearest interpolation.
       "pad"  - In pad mode, the image is resized and and the rest is padded
            to retain the aspect ration of the original image.
    Returns
    -------
    """
    # convert the values to range 0-255 as tf.io.read_file does
    # depth_image = tf.image.convert_image_dtype(depth_image, dtype=tf.uint8)
    # resize image
    type = depth_image.dtype
    if resize_mode == RESIZE_MODE_PAD:
        depth_image = tf.image.resize_with_pad(depth_image, shape[0], shape[1])
    elif resize_mode == RESIZE_MODE_CROP:
        height, width, channels = depth_image.shape
        if height > width:
            offset = tf.cast((height - width) / 2, tf.int32)
            cropped = depth_image[offset:height - offset, :, :]
        elif width > height:
            offset = tf.cast((width - height) / 2, tf.int32)
            cropped = depth_image[:, offset:width - offset, :]
        else:
            cropped = depth_image
        depth_image = resize_bilinear_nearest(cropped, shape)
        # depth_image = depth_image[tf.newaxis, ...]
        # depth_image = tf.image.crop_and_resize(depth_image, [[0, 80 / 640.0, 480 / 480.0, 560 / 640.0]],
        #                                        [0], shape)
        # depth_image = depth_image[0]
    else:
        raise ValueError(F"Unknown resize mode: {resize_mode}")
    # depth_image = tf.where(depth_image > 2000, 0, depth_image)
    depth_image = tf.cast(depth_image, dtype=type)
    return depth_image

def resize_bilinear_nearest(image_in, shape):
    img_shape = tf.shape(image_in)
    img_height = img_shape[0]
    img_width = img_shape[1]

    width = shape[0]
    height = shape[1]

    image = tf.cast(image_in, tf.float32)
    image = tf.reshape(image, [-1])

    zero_constant = tf.constant(0., dtype=tf.float64)
    x_ratio = tf.cast(img_width - 1, tf.float64) / tf.cast(width - 1, tf.float64) if width > 1 else zero_constant
    y_ratio = tf.cast(img_height - 1, tf.float64) / tf.cast(height - 1, tf.float64) if height > 1 else zero_constant

    xx = tf.range(width)
    yy = tf.range(height)
    x, y = tf.meshgrid(xx, yy)
    x, y = tf.reshape(x, [-1]), tf.reshape(y, [-1])
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)

    inter_points_x = tf.cast(x_ratio * x, tf.float32)
    inter_points_y = tf.cast(y_ratio * y, tf.float32)

    x_l = tf.math.floor(inter_points_x)
    y_l = tf.math.floor(inter_points_y)
    x_h = tf.math.ceil(inter_points_x)
    y_h = tf.math.ceil(inter_points_y)

    x_weight = inter_points_x - x_l
    y_weight = inter_points_y - y_l

    x_l = tf.cast(x_l, tf.int32)
    y_l = tf.cast(y_l, tf.int32)
    x_h = tf.cast(x_h, tf.int32)
    y_h = tf.cast(y_h, tf.int32)

    a = tf.gather(image, y_l * img_width + x_l)
    b = tf.gather(image, y_l * img_width + x_h)
    c = tf.gather(image, y_h * img_width + x_l)
    d = tf.gather(image, y_h * img_width + x_h)

    bilinear = a * (1 - x_weight) * (1 - y_weight) + \
               b * x_weight * (1 - y_weight) + \
               c * y_weight * (1 - x_weight) + \
               d * x_weight * y_weight

    # Find nearest for each set of points point
    ab_nearest = tf.where(x_weight < .5, a, b)
    cd_nearest = tf.where(x_weight < .5, c, d)
    nearest = tf.where(y_weight < .5, ab_nearest, cd_nearest)

    # Find points where either of a and b is zero
    mask_is_zero = tf.math.logical_or(
        tf.math.logical_or(a == 0, b == 0),
        tf.math.logical_or(c == 0, d == 0))

    # Apply nearest interpolation
    resized = tf.where(mask_is_zero, nearest, bilinear)

    return tf.reshape(resized, [height, width, 1])


def resize_bilinear_nearest_batch(images_in, shape):
    """
    Input images are expected to be the same shape.
    """
    batch_size, img_height, img_width, channels = tf.shape(images_in)
    width, height = shape[:2]

    images = tf.cast(images_in, tf.float32)
    images = tf.reshape(images, [batch_size, -1])

    x_ratio = tf.cast(img_width - 1, tf.float64) / (width - 1) if width > 1 else 0
    y_ratio = tf.cast(img_height - 1, tf.float64) / (height - 1) if height > 1 else 0

    xx = tf.range(width)
    yy = tf.range(height)
    x, y = tf.meshgrid(xx, yy)
    x, y = tf.reshape(x, [-1]), tf.reshape(y, [-1])
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float64)

    inter_points_x = tf.cast(x_ratio * x, tf.float32)
    inter_points_y = tf.cast(y_ratio * y, tf.float32)

    x_l = tf.math.floor(inter_points_x)
    y_l = tf.math.floor(inter_points_y)
    x_h = tf.math.ceil(inter_points_x)
    y_h = tf.math.ceil(inter_points_y)

    x_weight = inter_points_x - x_l
    y_weight = inter_points_y - y_l

    x_l = tf.cast(x_l, tf.int32)
    y_l = tf.cast(y_l, tf.int32)
    x_h = tf.cast(x_h, tf.int32)
    y_h = tf.cast(y_h, tf.int32)

    a_indices = y_l * img_width + x_l
    b_indices = y_l * img_width + x_h
    c_indices = y_h * img_width + x_l
    d_indices = y_h * img_width + x_h

    # batches = tf.range(batch_size)
    a_indices = tf.tile(a_indices[tf.newaxis, :], [batch_size, 1])
    b_indices = tf.tile(b_indices[tf.newaxis, :], [batch_size, 1])
    c_indices = tf.tile(c_indices[tf.newaxis, :], [batch_size, 1])
    d_indices = tf.tile(d_indices[tf.newaxis, :], [batch_size, 1])
    a = tf.gather(images, a_indices, axis=1, batch_dims=1)
    b = tf.gather(images, b_indices, axis=1, batch_dims=1)
    c = tf.gather(images, c_indices, axis=1, batch_dims=1)
    d = tf.gather(images, d_indices, axis=1, batch_dims=1)

    bilinear = a * (1 - x_weight) * (1 - y_weight) + \
               b * x_weight * (1 - y_weight) + \
               c * y_weight * (1 - x_weight) + \
               d * x_weight * y_weight

    # Find nearest for each set of points point
    ab_nearest = tf.where(x_weight < .5, a, b)
    cd_nearest = tf.where(x_weight < .5, c, d)
    nearest = tf.where(y_weight < .5, ab_nearest, cd_nearest)

    # Find points where either of a and b is zero
    mask_is_zero = tf.math.logical_or(
        tf.math.logical_or(a == 0, b == 0),
        tf.math.logical_or(c == 0, d == 0))

    # Apply nearest interpolation
    resized = tf.where(mask_is_zero, nearest, bilinear)

    return tf.reshape(resized, [batch_size, height, width, 1])


def set_depth_unit(images, target_depth_unit, previous_depth_unit):
    """
    Converts image pixel values to the specified unit.
    """
    dtype = images.dtype
    images = tf.cast(images, dtype=tf.float32)
    images *= previous_depth_unit / target_depth_unit
    images = tf.cast(images, dtype=dtype)
    return images


def read_image_from_file(image_file_path, dtype, shape):
    """
    Loads an image from file, sets the corresponding dtype
    and shape.

    Parameters
    -------
    image_file_path
    dtype
    shape
        An array-like of two values [width, height].


    Returns
    -------
    depth_image
        A 3-D Tensor of shape [height, width, 1].
    """
    depth_image_file_content = tf.io.read_file(image_file_path)

    # loads depth images and converts values to fit in dtype.uint8
    depth_image = tf.io.decode_image(depth_image_file_content, channels=1, dtype=dtype)

    depth_image.set_shape([shape[1], shape[0], 1])
    return depth_image

@timing
def average_nonzero_depth(images):
    """
    Computes the average pixel value

    Parameters
    ----------
    images
        A 4-D Tensor of shape [batch, height, width, 1].

    Returns
    -------
    depths
        A 1-D Tensor of shape [batch].
    """
    # TODO
    #if tf.rank(images) != 4:
    #    raise TypeError(F"Invalid rank: {tf.rank(images)}, expected 4.")

    axes = [1, 2, 3]
    sums = tf.math.reduce_sum(images, axis=axes)
    counts = tf.math.count_nonzero(images, axis=axes, dtype=sums.dtype)
    mean = tf.math.divide_no_nan(sums, counts)
    return mean

#
# @tf.function(input_signature=(
#         tf.TensorSpec(shape=None, dtype=tf.int32),
#         tf.TensorSpec(shape=None, dtype=tf.int32),))
def create_coord_pairs(width, height, indexing):
    # Create all coordinate pairs
    x = tf.range(width)
    y = tf.range(height)
    xx, yy = tf.meshgrid(x, y, indexing=indexing)
    xx = tf.reshape(xx, [-1])
    yy = tf.reshape(yy, [-1])
    # Stack along a new axis to create pairs in the last dimension
    coords = tf.stack([xx, yy], axis=-1)
    coords = tf.cast(coords, tf.float32)  # [im_width * im_height, 2]
    return coords


def crop_with_pad(image, bbox):
    x_start, y_start, x_end, y_end = tf.split(bbox, [1, 1, 1, 1])

    x_start_bound = tf.maximum(x_start, 0)[0]
    y_start_bound = tf.maximum(y_start, 0)[0]
    x_end_bound = tf.minimum(image.shape[1], x_end)[0]
    y_end_bound = tf.minimum(image.shape[0], y_end)[0]

    cropped_image = image[y_start_bound:y_end_bound, x_start_bound:x_end_bound]

    # Pad the cropped image if we were out of bounds
    padded_image = tf.pad(cropped_image, [[(y_start_bound - y_start)[0], (y_end - y_end_bound)[0]],
                                          [(x_start_bound - x_start)[0], (x_end - x_end_bound)[0]],
                                          [0, 0]])
    return padded_image


def zoom_in(depth_image, bboxes, zoom=None, new_distance=None):
    #if tf.shape(bboxes)[0] == 0:
    #    raise ValueError("There is no bounding box. Unable to zoom in.")
    if zoom is None and new_distance is None:
        raise ValueError("Zoom factor and new distance cannot be both None.")

    image_shape = tf.shape(depth_image)
    bbox = tf.cast(bboxes[0], tf.int32)
    x1, x2 = bbox[0], bbox[2]
    y1, y2 = bbox[1], bbox[3]
    subimage = depth_image[y1:y2, x1:x2, :]
    mean_depth = average_nonzero_depth(subimage[tf.newaxis, ...])[0]

    if new_distance is None:
        new_distance = mean_depth / zoom
    if zoom is None:
        zoom = mean_depth / new_distance

    new_image_bbox = _bbox_of_zoomed_area(image_shape, bbox, zoom)
    cropped_image = crop_with_pad(depth_image, new_image_bbox)
    resized_image = resize_bilinear_nearest(cropped_image, image_shape)
    zoomed_image = _zoom_depth(resized_image, new_distance, mean_depth)

    # 1. Correct bboxes - amend coordinates to fit the new image
    cropped_bboxes = bboxes - tf.concat([new_image_bbox[:2], new_image_bbox[:2]], axis=-1)
    resize_coeff = image_shape[:2] / tf.shape(cropped_image)[:2]
    resized_bboxes = tf.cast(cropped_bboxes, tf.float64) * tf.concat([resize_coeff, resize_coeff], axis=-1)
    resized_bboxes = tf.cast(resized_bboxes, tf.int32)
    # 2. Check if they are out of bounds
    bboxes_start = tf.math.maximum(resized_bboxes[:, :2], tf.constant([0, 0]))
    bboxes_end = tf.math.minimum(resized_bboxes[:, 2:], image_shape[:2] - 1)
    cropped_bboxes = tf.concat([bboxes_start, bboxes_end], axis=-1)
    # remove too narrow boxes because of the crop
    ious = _intersection_over_box1(bboxes, new_image_bbox)
    bboxes_mask_indices = tf.where(ious >= 0.5)
    zoomed_bboxes = tf.gather_nd(cropped_bboxes, bboxes_mask_indices)

    return zoomed_image, zoomed_bboxes


def _intersection_over_box1(box1, box2):
    box1 = tf.cast(box1, dtype=tf.float32)
    box2 = tf.cast(box2, dtype=tf.float32)

    box1_wh = box1[..., 2:] - box1[..., :2]
    box2_wh = box2[..., 2:] - box2[..., :2]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]

    left_up = tf.maximum(box1[..., :2], box2[..., :2])
    right_down = tf.minimum(box1[..., 2:], box2[..., 2:])
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    return tf.math.divide_no_nan(inter_area, box1_area)


def _bbox_of_zoomed_area(image_shape, hand_bbox, zoom):
    x1, x2 = hand_bbox[0], hand_bbox[2]
    y1, y2 = hand_bbox[1], hand_bbox[3]
    new_size = tf.cast(image_shape[:2], dtype=tf.float32) / zoom
    bbox_size = [x2 - x1, y2 - y1]
    # Random position of the zoomed hand
    center = [(x1 + x2) / 2, (y1 + y2) / 2]
    delta = (new_size - bbox_size) / 2
    p1 = center - delta
    p2 = center + delta

    new_image_center = tf.random.uniform(shape=[2], minval=p1, maxval=p2)
    new_image_p1 = new_image_center - new_size / 2
    new_image_p2 = new_image_center + new_size / 2
    new_image_p1 = tf.cast(new_image_p1, tf.int32)
    new_image_p2 = tf.cast(new_image_p2, tf.int32)
    new_image_bbox = tf.concat([new_image_p1, new_image_p2], axis=-1)
    return new_image_bbox


def _zoom_depth(image, new_depth, previous_depth):
    zoom_distance = new_depth - previous_depth
    zoomed_image = tf.where(image > 0, image + zoom_distance, 0)
    return zoomed_image


if __name__ == "__main__":
    # Check bilinear nearest resizing
    im = tf.zeros([5, 194, 195, 1])
    res = resize_bilinear_nearest_batch(im, [96, 96])
    tf.print(res.shape)
