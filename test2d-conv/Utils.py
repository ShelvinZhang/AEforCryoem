import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep


def variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32):
    # https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/layers/python/layers/initializers.py
    if not dtype.is_floating:
        raise TypeError('Cannot create initializer for non-floating point type.')
    if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
        raise TypeError('Unknown mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)

    def _initializer(shape, dtype=dtype, partition_info=None):
        """Initializer function."""
        if not dtype.is_floating:
            raise TypeError('Cannot create initializer for non-floating point type.')
        # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
        # This is the right thing for matrix multiply and convolutions.
        if shape:
            fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
            fan_out = float(shape[-1])
        else:
            fan_in = 1.0
            fan_out = 1.0
        for dim in shape[:-2]:
            fan_in *= float(dim)
            fan_out *= float(dim)
        if mode == 'FAN_IN':
            # Count only number of input connections.
            n = fan_in
        elif mode == 'FAN_OUT':
            # Count only number of output connections.
            n = fan_out
        elif mode == 'FAN_AVG':
            # Average number of inputs and output connections.
            n = (fan_in + fan_out) / 2.0
        if uniform:
            # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
            limit = np.sqrt(3.0 * factor / n)
            return tf.random_uniform(shape, -limit, limit, dtype, seed=seed)
        else:
            # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
            trunc_stddev = np.sqrt(1.3 * factor / n)
            return tf.truncated_normal(shape, 0.0, trunc_stddev, dtype, seed=seed)

    return _initializer


def xavier_initializer(uniform=True, seed=None, dtype=tf.float32):
    return variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=uniform, seed=seed, dtype=dtype)


'''
def xavier_init(shape, constant=1):
    # shape must be a list with at least two elements
    assert len(shape) > 1
    fan_in = shape[0]
    fan_out = shape[1]
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
'''


def zscore(data, axis=0, truncate=False, scale=False):
    if axis == 1:
        data = data.transpose()
    preprocessor = prep.StandardScaler().fit(data)
    data = preprocessor.transform(data)
    if truncate:
        data = np.clip(data, -3, 3)
    if scale:
        prep.minmax_scale(data, feature_range=(-1, 1))
    if axis == 1:
        return data.transpose()
    return data, preprocessor


def min_max_scale(data):
    preprocessor = prep.MinMaxScaler().fit(data)
    data = preprocessor.transform(data)
    return data


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def montage_batch(images):
    """Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders.

    Parameters
    ----------
    batch : numpy.ndarray
        Input array to create montage of.

    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    m = np.ones(
        (images.shape[1] * n_plots + n_plots + 1,
         images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter, ...]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w, :] = this_img
    return m
