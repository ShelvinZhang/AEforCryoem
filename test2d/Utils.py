import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep


def xavier_init(shape, constant=1):
    # shape must be a list with at least two elements
    assert len(shape) > 1
    fan_in = shape[0]
    fan_out = shape[1]
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def zscore_scale(data):
    preprocessor = prep.StandardScaler().fit(data)
    data = preprocessor.transform(data)
    return data


def min_max_scale(data):
    preprocessor = prep.MinMaxScaler().fit(data)
    data = preprocessor.transform(data)
    return data


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

