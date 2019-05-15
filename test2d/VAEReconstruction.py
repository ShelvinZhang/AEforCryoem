import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import scipy.io as io

from Model import VariationalAutoencoder
import Utils
import mrcfile


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Directory for storing data')
flags.DEFINE_string('file_name', 'test.mrcs', 'Data file name')
flags.DEFINE_integer('training_epochs', 5000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_float('starter_learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('decay_epochs', 2000, 'Learning rate decay epochs')
flags.DEFINE_float('decay_rate', 0.1, 'Learning rate decay rate')
flags.DEFINE_integer('display_step', 1, 'Display step')
flags.DEFINE_list('plot_step', [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], 'Step to plot the picture')
flags.DEFINE_integer('n_plot', 5, 'Number of pictures to plot')
flags.DEFINE_string('summaries_dir', './summary', 'Summaries directory')


def main(*args):
    raw = mrcfile.open(FLAGS.data_dir + FLAGS.file_name)
    header = raw.header
    order = shuffle(np.array(range(header.NZ)), random_state=1)
    x_train = Utils.zscore_scale(raw.data.reshape(-1, header.NX * header.NY))
    x_train = Utils.min_max_scale(x_train)[order]
    del raw

    n_samples = header.NZ
    total_batch = int(n_samples / FLAGS.batch_size)
    decay_steps = FLAGS.decay_epochs * total_batch
    vae = VariationalAutoencoder(FLAGS)

    vae.restore()
    z_mean, z_log_sigma, z = vae.encode(x_train)
    zz = np.mean(z_mean, axis=0, keepdims=True)
    fig_manifold, ax_manifold = plt.subplots(1, 1)
    imgs = []
    j = 9
    for img_j in np.linspace(zz[0][j]-3, zz[0][j]+3, 20):
        b = zz
        b[0][j] = img_j
        recon = vae.generate(b)
        imgs.append(np.reshape(recon, (1, 112, 112, 1)))
    imgs_cat = np.concatenate(imgs)
    ax_manifold.imshow(Utils.montage_batch(imgs_cat))
    fig_manifold.savefig('manifold.png')


if __name__ == '__main__':
    tf.app.run()